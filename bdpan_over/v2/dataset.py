import os
import random
import paddle
import paddle.vision.transforms.functional as F
from paddle.io import Dataset
from paddle.vision.transforms import RandomCrop, ToTensor

import numpy as np
from PIL import Image
import numbers


def _check_input(value,
                 name,
                 center=1,
                 bound=(0, float('inf')),
                 clip_first_on_zero=True):
    if isinstance(value, numbers.Number):
        if value < 0:
            raise ValueError(
                "If {} is a single number, it must be non negative.".format(
                    name))
        value = [center - value, center + value]
        if clip_first_on_zero:
            value[0] = max(value[0], 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError("{} values should be between {}".format(name,
                                                                     bound))
    else:
        raise TypeError(
            "{} should be a single number or a list/tuple with lenght 2.".
                format(name))

    if value[0] == value[1] == center:
        value = None
    return value


class OverDataset(Dataset):

    def __init__(self,
                 data_dir,
                 data_size,
                 img_size=512,
                 # big image aug
                 patch_count=[0, 1, 2],
                 patch_count_p=[0.05, 0.7, 0.25],
                 # crop image aug
                 dense_crop_p=None,
                 dense_crop_rate=0.1,
                 dense_crop_max_count=10,
                 flip_p=None,
                 scale_p=None,
                 scale_range=(0.75, 1.25),
                 use_hsv=False,
                 cache_img_period=None,
                 cache_max_size=128,
                 is_val=False,
                 is_to_tensor=True,
                 ):
        super(OverDataset, self).__init__()
        self.im_dir = os.path.join(data_dir, 'images')
        self.patch_dir = os.path.join(data_dir, 'patches')
        self.data_size = data_size
        self.img_size = img_size
        # big image aug
        self.patch_count = patch_count
        self.patch_count_p = patch_count_p
        # crop image aug
        self.dense_crop_p = dense_crop_p
        self.dense_crop_rate = dense_crop_rate
        self.dense_crop_max_count = dense_crop_max_count
        self.flip_p = flip_p
        self.scale_p = scale_p
        self.scale_range = scale_range
        self.use_hsv = use_hsv
        self.cache_img_period = cache_img_period
        self.cache_max_size = cache_max_size
        self.is_val = is_val
        self.is_to_tensor = is_to_tensor
        self.patch_info = self._get_patch_info()
        self.im_info = self._get_im_info()
        self.cache = {}

        self.random_crop = RandomCrop(img_size, fill=0)
        self.color_brightness_value = _check_input(0.4, 'brightness')
        self.color_contrast_value = _check_input(0.4, 'contrast')
        self.color_saturation_value = _check_input(0.7, 'saturation')
        self.color_hue_value = _check_input(0.015, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.to_tensor = ToTensor()

    def _get_patch_info(self):
        ret = []
        with open(os.path.join(self.patch_dir, 'patches.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = list(map(lambda x: x.strip(), line.split(',')))
            new_items = []
            new_items.append(os.path.join(self.patch_dir, 'images', items[0]))
            new_items.append(os.path.join(self.patch_dir, 'labels', items[0]))
            new_items.append(int(items[1]))
            new_items.append(int(items[2]))
            ret.append(new_items)
        return ret

    def _get_im_info(self):
        return list(map(lambda x: os.path.join(self.im_dir, x), os.listdir(self.im_dir)))

    def _get_random_im_filename(self):
        rnd_idx = random.randint(0, len(self.im_info) - 1)
        im_path = self.im_info[rnd_idx]
        return os.path.basename(im_path)

    def _get_random_im(self):
        rnd_idx = random.randint(0, len(self.im_info) - 1)
        im_path = self.im_info[rnd_idx]
        return np.array(Image.open(im_path)), os.path.basename(im_path)

    def _get_random_patch(self):
        rnd_idx = random.randint(0, len(self.patch_info) - 1)
        patch_path, mask_path, patch_cls, patch_pos = self.patch_info[rnd_idx]
        patch_im = np.array(Image.open(patch_path))
        patch_mask = np.array(Image.open(mask_path))
        return patch_im, patch_mask, patch_cls, patch_pos

    def _get_item_from_cache(self, im_filename):
        try:
            item = self.cache[im_filename]
            im = item[0]
            gt = item[1]
            mask = item[2]
            left_count = item[3]
            if left_count - 1 <= 0:
                try:
                    del self.cache[im_filename]
                    # print(f'cache del {im_filename}')
                except:
                    pass
            else:
                item[3] = left_count - 1
            return im, gt, mask
        except:
            return None

    def _get_fuse_item(self):
        def patch_flip(patch_im, mask_im, p_cls, p_pos):
            if p_cls == 2:
                return patch_im, mask_im, p_pos
            if random.random() < 0.5:
                return patch_im, mask_im, p_pos
            if p_pos == 0:
                patch_im = patch_im[::-1, :, :]
                mask_im = mask_im[::-1, :]
                return patch_im, mask_im, 2
            if p_pos == 1:
                patch_im = patch_im[:, ::-1, :]
                mask_im = mask_im[:, ::-1]
                return patch_im, mask_im, 3
            if p_pos == 2:
                patch_im = patch_im[::-1, :, :]
                mask_im = mask_im[::-1, :]
                return patch_im, mask_im, 0
            if p_pos == 3:
                patch_im = patch_im[:, ::-1, :]
                mask_im = mask_im[:, ::-1]
                return patch_im, mask_im, 1
            raise NotImplemented()

        def calc_start_point(h, w, p_h, p_w, p_pos):
            if p_pos == 0:
                return 0, random.randint(0, w - p_w)
            if p_pos == 1:
                return random.randint(0, h - p_h), w - p_w
            if p_pos == 2:
                return h - p_h, random.randint(0, w - p_w)
            if p_pos == 3:
                return random.randint(0, h - p_h), 0
            return 0, 0

        im, im_filename = self._get_random_im()
        gt = im.copy()
        h, w = im.shape[0], im.shape[1]
        im_mask = np.zeros((h, w), dtype=np.uint8)
        patch_count = random.choices(self.patch_count, self.patch_count_p, k=1)[0]
        if patch_count > 0:
            for i in range(patch_count):
                im_mask_tmp = np.zeros((h, w), dtype=np.uint8)
                patch_im, patch_mask, patch_cls, patch_pos = self._get_random_patch()
                p_h, p_w = patch_im.shape[0], patch_im.shape[1]
                if p_h >= h or p_w >= w:
                    continue
                patch_im, patch_mask, patch_pos = patch_flip(patch_im, patch_mask, patch_cls, patch_pos)
                paste_y, paste_x = calc_start_point(h, w, p_h, p_w, patch_pos)
                im_mask_tmp[paste_y:paste_y + p_h, paste_x:paste_x + p_w] = patch_mask
                im[:, :, 0][im_mask_tmp == 1] = patch_im[:, :, 0][patch_mask == 1]
                im[:, :, 1][im_mask_tmp == 1] = patch_im[:, :, 1][patch_mask == 1]
                im[:, :, 2][im_mask_tmp == 1] = patch_im[:, :, 2][patch_mask == 1]
                im_mask = im_mask + im_mask_tmp
            im_mask[im_mask > 1] = 1
        if self.cache_img_period is not None:
            if len(self.cache) > self.cache_max_size:
                current_keys = list(self.cache.keys())
                random.shuffle(current_keys)
                del self.cache[current_keys[0]]
                # print(f'remove cache {current_keys[0]}')
            self.cache[im_filename] = [im, gt, im_mask, self.cache_img_period]
        return im, gt, im_mask

    def _is_dense_crop(self, mask_img):
        mask_arr = np.array(mask_img)
        mask_rate = np.sum(mask_arr > 0) / (self.img_size * self.img_size)
        if mask_rate > self.dense_crop_rate:
            return True
        return False

    def _trans_horizontal_flip(self, image_img, gt_img, mask_img):
        if self.flip_p is not None:
            if random.random() < self.flip_p:
                image_img = F.hflip(image_img)
                gt_img = F.hflip(gt_img)
                mask_img = F.hflip(mask_img)
        return image_img, gt_img, mask_img

    def _trans_scale(self, image_img, gt_img, mask_img):
        if self.scale_p is not None and random.random() < self.scale_p:
            rnd_scale = self.scale_range[0] + random.random() * (self.scale_range[1] - self.scale_range[0])
            im_w, im_h = image_img.width, image_img.height
            g_w, g_h = gt_img.width, gt_img.height
            m_w, m_h = mask_img.width, mask_img.height
            assert im_w == g_w and im_h == g_h and im_w == m_w and im_h == m_h
            new_w = int(im_w * rnd_scale)
            new_h = int(im_h * rnd_scale)
            image_img = F.resize(image_img, (new_h, new_w))
            gt_img = F.resize(gt_img, (new_h, new_w))
            mask_img = F.resize(mask_img, (new_h, new_w))
        return image_img, gt_img, mask_img

    def _trans_crop(self, image_img, gt_img, mask_img):
        im_w, im_h = image_img.width, image_img.height
        g_w, g_h = gt_img.width, gt_img.height
        m_w, m_h = mask_img.width, mask_img.height
        assert im_w == g_w and im_h == g_h and im_w == m_w and im_h == m_h
        if self.dense_crop_p is None:
            max_crop_count = 1
        elif random.random() > self.dense_crop_p:
            max_crop_count = 1
        else:
            max_crop_count = self.dense_crop_max_count

        if im_w < self.img_size:
            pad_w = ((self.img_size - im_w) + 1) // 2
            image_img = F.pad(image_img, (pad_w, 0, pad_w, 0), fill=(0, 0, 0))
            gt_img = F.pad(gt_img, (pad_w, 0, pad_w, 0), fill=(0, 0, 0))
            mask_img = F.pad(mask_img, (pad_w, 0, pad_w, 0), fill=0)
        if im_h < self.img_size:
            pad_h = ((self.img_size - im_h) + 1) // 2
            image_img = F.pad(image_img, (0, pad_h, 0, pad_h), fill=(0, 0, 0))
            gt_img = F.pad(gt_img, (0, pad_h, 0, pad_h), fill=(0, 0, 0))
            mask_img = F.pad(mask_img, (0, pad_h, 0, pad_h), fill=0)

        for _ in range(max_crop_count):
            param = self.random_crop._get_param(image_img, (self.img_size, self.img_size))
            crop_image_img = F.crop(image_img, *param)
            crop_gt_img = F.crop(gt_img, *param)
            crop_mask_img = F.crop(mask_img, *param)
            if self.dense_crop_p is not None and max_crop_count > 1 and self._is_dense_crop(crop_mask_img):
                break
        image_img, gt_img, mask_img = crop_image_img, crop_gt_img, crop_mask_img
        return image_img, gt_img, mask_img

    def _trans_hsv(self, image_img, gt_img, mask_img):
        if self.use_hsv:
            brightness_factor = random.uniform(self.color_brightness_value[0], self.color_brightness_value[1])
            contrast_factor = random.uniform(self.color_contrast_value[0], self.color_contrast_value[1])
            saturation_factor = random.uniform(self.color_saturation_value[0], self.color_saturation_value[1])
            hue_factor = random.uniform(self.color_hue_value[0], self.color_hue_value[1])
            image_img = F.adjust_brightness(image_img, brightness_factor)
            image_img = F.adjust_contrast(image_img, contrast_factor)
            image_img = F.adjust_saturation(image_img, saturation_factor)
            image_img = F.adjust_hue(image_img, hue_factor)
            gt_img = F.adjust_brightness(gt_img, brightness_factor)
            gt_img = F.adjust_contrast(gt_img, contrast_factor)
            gt_img = F.adjust_saturation(gt_img, saturation_factor)
            gt_img = F.adjust_hue(gt_img, hue_factor)
        return image_img, gt_img, mask_img

    def __getitem__(self, idx):
        im, gt, mask = None, None, None
        im_filename = self._get_random_im_filename()
        if self.cache_img_period is not None:
            if im_filename in self.cache.keys():
                item = self._get_item_from_cache(im_filename)
                if item is not None:
                    im, gt, mask = item
                    # print('shot cache')
        if im is None:
            im, gt, mask = self._get_fuse_item()
        im = Image.fromarray(im)
        gt = Image.fromarray(gt)
        mask = Image.fromarray(mask)
        if not self.is_val:
            im, gt, mask = self._trans_scale(im, gt, mask)
            im, gt, mask = self._trans_crop(im, gt, mask)
            im, gt, mask = self._trans_horizontal_flip(im, gt, mask)
            im, gt, mask = self._trans_hsv(im, gt, mask)
        if self.is_to_tensor:
            im = self.to_tensor(im)
            gt = self.to_tensor(gt)
            mask = np.array(mask, dtype=np.long)
            has_mask = 1 if np.sum(mask) >= 1 else 0
            mask = paddle.to_tensor(mask)
            has_mask = paddle.squeeze(paddle.to_tensor(np.array(has_mask, dtype=np.long)))
        else:
            im = np.array(im)
            gt = np.array(gt)
            mask = np.array(mask)
            has_mask = 1 if np.sum(mask) >= 1 else 0
            has_mask = np.array(has_mask)
        return im, gt, mask, has_mask

    def __len__(self):
        return self.data_size


class OverTestDataset(Dataset):

    def __init__(self, data_dir):
        super(OverTestDataset, self).__init__()
        self.data_dir = data_dir
        self.filepath_list = list(map(lambda x: os.path.join(self.data_dir, x), os.listdir(self.data_dir)))
        self.to_tensor = ToTensor()

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        im = Image.open(filepath)
        return self.to_tensor(im), filepath

    def __len__(self):
        return len(self.filepath_list)








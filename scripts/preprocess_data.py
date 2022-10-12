import os
import shutil

import numpy as np
from PIL import Image


def process_original_dataset(root_dir, out_dir):

    def cal_patch_side(h, w, min_h, max_h, min_w, max_w):
        gap_top = abs(min_h - 0)
        gap_bottom = abs(h - max_h)
        gap_left = abs(min_w - 0)
        gap_right = abs(w - max_w)
        ready_sort = [
            (gap_top, 0),
            (gap_right, 1),
            (gap_bottom, 2),
            (gap_left, 3),
        ]
        ready_sort = sorted(ready_sort, key=lambda x: x[0])
        return ready_sort[0][1]

    def process_patch(im_path, out_dir, im_info_list, cls):
        for im_filename in list(filter(lambda x: x.endswith('.jpg'), os.listdir(im_path))):
            lb_filename = im_filename.replace('.jpg', '.png')
            im_arr = np.array(Image.open(os.path.join(im_path, im_filename)))
            lb_arr = np.array(Image.open(os.path.join(im_path, lb_filename)))
            h_axis, w_axis = np.where(lb_arr == 1)
            min_h_axix, max_h_axis = np.min(h_axis), np.max(h_axis)
            min_w_axix, max_w_axis = np.min(w_axis), np.max(w_axis)
            clip_im_arr = im_arr[min_h_axix:max_h_axis, min_w_axix:max_w_axis]
            clip_lb_arr = lb_arr[min_h_axix:max_h_axis, min_w_axix:max_w_axis]
            clip_im_arr = np.array(Image.fromarray(clip_im_arr).convert('RGBA'))
            clip_im_arr[:, :, 3][clip_lb_arr == 0] = 0
            final_im_filename = f'cls_{cls}_{im_filename.replace(".jpg", ".png")}'
            final_lb_filename = f'cls_{cls}_{lb_filename}'
            Image.fromarray(clip_im_arr).save(os.path.join(out_dir, 'patches', 'images', final_im_filename))
            Image.fromarray(clip_lb_arr).save(os.path.join(out_dir, 'patches', 'labels', final_lb_filename))
            im_info_list.append([
                final_im_filename,
                cls,
                str(cal_patch_side(
                    im_arr.shape[0], im_arr.shape[1],
                    min_h_axix, max_h_axis, min_w_axix, max_w_axis))
            ])

    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'patches', 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'patches', 'labels'), exist_ok=True)
    im_info_list = []
    for dirname in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dirname)
        if os.path.isdir(dir_path):
            gt_path = os.path.join(dir_path, 'gt')
            hand_path = os.path.join(dir_path, 'hand')
            head_path = os.path.join(dir_path, 'head')
            for filename in os.listdir(gt_path):
                shutil.copy(os.path.join(gt_path, filename), os.path.join(out_dir, 'images', filename))
            # for im_filename in list(filter(lambda x: x.endswith('.jpg'), os.listdir(hand_path))):
            #     lb_filename = im_filename.replace('.jpg', '.png')
            #     im_arr = np.array(Image.open(os.path.join(hand_path, im_filename)))
            #     lb_arr = np.array(Image.open(os.path.join(hand_path, lb_filename)))
            #     h_axis, w_axis = np.where(lb_arr == 1)
            #     min_h_axix, max_h_axis = np.min(h_axis), np.max(h_axis)
            #     min_w_axix, max_w_axis = np.min(w_axis), np.max(w_axis)
            #     clip_im_arr = im_arr[min_h_axix:max_h_axis, min_w_axix:max_w_axis]
            #     clip_lb_arr = lb_arr[min_h_axix:max_h_axis, min_w_axix:max_w_axis]
            #     clip_im_arr = np.array(Image.fromarray(clip_im_arr).convert('RGBA'))
            #     clip_im_arr[:, :, 3][clip_lb_arr == 0] = 0
            #     Image.fromarray(clip_im_arr).save(os.path.join(out_dir, 'patches', 'images', im_filename.replace('.jpg', '.png')))
            #     Image.fromarray(clip_lb_arr).save(os.path.join(out_dir, 'patches', 'labels', lb_filename))
            #     im_info_list.append([
            #         im_filename.replace('.jpg', '.png'),
            #         '1'
            #     ])
            process_patch(hand_path, out_dir, im_info_list, '1')
            process_patch(head_path, out_dir, im_info_list, '2')
    with open(os.path.join(out_dir, 'patches', 'patches.txt'), 'w+') as f:
        for im_info in im_info_list:
            f.write(','.join(im_info))
            f.write('\n')


if __name__ == '__main__':
    process_original_dataset(
        r'F:\BaiduNetdiskDownload\bdpan_over\train',
        r'F:\BaiduNetdiskDownload\bdpan_over_processed'
    )
    print(10086)

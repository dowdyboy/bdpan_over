import os
import sys
import glob
import cv2
from PIL import Image
import numpy as np
from paddle.io import DataLoader
import paddle

from dowdyboy_lib.assist import calc_time_used
from bdpan_over.v4.dataset import OverTestDataset
from bdpan_over.v4.model import OverRestoreModelV4, OverClassifyModelV4

assert len(sys.argv) == 3

src_image_dir = sys.argv[1]
save_dir = sys.argv[2]


def pd_tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    img = tensor.squeeze().cpu().numpy()
    img = img.clip(min_max[0], min_max[1])
    img = (img - min_max[0]) / (min_max[1] - min_max[0])
    if out_type == np.uint8:
        # scaling
        img = img * 255.0
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
    img = img.round()
    # img = img[:, :, ::-1]  # to BGR2RGB
    return img.astype(out_type)


def build_data():
    test_dataset = OverTestDataset(src_image_dir)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=0, drop_last=False)
    return test_loader, test_dataset


def build_model():
    restore_model = OverRestoreModelV4(is_pretrain=False)
    cls_model = OverClassifyModelV4(is_pretrain=False)
    return restore_model, cls_model


def test_step(restore_model, cls_model, bat):
    bat_im, bat_filepath, _ = bat
    pad = 12
    m = paddle.nn.Pad2D(pad, mode='reflect')
    bat_im = m(bat_im)
    _, _, h, w = bat_im.shape
    step = 1000
    res = paddle.zeros_like(bat_im)

    cls_count = 0
    all_count = 0

    for i in range(0, h, step):
        for j in range(0, w, step):
            all_count += 1
            if h - i < step + 2 * pad:
                i = h - (step + 2 * pad)
            if w - j < step + 2 * pad:
                j = w - (step + 2 * pad)
            clip = bat_im[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
            cls_out = cls_model(clip)
            cls_pred = paddle.argmax(cls_out, axis=1)[0].item()
            if cls_pred == 0:
                res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = clip[:, :, pad:-pad, pad:-pad]
                cls_count += 1
            else:
                restore_pred = restore_model(clip)
                res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = restore_pred[:, :, pad:-pad, pad:-pad]

    # print(f'{cls_count} / {all_count}')
    res = res[:, :, pad:-pad, pad:-pad]
    res = pd_tensor2img(res)
    Image.fromarray(res).save(
        os.path.join(save_dir, os.path.basename(bat_filepath[0]))
    )


def main():
    test_loader, test_dataset = build_data()
    restore_chk_path = 'checkpoint/v4/restore_epoch_681/model_0.pd'
    cls_chk_path = 'checkpoint/v4/cls_epoch_335/model_0.pd'
    restore_model, cls_model = build_model()
    restore_model.load_dict(paddle.load(restore_chk_path))
    cls_model.load_dict(paddle.load(cls_chk_path))
    restore_model.eval()
    cls_model.eval()

    for bat in test_loader:
        with paddle.no_grad():
            test_step(restore_model, cls_model, bat)
            # s = calc_time_used(test_step, [restore_model, cls_model, bat], count=1)
            # print(f'time used : {s} sec')


if __name__ == "__main__":

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main()


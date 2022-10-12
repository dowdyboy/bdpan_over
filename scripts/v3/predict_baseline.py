import os
import sys
import shutil
import cv2
from PIL import Image
import numpy as np
from paddle.io import DataLoader
import paddle

assert len(sys.argv) == 3

src_image_dir = sys.argv[1]
save_dir = sys.argv[2]


if __name__ == "__main__":

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # for filename in os.listdir(src_image_dir):
    #     im_path = os.path.join(src_image_dir, filename)
    #     shutil.copy(im_path, os.path.join(save_dir, filename))

    for filename in os.listdir(src_image_dir):
        im_path = os.path.join(src_image_dir, filename)
        im = Image.open(im_path)
        ori_w, ori_h = im.size
        new_w, new_h = int(ori_w * 0.5), int(ori_h * 0.5)
        im = im.resize((new_w, new_h), Image.BILINEAR)
        im = im.resize((ori_w, ori_h), Image.BILINEAR)
        im.save(os.path.join(save_dir, filename))


import paddle
import paddle.nn as nn

from PIL import Image
import numpy as np


class RestoreSegmentLoss(nn.Layer):
    
    def __init__(self, step_thr):
        super(RestoreSegmentLoss, self).__init__()
        self.step_thr = step_thr
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

    def _loss_restore(self, pred, gt):
        loss_mse = self.mse(pred, gt)
        loss_l1 = self.l1(pred, gt)
        return loss_mse + loss_l1 * 0.5

    def forward(self, step, pred_im, bat_gt, pred_mask, bat_mask):
        if step < self.step_thr:
            loss_restore = self._loss_restore(pred_im, bat_gt)
            pred_mask = paddle.transpose(pred_mask, [0, 2, 3, 1])
            loss_seg = self.ce(pred_mask, bat_mask)
            loss = loss_restore + loss_seg
        else:
            seg_im = pred_mask.numpy()
            seg_im = paddle.to_tensor(seg_im)
            seg_im = paddle.argmax(seg_im, axis=1)
            seg_im = paddle.stack([seg_im, seg_im, seg_im], axis=1)
            seg_im = seg_im.astype('float32')
            loss_over = self._loss_restore(pred_im * seg_im, bat_gt * seg_im)
            loss_no_over = self._loss_restore(pred_im * (1. - seg_im), bat_gt * (1. - seg_im))
            pred_mask = paddle.transpose(pred_mask, [0, 2, 3, 1])
            loss_seg = self.ce(pred_mask, bat_mask)
            loss = loss_over + 0.1 * loss_no_over + loss_seg
        return loss


class ClassifyLoss(nn.Layer):

    def __init__(self):
        super(ClassifyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        return self.ce(pred, gt)


# class RestoreLoss(nn.Layer):
#
#     def __init__(self):
#         super(RestoreLoss, self).__init__()
#         self.mse = nn.MSELoss()
#         self.l1 = nn.L1Loss()
#
#     def forward(self, pred, gt):
#         mse_loss = self.mse(pred, gt)
#         l1_loss = self.l1(pred, gt)
#         return mse_loss + l1_loss
#
#
# class SegLoss(nn.Layer):
#
#     def __init__(self):
#         super(SegLoss, self).__init__()
#         self.ce = nn.CrossEntropyLoss()
#
#     def forward(self, pred, gt):
#         return self.ce(pred, gt)

import paddle
import paddle.nn as nn


class RestoreLoss(nn.Layer):

    def __init__(self):
        super(RestoreLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, gt):
        mse_loss = self.mse(pred, gt)
        l1_loss = self.l1(pred, gt)
        return mse_loss + l1_loss


class ClassifyLoss(nn.Layer):

    def __init__(self):
        super(ClassifyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        return self.ce(pred, gt)


class SegLoss(nn.Layer):

    def __init__(self):
        super(SegLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        return self.ce(pred, gt)

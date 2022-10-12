import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import ConvNormActivation
from .shufflenetv2 import ShuffleNetV2
from functools import partial


class SqueezeExpandLayer(nn.Layer):

    def __init__(self, in_channel, out_channel, split_count):
        super(SqueezeExpandLayer, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.split_count = split_count
        assert out_channel % split_count == 0
        self.squeeze_dp_conv = nn.Conv2D(in_channel, in_channel, 3, 1, 1, groups=in_channel, )
        self.expand_convs = nn.LayerList([
            ConvNormActivation(
                in_channel, out_channel // split_count,
                kernel_size=1, stride=1, padding=0,
            )
            for _ in range(split_count)
        ])

    def forward(self, x):
        outs = []
        x = self.squeeze_dp_conv(x)
        for expand_conv in self.expand_convs:
            outs.append(
                expand_conv(x)
            )
        out = paddle.concat(outs, axis=1)
        return out


class TransConvLayer(nn.Layer):

    def __init__(self, in_channel, out_channel, split=4):
        super(TransConvLayer, self).__init__()
        self.conv_a = SqueezeExpandLayer(in_channel, in_channel, split)
        self.final_conv = nn.Conv2D(in_channel, out_channel, 1, 1, 0, )

    def forward(self, x):
        x = x + self.conv_a(x)
        x = self.final_conv(x)
        return x


# TODO: 是不是可以先下采样后再提取特征？
class DownSampleConvLayer(nn.Layer):

    def __init__(self, in_channel, out_channel):
        super(DownSampleConvLayer, self).__init__()
        self.conv_a = SqueezeExpandLayer(in_channel, 2 * in_channel, 4)
        self.conv_b = SqueezeExpandLayer(2 * in_channel, 4 * in_channel, 2)
        self.conv_c = SqueezeExpandLayer(4 * in_channel, 2 * in_channel, 4)
        self.conv_d = SqueezeExpandLayer(2 * in_channel, in_channel, 2)
        self.mp = nn.MaxPool2D(3, 2, 1)
        self.last_conv = nn.Conv2D(in_channel, out_channel, 1, 1, 0, )

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(a)
        c = a + self.conv_c(b)
        d = self.conv_d(c)
        x = x + d
        x = self.mp(x)
        x = self.last_conv(x)
        return x


class OverModelClassifyHeadV1(nn.Layer):
    
    def __init__(self, in_channel=192, num_classes=2):
        super(OverModelClassifyHeadV1, self).__init__()
        self.last_conv = ConvNormActivation(
            in_channels=in_channel,
            out_channels=in_channel * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            norm_layer=partial(nn.BatchNorm2D, epsilon=0.001, momentum=0.99),
            activation_layer=nn.Hardswish)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channel * 4, in_channel * 2),
            nn.Hardswish(),
            nn.Dropout(p=0.15),
            nn.Linear(in_channel * 2, num_classes))

    def forward(self, feat):
        out = self.last_conv(feat)
        out = self.avg_pool(out)
        out = paddle.flatten(out, 1)
        out = self.classifier(out)
        return out


class OverModelEncoderV1(nn.Layer):

    def __init__(self, feats_out_channel=[24, 24, 48, 96, 192], num_classes=2, is_pretrain=True):
        super(OverModelEncoderV1, self).__init__()
        self.shuffle25_net = ShuffleNetV2(scale=0.25, act='relu', num_classes=num_classes, with_pool=True)
        self.shuffle33_net = ShuffleNetV2(scale=0.33, act='relu', num_classes=num_classes, with_pool=True)
        self.shuffle50_net = ShuffleNetV2(scale=0.5, act='relu', num_classes=num_classes, with_pool=True)
        self.shuffle100_net = ShuffleNetV2(scale=1.0, act='relu', num_classes=num_classes, with_pool=True)
        if is_pretrain:
            self._load_pretrain()

        self.shuffle25_select_feats = [0, 1, 5, 13, 17]
        self.shuffle33_select_feats = [0, 1, 5, 13, 17]
        self.shuffle50_select_feats = [0, 1, 5, 13, 17]
        self.shuffle100_select_feats = [0, 1, 5, 13, 17]

        shuffle25_feats_in_channel = [24, 24, 24, 48, 96]
        shuffle33_feats_in_channel = [24, 24, 32, 64, 128]
        shuffle50_feats_in_channel = [24, 24, 48, 96, 192]
        shuffle100_feats_in_channel = [24, 24, 116, 232, 464]

        all_feats_out_channel = feats_out_channel
        # all_feats_out_channel = [24, 24, 48, 96, 192]
        # all_feats_out_channel = [8, 16, 32, 48, 64]

        self.shuffle25_trans_convs = nn.LayerList([
            TransConvLayer(in_channel, all_feats_out_channel[i])
            for i, in_channel in enumerate(shuffle25_feats_in_channel)
        ])
        self.shuffle33_trans_convs = nn.LayerList([
            TransConvLayer(in_channel, all_feats_out_channel[i])
            for i, in_channel in enumerate(shuffle33_feats_in_channel)
        ])
        self.shuffle50_trans_convs = nn.LayerList([
            TransConvLayer(in_channel, all_feats_out_channel[i])
            for i, in_channel in enumerate(shuffle50_feats_in_channel)
        ])
        self.shuffle100_trans_convs = nn.LayerList([
            TransConvLayer(in_channel, all_feats_out_channel[i])
            for i, in_channel in enumerate(shuffle100_feats_in_channel)
        ])
        self.downsample_convs = nn.LayerList()
        for i in range(len(all_feats_out_channel) - 1):
            self.downsample_convs.append(
                DownSampleConvLayer(all_feats_out_channel[i], all_feats_out_channel[i + 1])
            )

    def _load_pretrain(self):
        #######
        net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_25.pdparams'
        net_weight = paddle.load(net_weight_path)
        self.shuffle25_net.set_state_dict(net_weight)
        print(f'successful load {net_weight_path}')
        #######
        net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_33.pdparams'
        net_weight = paddle.load(net_weight_path)
        self.shuffle33_net.set_state_dict(net_weight)
        print(f'successful load {net_weight_path}')
        #######
        net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_5.pdparams'
        net_weight = paddle.load(net_weight_path)
        self.shuffle50_net.set_state_dict(net_weight)
        print(f'successful load {net_weight_path}')
        #######
        net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x1_0.pdparams'
        net_weight = paddle.load(net_weight_path)
        self.shuffle100_net.set_state_dict(net_weight)
        print(f'successful load {net_weight_path}')

    def forward(self, x):
        out25, feats25 = self.shuffle25_net.extract_feats(x)
        out33, feats33 = self.shuffle33_net.extract_feats(x)
        out50, feats50 = self.shuffle50_net.extract_feats(x)
        out100, feats100 = self.shuffle100_net.extract_feats(x)
        shuffle25_feats = []
        shuffle33_feats = []
        shuffle50_feats = []
        shuffle100_feats = []
        for i, idx in enumerate(self.shuffle25_select_feats):
            shuffle25_feats.append(self.shuffle25_trans_convs[i](feats25[idx]))
        for i, idx in enumerate(self.shuffle33_select_feats):
            shuffle33_feats.append(self.shuffle33_trans_convs[i](feats33[idx]))
        for i, idx in enumerate(self.shuffle50_select_feats):
            shuffle50_feats.append(self.shuffle50_trans_convs[i](feats50[idx]))
        for i, idx in enumerate(self.shuffle100_select_feats):
            shuffle100_feats.append(self.shuffle100_trans_convs[i](feats100[idx]))
        fuse_feats = []
        for i in range(len(shuffle33_feats)):
            fuse_feats.append(
                shuffle33_feats[i] + shuffle50_feats[i] + shuffle25_feats[i] + shuffle100_feats[i]
            )
        final_fuse_feats = []
        fuse_x = fuse_feats[0]
        final_fuse_feats.append(fuse_x)
        for i in range(1, len(fuse_feats)):
            fuse_x = self.downsample_convs[i - 1](fuse_x) + fuse_feats[i]
            final_fuse_feats.append(fuse_x)
        return final_fuse_feats, \
               [shuffle25_feats, shuffle33_feats, shuffle50_feats, shuffle100_feats], \
                [out25, out33, out50, out100]


class UpSampleConvLayer(nn.Layer):

    def __init__(self, in_channel, out_channel, ):
        super(UpSampleConvLayer, self).__init__()
        self.conv_a = SqueezeExpandLayer(in_channel, in_channel // 2, 4)
        self.conv_b = SqueezeExpandLayer(in_channel // 2, in_channel // 2, 2)
        self.conv_c = SqueezeExpandLayer(in_channel // 2, in_channel // 2, 2)
        self.conv_d = SqueezeExpandLayer(in_channel // 2, in_channel, 4)
        self.t_conv = nn.Conv2DTranspose(in_channel, in_channel,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         groups=in_channel)
        self.p_conv = ConvNormActivation(in_channel, out_channel, 1, 1, 0, )

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(a) + a
        c = self.conv_c(b) + b
        d = self.conv_d(c)
        x = x + d
        x = self.t_conv(x)
        x = self.p_conv(x)
        return x


class OverModelDecoderV1(nn.Layer):

    def __init__(self, feats_in_channel=[24, 24, 48, 96, 192], out_channel=16, ):
        super(OverModelDecoderV1, self).__init__()
        feats_in_channel = list(reversed(feats_in_channel))
        self.up_convs = nn.LayerList()
        for i in range(len(feats_in_channel) - 1):
            self.up_convs.append(
                UpSampleConvLayer(
                    feats_in_channel[i] if i == 0 else feats_in_channel[i] * 2,
                    feats_in_channel[i + 1],
                )
            )
        self.final_up_conv = UpSampleConvLayer(feats_in_channel[-1] * 2, out_channel)

    def forward(self, feats):
        feats = list(reversed(feats))
        out = self.up_convs[0](feats[0])
        for i in range(1, len(self.up_convs)):
            out = paddle.concat([out, feats[i]], axis=1)
            out = self.up_convs[i](out)
        out = paddle.concat([out, feats[-1]], axis=1)
        out = self.final_up_conv(out)
        return out


class OverModelSegHeadV1(nn.Layer):

    def __init__(self, in_channel=16, num_classes=2, use_drop=True,):
        super(OverModelSegHeadV1, self).__init__()
        self.use_drop = use_drop
        self.conv_a = ConvNormActivation(in_channel, in_channel // 2, 1, 1, 0, )
        self.conv_b = ConvNormActivation(in_channel // 2, in_channel // 2, 3, 1, 1, )
        self.conv_c = ConvNormActivation(in_channel // 2, in_channel, 1, 1, 0, )
        self.drop = nn.Dropout(p=0.15)
        self.cls_conv = nn.Conv2D(in_channel, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(a) + a
        c = self.conv_c(b)
        x = x + c
        if self.use_drop:
            x = self.drop(x)
        x = self.cls_conv(x)
        return x


class OverModelV1(nn.Layer):

    def __init__(self, feats_channel=[24, 24, 48, 96, 192], out_channel=16,
                 is_pretrain=True, fuse_weight=5.5):
        super(OverModelV1, self).__init__()
        self.fuse_weight = fuse_weight
        self.encoder = OverModelEncoderV1(
            feats_out_channel=feats_channel,
            num_classes=2,
            is_pretrain=is_pretrain,
        )
        self.decoder_fuse = OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel)
        self.decoder_list = nn.LayerList([
            OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel),
            OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel),
            OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel),
            OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel)
        ])
        # self.decoder_25 = OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel)
        # self.decoder_33 = OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel)
        # self.decoder_50 = OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel)
        # self.decoder_100 = OverModelDecoderV1(feats_in_channel=feats_channel, out_channel=out_channel)
        self.head_cls = OverModelClassifyHeadV1(in_channel=feats_channel[-1], num_classes=2)
        self.head_seg = OverModelSegHeadV1(in_channel=out_channel, num_classes=2, use_drop=True)
        self.head_img = OverModelSegHeadV1(in_channel=out_channel, num_classes=3, use_drop=False)

    def forward(self, x):
        fuse_feats, shuffle_feats_list, shuffle_out_list = self.encoder(x)
        fuse_out = self.head_cls(fuse_feats[-1]) * self.fuse_weight
        out_list = [fuse_out]
        out_list.extend(shuffle_out_list)
        fuse_dec_out = self.decoder_fuse(fuse_feats) * self.fuse_weight
        dec_out_list = [fuse_dec_out]
        for i in range(len(self.decoder_list)):
            dec_out_list.append(self.decoder_list[i](shuffle_feats_list[i]))
        dec_out = paddle.sum(paddle.stack(dec_out_list, axis=0), axis=0)
        # dec_out = paddle.sum(dec_out_list)
        seg_out = self.head_seg(dec_out)
        img_out = self.head_img(dec_out)
        cls_out = paddle.sum(paddle.stack(out_list, axis=0), axis=0)
        img_out = paddle.clip(img_out, min=0., max=1.)
        return img_out, seg_out, cls_out


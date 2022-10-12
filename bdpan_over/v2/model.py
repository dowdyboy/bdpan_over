import paddle
import paddle.nn as nn
from .shufflenetv2 import ShuffleNetV2
from paddle.vision.ops import ConvNormActivation


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


class OverModelEncoderV2(nn.Layer):

    def __init__(self,
                 scales=['0.25', '0.33', '0.5', '1.0'],
                 feats_out_channel=[24, 24, 48, 96, 192],
                 num_classes=2,
                 is_pretrain=True,
                 ):
        super(OverModelEncoderV2, self).__init__()
        self.scales = scales
        self.shuffle_net_list = nn.LayerList()
        self._create_models(scales, num_classes)
        if is_pretrain:
            self._load_pretrain()
        self.shuffle_select_feats = [0, 1, 5, 13, 17]
        self.shuffle_feats_in_channel = list(map(lambda x: self._get_shuffle_feats_in_channel(x), scales))
        self.shuffle_feats_out_channel = feats_out_channel
        self.shuffle_trans_convs = self._create_trans_convs()
        self.down_sample_convs = self._create_down_convs()

    def _create_models(self, scale_list, num_classes):
        for scale in scale_list:
            assert scale in ['swish', '0.25', '0.33', '0.5', '1.0', '1.5', '2.0']
            if scale != 'swish':
                self.shuffle_net_list.append(
                    ShuffleNetV2(scale=float(scale), act='relu', num_classes=num_classes, with_pool=True)
                )
            else:
                self.shuffle_net_list.append(
                    ShuffleNetV2(scale=1.0, act='swish', num_classes=num_classes, with_pool=True)
                )

    def _load_pretrain(self):
        for i in range(len(self.shuffle_net_list)):
            if self.scales[i] == '0.25':
                net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_25.pdparams'
            elif self.scales[i] == '0.33':
                net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_33.pdparams'
            elif self.scales[i] == '0.5':
                net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_5.pdparams'
            elif self.scales[i] == '1.0':
                net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x1_0.pdparams'
            elif self.scales[i] == '1.5':
                net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x1_5.pdparams'
            elif self.scales[i] == '2.0':
                net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x2_0.pdparams'
            elif self.scales[i] == 'swish':
                net_weight_path = 'checkpoint/pretrain/shufflenet_v2_swish.pdparams'
            else:
                raise NotImplementedError()
            net_weight = paddle.load(net_weight_path)
            self.shuffle_net_list[i].set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')

    def _get_shuffle_feats_in_channel(self, scale):
        if scale == '0.25':
            return [24, 24, 24, 48, 96]
        elif scale == '0.33':
            return [24, 24, 32, 64, 128]
        elif scale == '0.5':
            return [24, 24, 48, 96, 192]
        elif scale == '1.0':
            return [24, 24, 116, 232, 464]
        elif scale == '1.5':
            return [24, 24, 176, 352, 704]
        elif scale == '2.0':
            return [24, 24, 224, 488, 976]
        elif scale == 'swish':
            return [24, 24, 116, 232, 464]
        else:
            raise NotImplementedError()

    def _create_trans_convs(self):
        ret = nn.LayerList()
        for idx in range(len(self.scales)):
            ret.append(
                nn.LayerList([
                    TransConvLayer(in_channel, self.shuffle_feats_out_channel[i])
                    for i, in_channel in enumerate(self.shuffle_feats_in_channel[idx])
                ])
            )
        return ret

    def _create_down_convs(self):
        down_convs = nn.LayerList()
        for i in range(len(self.shuffle_feats_out_channel) - 1):
            down_convs.append(
                DownSampleConvLayer(self.shuffle_feats_out_channel[i], self.shuffle_feats_out_channel[i + 1])
            )
        return down_convs

    def forward(self, x):
        out_list = []
        feats_list = []
        for i in range(len(self.scales)):
            out, feats = self.shuffle_net_list[i].extract_feats(x)
            out_list.append(out)
            feats_list.append(feats)
        trans_feats_list = []
        for net_i in range(len(self.scales)):
            trans_feats = []
            for i, idx in enumerate(self.shuffle_select_feats):
                trans_feats.append(
                    self.shuffle_trans_convs[net_i][i](feats_list[net_i][idx])
                )
            trans_feats_list.append(trans_feats)
        fuse_feats = []
        for i in range(len(self.shuffle_select_feats)):
            f = None
            for net_i in range(len(self.scales)):
                if f is None:
                    f = trans_feats_list[net_i][i]
                else:
                    f += trans_feats_list[net_i][i]
            fuse_feats.append(f)

        final_fuse_feats = []
        fuse_x = fuse_feats[0]
        final_fuse_feats.append(fuse_x)
        for i in range(1, len(fuse_feats)):
            fuse_x = self.down_sample_convs[i - 1](fuse_x) + fuse_feats[i]
            final_fuse_feats.append(fuse_x)

        return final_fuse_feats, \
               trans_feats_list, \
               out_list


class OverModelDecoderV2(nn.Layer):

    def __init__(self, feats_in_channel=[24, 24, 48, 96, 192], out_channel=16, ):
        super(OverModelDecoderV2, self).__init__()
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


class OverModelHeadV2(nn.Layer):

    def __init__(self, in_channel=16, num_classes=2, use_drop=True,):
        super(OverModelHeadV2, self).__init__()
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


class OverClassifyModelV2(nn.Layer):

    def __init__(self, num_classes=2, is_pretrain=True):
        super(OverClassifyModelV2, self).__init__()
        self.shuffle50_net = ShuffleNetV2(scale=0.5, act='relu', num_classes=num_classes, with_pool=True)
        if is_pretrain:
            net_weight_path = 'checkpoint/pretrain/shufflenet_v2_x0_5.pdparams'
            net_weight = paddle.load(net_weight_path)
            self.shuffle50_net.set_state_dict(net_weight)
            print(f'successful load {net_weight_path}')

    def forward(self, x):
        return self.shuffle50_net(x)


class OverSegmentModelV2(nn.Layer):

    def __init__(self, is_pretrain=True):
        super(OverSegmentModelV2, self).__init__()
        self.encoder = OverModelEncoderV2(
            scales=['0.25', '0.33', '0.5'],
            feats_out_channel=[24, 24, 32, 64, 128],
            num_classes=2,
            is_pretrain=is_pretrain,
        )
        self.decoder = OverModelDecoderV2(
            feats_in_channel=[24, 24, 32, 64, 128],
            out_channel=16,
        )
        self.head = OverModelHeadV2(
            in_channel=16,
            num_classes=2,
            use_drop=True,
        )

    def forward(self, x):
        fuse_feats, _, _ = self.encoder(x)
        fuse_dec_out = self.decoder(fuse_feats)
        out = self.head(fuse_dec_out)
        return out


class OverRestoreModelV2(nn.Layer):

    def __init__(self, is_pretrain=True):
        super(OverRestoreModelV2, self).__init__()
        self.encoder = OverModelEncoderV2(
            scales=['1.0', '1.5', '2.0'],
            feats_out_channel=[24, 24, 176, 352, 704],
            num_classes=2,
            is_pretrain=is_pretrain,
        )
        self.decoder = OverModelDecoderV2(
            feats_in_channel=[24, 24, 176, 352, 704],
            out_channel=16,
        )
        self.head = OverModelHeadV2(
            in_channel=16,
            num_classes=3,
            use_drop=False,
        )

    def forward(self, x):
        fuse_feats, _, _ = self.encoder(x)
        fuse_dec_out = self.decoder(fuse_feats)
        out = self.head(fuse_dec_out)
        out = paddle.clip(out, min=0., max=1.)
        return out


class OverModelV2(nn.Layer):

    def __init__(self):
        super(OverModelV2, self).__init__()
        self.cls_model = OverClassifyModelV2()
        self.seg_model = OverSegmentModelV2()
        self.restore_mode = OverRestoreModelV2()

    def forward(self, x):
        out_restore = self.restore_mode(x)
        out_seg = self.seg_model(x)
        out_cls = self.cls_model(x)
        return out_restore, out_seg, out_cls


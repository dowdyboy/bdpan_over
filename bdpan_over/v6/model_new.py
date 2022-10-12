import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial
import math


def to_2tuple(x):
    return tuple([x] * 2)

def swapdim(x, dim1, dim2):
    a = list(range(len(x.shape)))
    a[dim1], a[dim2] = a[dim2], a[dim1]

    return x.transpose(a)

def drop_path(x, drop_prob = 0., training = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.to_tensor(keep_prob) + paddle.rand(shape)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output

class DropPath(nn.Layer):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = swapdim(x, 1, 2)
        x = x.reshape([B, C, H, W])
        x = self.dwconv(x)
        x = x.flatten(2)
        x = swapdim(x, 1, 2)

        return x


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2D(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2D(7)
            self.sr = nn.Conv2D(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
                x_ = self.sr(x_).reshape([B, C, -1]).transpose([0, 2, 1])
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
            else:
                kv = self.kv(x).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        else:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            x_ = self.sr(self.pool(x_)).reshape([B, C, -1]).transpose([0, 2, 1])
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape([B, -1, 2, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        attn = (q @ swapdim(k, -2, -1)) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = swapdim((attn @ v), 1, 2).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Layer):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2)
        x = swapdim(x, 1, 2)
        x = self.norm(x)
        return x, H, W


class PyramidVisionTransformerV2(nn.Layer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x for x in paddle.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            # 首次4倍下采样
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            # 首次2倍下采样
            # patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** i),
            #                                 patch_size=7 if i == 0 else 3,
            #                                 stride=2 if i == 0 else 2,
            #                                 in_chans=in_chans if i == 0 else embed_dims[i - 1],
            #                                 embed_dim=embed_dims[i])

            block = nn.LayerList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else Identity()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        feats = []
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
                feats.append(x)
            else:
                feats.append(x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2]))
        return x.mean(axis=1), feats

    def forward(self, x):
        x, _ = self.forward_features(x)
        x = self.head(x)
        return x


def pvt_v2_b0(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b1(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b2(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    return model


def pvt_v2_b3(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b4(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b5(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pvt_v2_b2_li(**kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True,
        **kwargs)
    return model


########################


class OverlapUpSampleEmbed(nn.Layer):

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2DTranspose(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2),
                                       output_padding=stride - patch_size[0] % 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2)
        x = swapdim(x, 1, 2)
        x = self.norm(x)
        return x, H, W


class DecoderBlock(nn.Layer):

    def __init__(self, in_channel, mlp_ratio=4,
                 block_count=3,
                 block_conv_ratios=[2, 2, 2],
                 block_conv_kspd=[
                     [3, 1, 1, 1],
                     [5, 1, 4, 2],
                     [7, 1, 12, 4]
                 ],
                 ffn_act=False,
                 ):
        super(DecoderBlock, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(
                in_channel,
                in_channel * mlp_ratio,
                ),
            nn.ReLU() if ffn_act else nn.Identity(),
            nn.Linear(
                in_channel * mlp_ratio,
                in_channel,
                ),
        )
        self.convs = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(in_channel, int(in_channel * block_conv_ratios[i]),
                          kernel_size=block_conv_kspd[i][0],
                          stride=block_conv_kspd[i][1],
                          padding=block_conv_kspd[i][2],
                          dilation=block_conv_kspd[i][3]),
                nn.BatchNorm2D(int(in_channel * block_conv_ratios[i])),
                nn.ReLU(),
                nn.Conv2D(int(in_channel * block_conv_ratios[i]), in_channel,
                          kernel_size=1, stride=1, padding=0),
            )
            for i in range(block_count)
        ])

    def forward(self, x, H, W):
        B = x.shape[0]
        x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
        for conv in self.convs:
            x = x + conv(x)
        x = x.transpose([0, 2, 3, 1]).reshape([B, H * W, -1])
        x = self.ffn(x)
        return x


class PyramidVisionTransformerV2Decoder(nn.Layer):

    def __init__(self, out_chans=16, embed_dims=[256, 160, 64, 32],
                 mlp_ratios=[4, 4, 4, 4], mlp_act=False,
                 norm_layer=nn.LayerNorm, num_stages=4, ):
        super(PyramidVisionTransformerV2Decoder, self).__init__()
        self.num_stages = num_stages

        for i in range(num_stages):
            patch_embed = OverlapUpSampleEmbed(patch_size=7 if i == (num_stages - 1) else 3,
                                            stride=4 if i == (num_stages - 1) else 2,
                                            in_chans=embed_dims[i] if i == 0 else embed_dims[i] * 2,
                                            embed_dim=out_chans if i == (num_stages-1) else embed_dims[i + 1])

            # block = nn.LayerList([
            #     nn.Linear(
            #         out_chans if i == (num_stages-1) else embed_dims[i + 1],
            #         out_chans * mlp_ratios[i] if i == (num_stages-1) else embed_dims[i + 1] * mlp_ratios[i]
            #     ),
            #     nn.Linear(
            #         out_chans * mlp_ratios[i] if i == (num_stages-1) else embed_dims[i + 1] * mlp_ratios[i],
            #         out_chans if i == (num_stages-1) else embed_dims[i + 1],
            #     ),
            # ])
            block = DecoderBlock(
                in_channel=out_chans if i == (num_stages-1) else embed_dims[i + 1],
                mlp_ratio=mlp_ratios[i], ffn_act=mlp_act,
            )
            norm = norm_layer(
                out_chans if i == (num_stages-1) else embed_dims[i + 1]
            )

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, feats):
        feats = list(reversed(feats))
        B = feats[0].shape[0]
        x = feats[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            if i != 0:
                x = paddle.concat([x, feats[i]], axis=1)
            x, H, W = patch_embed(x)
            # for k, blk in enumerate(block):
            #     x = blk(x)
            x = block(x, H, W)
            x = norm(x)
            x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])

        return x


class OverModelHeadV6(nn.Layer):

    def __init__(self, in_channel=16, num_classes=2, use_drop=True,):
        super(OverModelHeadV6, self).__init__()
        self.use_drop = use_drop
        self.before_conv = nn.Conv2D(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout(p=0.15)
        self.cls_conv = nn.Conv2D(in_channel, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = x + self.before_conv(x)
        if self.use_drop:
            x = self.drop(x)
        x = self.cls_conv(x)
        return x


class OverRestoreSegmentModelV6(nn.Layer):

    def __init__(self,
                 img_size,
                 embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[8, 8, 4, 4],
                 mlp_act=False,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
                 depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_chans=16,
                 use_sigmoid=False,
                 ):
        super(OverRestoreSegmentModelV6, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.encoder = PyramidVisionTransformerV2(
            embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias,
            norm_layer=norm_layer, depths=depths, sr_ratios=sr_ratios,
            img_size=img_size, num_classes=0)
        self.decoder = PyramidVisionTransformerV2Decoder(
            out_chans=out_chans,
            embed_dims=list(reversed(embed_dims)),
            mlp_ratios=list(reversed(mlp_ratios)),
            mlp_act=mlp_act,
            norm_layer=norm_layer,
        )
        self.x_point_conv = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)
        )
        self.restore_head = OverModelHeadV6(in_channel=out_chans, num_classes=3, use_drop=False)
        self.seg_head = OverModelHeadV6(in_channel=out_chans, num_classes=2, use_drop=True)

    def forward(self, x):
        _, feats = self.encoder.forward_features(x)
        out = self.decoder(feats)
        out_seg = self.seg_head(out)
        out_restore = self.restore_head(out)
        out_restore = out_restore + self.x_point_conv(x)
        if self.use_sigmoid:
            out_restore = F.sigmoid(out_restore)
        else:
            out_restore = paddle.clip(out_restore, min=0., max=1.)
        return out_restore, out_seg


def over_model_b0(img_size, encoder_pretrain=None, use_sigmoid=False, mlp_act=False):
    model = OverRestoreSegmentModelV6(
        embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        img_size=img_size,
        use_sigmoid=use_sigmoid,
        mlp_act=mlp_act,
    )
    if encoder_pretrain is not None:
        model.encoder.set_state_dict(paddle.load(encoder_pretrain))
        print(f'success load {encoder_pretrain}')
    return model


def over_model_b2(img_size, encoder_pretrain=None, use_sigmoid=False, mlp_act=False):
    model = OverRestoreSegmentModelV6(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        img_size=img_size, out_chans=32,
        use_sigmoid=use_sigmoid,
        mlp_act=mlp_act,
    )
    if encoder_pretrain is not None:
        model.encoder.set_state_dict(paddle.load(encoder_pretrain))
        print(f'success load {encoder_pretrain}')
    return model


def over_model_b4(img_size, encoder_pretrain=None, use_sigmoid=False, mlp_act=False):
    model = OverRestoreSegmentModelV6(
        embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        img_size=img_size, out_chans=48,
        use_sigmoid=use_sigmoid,
        mlp_act=mlp_act,
    )
    if encoder_pretrain is not None:
        model.encoder.set_state_dict(paddle.load(encoder_pretrain))
        print(f'success load {encoder_pretrain}')
    return model



######################


# class PyramidVisionTransformerV2Decoder(nn.Layer):
#
#     def __init__(self, out_chans=16, embed_dims=[256, 160, 64, 32],
#                  num_heads=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
#                  depths=[3, 6, 4, 3], sr_ratios=[1, 2, 4, 8], num_stages=4, linear=False):
#         super(PyramidVisionTransformerV2Decoder, self).__init__()
#         self.depths = depths
#         self.num_stages = num_stages
#
#         dpr = [x for x in paddle.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#         cur = 0
#
#         for i in range(num_stages):
#             patch_embed = OverlapUpSampleEmbed(patch_size=7 if i == (num_stages - 1) else 3,
#                                                stride=4 if i == (num_stages - 1) else 2,
#                                                in_chans=embed_dims[i] if i == 0 else embed_dims[i] * 2,
#                                                embed_dim=out_chans if i == (num_stages-1) else embed_dims[i + 1])
#
#             block = nn.LayerList([Block(
#                 dim=out_chans if i == (num_stages-1) else embed_dims[i + 1],
#                 num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
#                 sr_ratio=sr_ratios[i], linear=linear)
#                 for j in range(depths[i])])
#             norm = norm_layer(
#                 out_chans if i == (num_stages-1) else embed_dims[i + 1]
#             )
#             cur += depths[i]
#
#             setattr(self, f"patch_embed{i + 1}", patch_embed)
#             setattr(self, f"block{i + 1}", block)
#             setattr(self, f"norm{i + 1}", norm)
#
#     def forward_features(self, x):
#         feats = []
#         B = x.shape[0]
#         for i in range(self.num_stages):
#             patch_embed = getattr(self, f"patch_embed{i + 1}")
#             block = getattr(self, f"block{i + 1}")
#             norm = getattr(self, f"norm{i + 1}")
#             x, H, W = patch_embed(x)
#             for blk in block:
#                 x = blk(x, H, W)
#             x = norm(x)
#             if i != self.num_stages - 1:
#                 x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
#                 feats.append(x)
#             else:
#                 feats.append(x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2]))
#         return x.mean(axis=1), feats
#
#     def forward(self, feats):
#         feats = list(reversed(feats))
#         B = feats[0].shape[0]
#         x = feats[0]
#         for i in range(self.num_stages):
#             patch_embed = getattr(self, f"patch_embed{i + 1}")
#             block = getattr(self, f"block{i + 1}")
#             norm = getattr(self, f"norm{i + 1}")
#             if i != 0:
#                 x = paddle.concat([x, feats[i]], axis=1)
#             x, H, W = patch_embed(x)
#             for k, blk in enumerate(block):
#                 x = blk(x, H, W)
#             x = norm(x)
#             x = x.reshape([B, H, W, -1]).transpose([0, 3, 1, 2])
#
#         return x




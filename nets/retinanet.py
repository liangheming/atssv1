import torch
import math
from torch import nn
from nets.common import CR, FPN, CBR

default_anchor_sizes = [32., 64., 128., 256., 512.]
default_strides = [8, 16, 32, 64, 128]
default_anchor_scales = [1., ]
default_anchor_ratios = [1.]


def switch_backbones(bone_name):
    from nets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, \
        resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
    if bone_name == "resnet18":
        return resnet18()
    elif bone_name == "resnet34":
        return resnet34()
    elif bone_name == "resnet50":
        return resnet50()
    elif bone_name == "resnet101":
        return resnet101()
    elif bone_name == "resnet152":
        return resnet152()
    elif bone_name == "resnext50_32x4d":
        return resnext50_32x4d()
    elif bone_name == "resnext101_32x8d":
        return resnext101_32x8d()
    elif bone_name == "wide_resnet50_2":
        return wide_resnet50_2()
    elif bone_name == "wide_resnet101_2":
        return wide_resnet101_2()
    else:
        raise NotImplementedError(bone_name)


class Scale(nn.Module):
    def __init__(self, init_val=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(data=init_val), requires_grad=True)

    def forward(self, x):
        return x * self.scale


class RetinaClsHead(nn.Module):
    def __init__(self, in_channel,
                 inner_channel=256,
                 num_anchors=9, num_cls=80, num_layers=4):
        super(RetinaClsHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_cls = num_cls
        self.bones = list()
        for i in range(num_layers):
            if i == 0:
                conv = CBR(in_channel, inner_channel, 3, 1)
            else:
                conv = CBR(inner_channel, inner_channel, 3, 1)
            self.bones.append(conv)
        self.bones = nn.Sequential(*self.bones)
        self.cls = nn.Conv2d(inner_channel, self.num_anchors * self.num_cls, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        x = self.bones(x)
        x = self.cls(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, self.num_cls) \
            .view(bs, -1, self.num_cls)
        return x


class RetinaRegHead(nn.Module):
    def __init__(self, in_channel, inner_channel=256, num_anchors=9, num_layers=4):
        super(RetinaRegHead, self).__init__()
        self.num_anchors = num_anchors
        self.bones = list()
        for i in range(num_layers):
            if i == 0:
                conv = CBR(in_channel, inner_channel, 3, 1)
            else:
                conv = CBR(inner_channel, inner_channel, 3, 1)
            self.bones.append(conv)
        self.bones = nn.Sequential(*self.bones)
        self.reg = nn.Conv2d(inner_channel, self.num_anchors * 4, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bones(x)
        x = self.reg(x)
        bs, _, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous() \
            .view(bs, h, w, self.num_anchors, 4) \
            .view(x.size(0), -1, 4)
        return x


class RetinaHead(nn.Module):
    def __init__(self, in_channel,
                 inner_channel,
                 num_cls=80,
                 num_convs=4,
                 layer_num=5,
                 anchor_sizes=None,
                 anchor_scales=None,
                 anchor_ratios=None,
                 strides=None):
        super(RetinaHead, self).__init__()
        self.num_cls = num_cls
        self.layer_num = layer_num
        if anchor_sizes is None:
            anchor_sizes = default_anchor_sizes
        self.anchor_sizes = anchor_sizes
        if anchor_scales is None:
            anchor_scales = default_anchor_scales
        self.anchor_scales = anchor_scales
        if anchor_ratios is None:
            anchor_ratios = default_anchor_ratios
        self.anchor_ratios = anchor_ratios
        if strides is None:
            strides = default_strides
        self.strides = strides
        self.anchor_nums = len(self.anchor_scales) * len(self.anchor_ratios)
        self.scales = nn.ModuleList([Scale(init_val=1.0) for _ in range(self.layer_num)])
        self.anchors = [torch.zeros(size=(0, 4))] * self.layer_num
        self.register_buffer("std", torch.tensor([0.1, 0.1, 0.2, 0.2]).float())

        self.cls_head = RetinaClsHead(in_channel, inner_channel, self.anchor_nums, num_cls, num_convs)
        self.reg_head = RetinaRegHead(in_channel, inner_channel, self.anchor_nums, num_convs)

    def build_anchors_delta(self, size=32.):
        """
        :param size:
        :return: [anchor_num, 4]
        """
        scales = torch.tensor(self.anchor_scales).float()
        ratio = torch.tensor(self.anchor_ratios).float()
        scale_size = (scales * size)
        w = (scale_size[:, None] * ratio[None, :].sqrt()).view(-1) / 2
        h = (scale_size[:, None] / ratio[None, :].sqrt()).view(-1) / 2
        delta = torch.stack([-w, -h, w, h], dim=1)
        return delta

    def build_anchors(self, feature_maps):
        """
        :param feature_maps:
        :return: list(anchor) anchor:[all,4] (x1,y1,x2,y2)
        """
        assert self.layer_num == len(feature_maps)
        assert len(self.anchor_sizes) == len(feature_maps)
        assert len(self.anchor_sizes) == len(self.strides)
        anchors = list()
        for stride, size, feature_map in zip(self.strides, self.anchor_sizes, feature_maps):
            # 9*4
            anchor_delta = self.build_anchors_delta(size)
            _, _, ny, nx = feature_map.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            # h,w,4
            grid = torch.stack([xv, yv, xv, yv], 2).float()
            anchor = (grid[:, :, None, :] + 0.5) * stride + anchor_delta[None, None, :, :]
            anchor = anchor.view(-1, 4)
            anchors.append(anchor)
        return anchors

    def forward(self, xs):
        cls_outputs = list()
        reg_outputs = list()
        for j, x in enumerate(xs):
            cls_outputs.append(self.cls_head(x))
            reg_outputs.append(self.scales[j](self.reg_head(x)))
        if self.anchors[0] is None or self.anchors[0].shape[0] != cls_outputs[0].shape[1]:
            with torch.no_grad():
                anchors = self.build_anchors(xs)
                assert len(anchors) == len(self.anchors)
                for i, anchor in enumerate(anchors):
                    self.anchors[i] = anchor.to(xs[0].device)
        if self.training:
            return cls_outputs, reg_outputs, self.anchors
        else:
            predicts_list = list()
            for cls_out, reg_out, anchor in zip(cls_outputs, reg_outputs, self.anchors):
                anchor_wh = anchor[:, [2, 3]] - anchor[:, [0, 1]]
                anchor_xy = anchor[:, [0, 1]] + 0.5 * anchor_wh
                scale_reg = reg_out * self.std
                scale_reg[..., :2] = anchor_xy + scale_reg[..., :2] * anchor_wh
                scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchor_wh
                scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
                scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]

                predicts_out = torch.cat([scale_reg, cls_out], dim=-1)
                predicts_list.append(predicts_out)
            return predicts_list


class RetinaNet(nn.Module):
    def __init__(self,
                 anchor_sizes=None,
                 anchor_scales=None,
                 anchor_ratios=None,
                 strides=None,
                 num_cls=80,
                 backbone='resnet50'
                 ):
        super(RetinaNet, self).__init__()
        self.backbones = switch_backbones(backbone)
        c3, c4, c5 = self.backbones.inner_channels
        self.neck = FPN(c3, c4, c5, 256)
        self.head = RetinaHead(256,
                               256,
                               num_cls,
                               4,
                               5,
                               anchor_sizes,
                               anchor_scales,
                               anchor_ratios,
                               strides)

    def load_backbone_weighs(self, weights):
        miss_state_dict = self.backbones.load_state_dict(weights, strict=False)
        print(miss_state_dict)

    def forward(self, x):
        c3, c4, c5 = self.backbones(x)
        p3, p4, p5, p6, p7 = self.neck([c3, c4, c5])
        out = self.head([p3, p4, p5, p6, p7])
        return out


#
if __name__ == '__main__':
    input_tensor = torch.rand(size=(4, 3, 640, 640)).float()
    net = RetinaNet(backbone="resnet18")
    mcls_output, mreg_output, manchor = net(input_tensor)
    for cls_out, reg_out, anchor_out in zip(mcls_output, mreg_output, manchor):
        print(cls_out.shape, reg_out.shape, anchor_out.shape)
# out = net(input_tensor)
# for item in out:
#     print(item.shape)

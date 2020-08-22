import torch
import math
from torch import nn
from nets.common import CGR, FPN

INF = 1e8


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


class FCOSHead(nn.Module):
    def __init__(self, in_channel=256,
                 inner_channel=256,
                 num_convs=4,
                 num_cls=80,
                 strides=None,
                 layer_num=5,
                 centerness_on_reg=False):
        super(FCOSHead, self).__init__()
        self.num_cls = num_cls
        self.layer_num = layer_num
        self.centerness_on_reg = centerness_on_reg
        if strides is None:
            strides = [8, 16, 32, 64, 128]
        self.strides = strides
        self.scales = nn.ModuleList([Scale(init_val=1.0) for _ in range(self.layer_num)])
        self.grids = [torch.zeros(size=(0, 4))] * 5
        self.cls_bones = list()
        self.reg_bones = list()
        for i in range(num_convs):
            if i == 0:
                cls_conv_func = CGR(in_channel, inner_channel, 3, 1)
                reg_conv_func = CGR(in_channel, inner_channel, 3, 1)
            else:
                cls_conv_func = CGR(inner_channel, inner_channel, 3, 1)
                reg_conv_func = CGR(inner_channel, inner_channel, 3, 1)
            self.cls_bones.append(cls_conv_func)
            self.reg_bones.append(reg_conv_func)
        self.cls_bones = nn.Sequential(*self.cls_bones)
        self.reg_bones = nn.Sequential(*self.reg_bones)

        self.cls_head = nn.Conv2d(inner_channel, num_cls, 3, 1, 1)
        self.reg_head = nn.Conv2d(inner_channel, 4, 3, 1, 1)
        self.centerness = nn.Conv2d(inner_channel, 1, 3, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_head.bias, -math.log((1 - 0.01) / 0.01))

    def build_grids(self, feature_maps):
        assert len(self.strides) == len(feature_maps)
        assert self.layer_num == len(feature_maps)

        grids = list()
        for i in range(self.layer_num):
            feature_map = feature_maps[i]
            stride = self.strides[i]
            _, _, ny, nx = feature_map.shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
            grid = torch.stack([xv,
                                yv], dim=2)
            grid = (grid + 0.5) * stride
            grids.append(grid)
        return grids

    def forward(self, xs):
        cls_outputs = list()
        reg_outputs = list()
        center_outputs = list()
        for i, x in enumerate(xs):
            cls_tower = self.cls_bones(x)
            reg_tower = self.reg_bones(x)
            cls_outputs.append(self.cls_head(cls_tower))
            reg_outputs.append((self.scales[i](self.reg_head(reg_tower))).exp())
            if self.centerness_on_reg:
                center_outputs.append(self.centerness(reg_tower))
            else:
                center_outputs.append(self.centerness(cls_tower))
        if self.grids[0] is None or self.grids[0].shape[0] != cls_outputs[0].shape[2]:
            with torch.no_grad():
                grids = self.build_grids(xs)
                assert len(grids) == len(self.grids)
                for i, grid in enumerate(grids):
                    self.grids[i] = grid.to(xs[0].device)
        if self.training:
            return cls_outputs, reg_outputs, center_outputs, self.grids
        else:
            bs = xs[0].shape[0]
            output = list()
            for cls_predict, reg_predict, center_predict, grid in zip(cls_outputs, reg_outputs, center_outputs,
                                                                      self.grids):
                cls_output = cls_predict.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_cls)
                reg_output = reg_predict.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)
                center_output = center_predict.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
                grid_output = grid.view(-1, 2)[None]
                reg_output[..., :2] = grid_output - reg_output[..., :2]
                reg_output[..., 2:] = grid_output + reg_output[..., 2:]
                cat_output = torch.cat([reg_output, center_output, cls_output], dim=-1)
                output.append(cat_output)
            return output


class FCOS(nn.Module):
    def __init__(self,
                 strides=None,
                 num_cls=80,
                 backbone='resnet50'
                 ):
        super(FCOS, self).__init__()
        self.backbones = switch_backbones(backbone)
        c3, c4, c5 = self.backbones.inner_channels
        self.neck = FPN(c3, c4, c5, 256)
        self.head = FCOSHead(
            256,
            256,
            4,
            num_cls,
            strides
        )

    def load_backbone_weighs(self, weights):
        miss_state_dict = self.backbones.load_state_dict(weights, strict=False)
        print(miss_state_dict)

    def forward(self, x):
        c3, c4, c5 = self.backbones(x)
        p3, p4, p5, p6, p7 = self.neck([c3, c4, c5])
        out = self.head([p3, p4, p5, p6, p7])
        return out


# if __name__ == '__main__':
#     input_tensor = torch.rand(size=(4, 3, 640, 640)).float()
#     net = FCOS(backbone="resnet18").eval()
#     out = net(input_tensor)
#     for grid in out:
#         print(grid.shape)
#     net(input_tensor)
#     cls_outputs, reg_outputs, center_outputs, grids = net(input_tensor)
#     for grid in reg_outputs:
#         print(grid.shape)

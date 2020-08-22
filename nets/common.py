from torch import nn


class CR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CBR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class CGR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=None, bias=True):
        super(CGR, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=bias)
        self.gn = nn.GroupNorm(32, out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class FPN(nn.Module):
    def __init__(self, c3, c4, c5, out_channel=256):
        super(FPN, self).__init__()

        self.bones = nn.Sequential(
            nn.Conv2d(c5, out_channel, 1, 1, 0),  # 0
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 1

            nn.Conv2d(c4, out_channel, 1, 1, 0),  # 2
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 3

            nn.Conv2d(c3, out_channel, 1, 1, 0),  # 4
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),  # 5
            nn.Conv2d(c5, out_channel, 3, 2, 1),  # 6
            nn.ReLU(),  # 7
            nn.Conv2d(out_channel, out_channel, 3, 2, 1),  # 8
        )

    def forward(self, x):
        c3, c4, c5 = x
        f5 = self.bones[0](c5)
        p5 = self.bones[1](f5)

        f4 = self.bones[2](c4) + nn.UpsamplingNearest2d(size=(c4.shape[2:]))(f5)
        p4 = self.bones[3](f4)

        f3 = self.bones[4](c3) + nn.UpsamplingNearest2d(size=(c3.shape[2:]))(f4)
        p3 = self.bones[5](f3)

        p6 = self.bones[6](c5)

        p7 = self.bones[8](self.bones[7](p6))

        return [p3, p4, p5, p6, p7]

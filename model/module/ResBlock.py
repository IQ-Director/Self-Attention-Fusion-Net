# -*- coding: utf-8 -*-
# @File    : ISAB.py
# @annotation    : 残差模块
import torch.nn as nn

class ResLayer(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(ResLayer, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        mid_channel = width // self.expansion
        assert width == mid_channel * self.expansion, f"out_channel should be divisible by {self.expansion}"

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=mid_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=mid_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, layer_nums, stride=1, groups=1, width_per_group=64):
        super(ResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer_nums = layer_nums
        self.stride = stride
        self.groups = groups
        self.width_per_group = width_per_group
        self.block = self.make_block(ResLayer)

    def make_block(self, block):
        downsample = None
        if self.stride != 1 or self.in_channel != self.out_channel:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(self.out_channel))

        layers = [block(self.in_channel,
                        self.out_channel,
                        downsample=downsample,
                        stride=self.stride,
                        groups=self.groups,
                        width_per_group=self.width_per_group)]

        for _ in range(1, self.layer_nums):
            layers.append(block(self.out_channel,
                                self.out_channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

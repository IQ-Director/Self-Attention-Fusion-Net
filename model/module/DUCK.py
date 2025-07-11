# -*- coding: utf-8 -*-
# @File    : DUCK.py
# @annotation    : DUCK模块

import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_initializer = nn.init.kaiming_uniform_


class SeparatedConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super(SeparatedConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, size), padding=(0, size // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(size, 1), padding=(size // 2, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class MidscopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(MidscopeConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', dilation=dilation_rate,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=2 * dilation_rate,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class WidescopeConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(WidescopeConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', dilation=dilation_rate,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=2 * dilation_rate,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=3 * dilation_rate,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class ResNetConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(ResNetConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', dilation=dilation_rate,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=dilation_rate,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.bn1(self.conv1(x))
        out = self.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))
        out = out + identity
        out = self.relu(out)
        return out


class DoubleConvolutionWithBatchNormalization(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(DoubleConvolutionWithBatchNormalization, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', dilation=dilation_rate,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', dilation=dilation_rate,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class Duck(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        """
        Args:
            in_channels (): 输入通道数
            out_channels (): 输出通道数
            size (): 卷积核尺寸
        """
        super(Duck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = WidescopeConv2DBlock(in_channels, out_channels)
        self.conv2 = MidscopeConv2DBlock(in_channels, out_channels)
        self.conv3 = ResNetConv2DBlock(in_channels, out_channels)
        self.conv4 = ResNetConv2DBlock(in_channels, out_channels, dilation_rate=2)
        self.conv5 = ResNetConv2DBlock(in_channels, out_channels, dilation_rate=3)
        self.conv6 = SeparatedConv2DBlock(in_channels, out_channels, size=size)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(x))
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = self.conv6(x)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = F.relu(self.bn2(x))
        return x


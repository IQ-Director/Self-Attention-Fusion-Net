# -*- coding: utf-8 -*-
# @File    : SAFusionNet.py
# @annotation    : 模型主要结构，包含特征提取框架和分类器，特征提取框架输出(batch,num_classes,embedding_dim)的标签向量,分类器输出(batch,num_classes)的最终概率(已使用sigmoid)

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from .module import *
from .Classifier import *


class SelfAttentionEncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_nums, edge_length, stride=1, dilation=1, groups=1, bias=False):
        """
        Notes:
            论文中的SAEB，接受两个输入，并输出两个特征图
        Args:
            in_channels (): 输入通道数
            out_channels (): 输出通道数
            layer_nums (): 残差模块层数
            edge_length (): 输入特征图的宽高
        """
        super(SelfAttentionEncodingBlock, self).__init__()
        self.l_res = ResBlock(in_channels, out_channels, layer_nums=layer_nums, stride=stride)
        self.r_res = ResBlock(in_channels, out_channels, layer_nums=layer_nums, stride=stride)
        self.ll_isab = ISAB(out_channels, edge_length // 2)
        self.lr_isab = ISAB(out_channels, edge_length // 2)
        self.rl_isab = ISAB(out_channels, edge_length // 2)
        self.rr_isab = ISAB(out_channels, edge_length // 2)
        self.lr_para = nn.Parameter(torch.tensor(1.0))
        self.rl_para = nn.Parameter(torch.tensor(1.0))

    def forward(self, l, r):
        l = self.l_res(l)
        r = self.r_res(r)
        ll_attn = self.ll_isab(l, l)
        rr_attn = self.rr_isab(r, r)
        lr_attn = self.lr_isab(l, r)
        rl_attn = self.rl_isab(r, l)
        l_attn = ll_attn + self.lr_para * lr_attn
        r_attn = rr_attn + self.rl_para * rl_attn
        l = l + l * l_attn
        r = r + r * r_attn
        return l, r

class SAFEFramework(nn.Module):
    def __init__(self, in_channels, num_classes, edge_length=224, group_channels=64, kernel_size=3, stride=1, padding=0,
                 dilation=1,
                 bias=False):
        """
        Notes:
            特征提取框架
        Args:
            in_channels (): 输入通道数
            num_classes (): 分类的类别数
            edge_length (): 输入特征图的宽高
            group_channels (): 每一组通道的数量
        """
        super(SAFEFramework, self).__init__()
        # 左右眼图像下采样至1/2
        self.l_pre = nn.Sequential(
            nn.Conv2d(in_channels, group_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(group_channels),
            nn.ReLU(inplace=True)
        )
        self.r_pre = nn.Sequential(
            nn.Conv2d(in_channels, group_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(group_channels),
            nn.ReLU(inplace=True)
        )

        self.v_enc = nn.Sequential(ResBlock(group_channels * 2, group_channels * 4, layer_nums=3, stride=2),
                                   ResBlock(group_channels * 4, group_channels * 8, layer_nums=6, stride=2))
        self.saeb1 = SelfAttentionEncodingBlock(group_channels, group_channels * 4, edge_length=edge_length // 2,
                                                layer_nums=3, stride=2)
        self.saeb2 = SelfAttentionEncodingBlock(group_channels * 4, group_channels * 8, edge_length=edge_length // 4,
                                                layer_nums=6, stride=2)

        self.samb = SAMB(group_channels * 8, edge_length=edge_length // 8)

        self.res1 = ResBlock(group_channels * 8, group_channels * 16, layer_nums=2, stride=2)
        self.res2 = ResBlock(group_channels * 16, group_channels * 32, layer_nums=2, stride=2)
        self.duck = Duck(group_channels * 32, group_channels * 32)

        self.label_emb = LabelLevelEmbeddingClf(num_labels=num_classes, embed_dim=group_channels * 32,
                                                num_heads=16)  # [batch, num_labels, mid_channels]

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        right = F.hflip(right)
        l = self.l_pre(left)
        r = self.r_pre(right)
        m = torch.cat([l, r], dim=1)

        l, r = self.saeb1(l, r)
        l, r = self.saeb2(l, r)
        v = self.v_enc(m)

        samb_out = self.samb(l, r, v)
        res1_out = self.res1(samb_out)

        res2_out = self.res2(res1_out)

        duck_out = self.duck(res2_out)

        label_emb = self.label_emb(duck_out)

        return label_emb

class SAFusionNet(nn.Module):
    def __init__(self,num_classes, in_channels=3, edge_length=224, group_channels=64, kernel_size=3, stride=1, padding=0,
                 dilation=1,
                 bias=False):
        """
        Notes:
            模型主结构，包括特征提取与分类器
        Args:
            in_channels (): 输入通道数
            num_classes (): 分类的类别数
            edge_length (): 输入特征图的宽高
            group_channels (): 每一组通道的数量
        Returns:
            (label_emb, drbm_out) 用于计算对比损失的标签水平向量，以及最终各类别的概率
        """
        super(SAFusionNet, self).__init__()
        self.main = SAFEFramework(in_channels, num_classes,edge_length=edge_length)
        self.drbm = HierarchicalDRBM(classes=num_classes, dim=group_channels * 32,
                                     hidden_per_class=group_channels * 32 // num_classes,
                                     top_hidden=group_channels * 32 // num_classes)

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        label_emb = self.main(left, right)

        drbm_out,_ = self.drbm(label_emb)

        return label_emb,drbm_out
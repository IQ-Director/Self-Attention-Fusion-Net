# -*- coding: utf-8 -*-
# @File    : LabelClf.py
# @annotation    : 论文中的LLEB

import torch
import torch.nn as nn
from torch.nn import init


class LabelLevelEmbeddingClf(nn.Module):
    def __init__(self, num_labels, embed_dim, num_heads):
        """
        Notes:
            输入为[batch, C, H, W]，输出为[batch, num_labels, C/embed_dim],输出可以直接用于MultiLabelContrastiveLoss计算
        Args:
            num_labels (): 类别数，也是输出时类别向量的行数
            embed_dim (): 特征图的通道数，也是输出中每个类别向量的维度
            num_heads (): 多头注意力的头数，需要被embed_dim整除
        """
        super(LabelLevelEmbeddingClf, self).__init__()

        # 标签嵌入 (每个标签一个嵌入向量)
        self.label_embedding = nn.Embedding(num_labels, embed_dim)
        init.uniform_(self.label_embedding.weight, a=-0.1, b=0.1)

        # 多头注意力机制 (Q=标签嵌入, K=V=图像特征)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, img_features):
        B, C, H, W = img_features.shape

        # 变换图像特征: [batch, C, H, W] -> [batch, H*W, C]
        img_features = img_features.flatten(2).permute(0, 2, 1)  # [batch, H*W, embed_dim]

        # 标签嵌入: [num_labels, embed_dim] -> [batch, num_labels, embed_dim]
        label_embeds = self.label_embedding.weight.unsqueeze(0).repeat(B, 1, 1)  # [batch, num_labels, embed_dim]

        # 多头注意力 (Q=标签嵌入, K=V=图像特征)
        attn_output, _ = self.multihead_attn(label_embeds, img_features, img_features)  # [batch, num_labels, embed_dim]

        return attn_output  # 返回每个标签的特征，进行MultiLabelContrastiveLoss计算

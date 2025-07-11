# -*- coding: utf-8 -*-
# @File    : ISAB.py
# @annotation    : 论文中的ISAB

import torch
import torch.nn as nn

class SpatialRelationAttention(nn.Module):
    def __init__(self, in_channels, edge_length):
        """
        Notes:
            ISAB模块的空间相关注意力（Spatial Relation Attention，SRA）
        """
        super(SpatialRelationAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1,)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1,)
        self.linear = nn.Linear(edge_length ** 2, 1)
        self.layernorm = nn.LayerNorm(normalized_shape=[edge_length,edge_length])

    def forward(self, q, k):
        assert q.shape == k.shape
        B, C, H, W = q.shape
        query = self.query_conv(q).view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        key = self.key_conv(k).view(B, C, -1)  # (B, C, HW)
        spatial_relation = torch.matmul(query, key)  # (B, HW, HW)
        spatial_value = self.linear(spatial_relation).squeeze(-1).view(B, H, W)  # (B, H, W)
        spatial_value = self.layernorm(spatial_value)
        return spatial_value.unsqueeze(1)  # (B, 1, H, W)


class ChannelRelationAttention(nn.Module):
    def __init__(self, in_channels, edge_length):
        """
        Notes:
            ISAB模块的通道相关注意力（Channel Relation Attention，CRA）
        """
        super(ChannelRelationAttention, self).__init__()
        self.linear_query = nn.Linear(edge_length ** 2, 1)
        self.linear_key = nn.Linear(edge_length ** 2, 1)
        self.linear_channel = nn.Linear(in_channels, out_features=1)
        self.layernorm = nn.LayerNorm(normalized_shape=[in_channels,1])

    def forward(self, q, k):
        assert q.shape == k.shape
        B, C, H, W = q.shape
        q_flatten = q.view(B, C, -1)  # (B, C, HW)
        k_flatten = k.view(B, C, -1)  # (B, C, HW)
        query = self.linear_query(q_flatten)  # (B, C, 1)
        key = self.linear_key(k_flatten).permute(0, 2, 1)  # (B, 1, C)
        channel_relation = torch.matmul(query, key)  # (B, C, C)
        channel_value = self.linear_channel(channel_relation)  # (B, C, 1)
        channel_value = self.layernorm(channel_value)
        return channel_value.unsqueeze(-1)  # (B, C, 1, 1)


class ISAB(nn.Module):
    def __init__(self, in_channels, edge_length):
        """
        Notes:
            眼间自注意力模块（Interocular Self-Attention Block，ISAB），接受两个输入，返回计算得到的权重矩阵
        Args:
            in_channels (): 输入特征图通道数
            edge_length (): 输入特征图的宽高
        """
        super(ISAB, self).__init__()
        self.spatial_attention = SpatialRelationAttention(in_channels, edge_length)
        self.channel_attention = ChannelRelationAttention(in_channels, edge_length)

    def forward(self, q, k):
        spatial_weight = self.spatial_attention(q, k)  # (B, 1, H, W)
        channel_weight = self.channel_attention(q, k)  # (B, C, 1, 1)
        attention = spatial_weight * channel_weight  # (B, C, H, W)
        return attention

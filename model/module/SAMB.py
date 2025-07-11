# -*- coding: utf-8 -*-
# @File    : SAMB.py
# @annotation    : 论文中的SAMB
import torch
import torch.nn as nn
from .DUCK import Duck
from .ISAB import ISAB
from .ASPP import ASPPBlock

class SAMB(nn.Module):
    def __init__(self, in_channels, edge_length):
        """
        Notes:
            自注意力融合模块（Self-Attention Merge Block，SAMB）\n
            接受三个输入，合并后输出一个特征图
        Args:
            in_channels (): 输入特征图通道数
            edge_length (): 输入特征图的宽高
        """
        super(SAMB, self).__init__()
        self.duck = Duck(in_channels*2,in_channels)
        self.isab_qk = ISAB(in_channels,edge_length)
        self.isab_v = ISAB(in_channels,edge_length)
        self.aspp = ASPPBlock(in_channels * 2, in_channels)

    def forward(self,q,k,v):
        concat_qk = torch.cat([q,k], dim=1)
        duck_out = self.duck(concat_qk)

        attention_qk = self.isab_qk(duck_out,v)   # (B, 1, H, W)
        attention_v = self.isab_v(v, duck_out)
        duck_out = duck_out + duck_out * attention_qk
        v = v + v * attention_v

        concat = torch.cat([duck_out, v], dim=1)
        output = self.aspp(concat)

        return output

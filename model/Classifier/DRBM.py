# -*- coding: utf-8 -*-
# @File    : DRBM.py
# @annotation    : 论文中的DLDRBM

import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.2):
        """
        Notes:
            多层感知机
        Args:
            in_features (): 输入节点数
            hidden_features (): 隐藏层节点数
            out_features (): 输出节点数
            drop (): 失活率
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or (out_features + in_features) // 2

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class STEBernoulli(torch.autograd.Function):
    """直通估计器（Straight-Through Estimator）"""

    @staticmethod
    def forward(ctx, probs):
        # 前向传播执行伯努利采样
        return torch.bernoulli(probs)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播直通梯度（忽略采样操作的不可导性）
        return grad_output


class DRBMLayer(nn.Module):
    """单层DRBM基础模块"""

    def __init__(self, n_visible, n_hidden, n_labels=0, use_residual=True):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_labels = n_labels
        self.use_residual = use_residual

        # 可见层 = 输入特征 + 标签（可选）
        self.W = nn.Parameter(torch.randn(n_visible + n_labels, n_hidden) * 0.02)
        self.v_bias = nn.Parameter(torch.zeros(n_visible + n_labels))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

        # 残差连接
        if use_residual:
            self.res_weight = nn.Parameter(torch.zeros(n_visible + n_labels, n_hidden))
            nn.init.orthogonal_(self.res_weight)

    def sample_h(self, v):
        """隐藏层采样（含残差连接）"""
        linear_out = F.linear(v, self.W.t(), self.h_bias)

        if self.use_residual:
            res_out = F.linear(v, self.res_weight.t())
            linear_out = linear_out + res_out

        return torch.sigmoid(linear_out)

    def sample_v(self, h):
        """可见层采样（含残差连接）"""
        linear_out = F.linear(h, self.W, self.v_bias)

        if self.use_residual:
            res_out = F.linear(h, self.res_weight)  # 与隐藏层相似的残差计算
            linear_out = linear_out + res_out

        return torch.sigmoid(linear_out)

    def forward(self, v):
        h_prob = self.sample_h(v)
        v_recon = self.sample_v(h_prob)
        return v_recon

    def contrastive_divergence_loss(self, v_pos, k=1):
        """对比散度损失，支持 STE 采样"""
        v_neg = v_pos.detach()

        for _ in range(k):
            h_prob = self.sample_h(v_neg)
            h_sample = STEBernoulli.apply(h_prob)

            v_prob = self.sample_v(h_sample)
            v_neg = STEBernoulli.apply(v_prob)

        # 正能量项
        pos_hidden_act = self.sample_h(v_pos)
        pos_energy = -F.linear(v_pos, self.W.t(), self.h_bias) * pos_hidden_act
        pos_energy = pos_energy.sum(dim=1).mean()

        # 负能量项
        neg_hidden_act = self.sample_h(v_neg)
        neg_energy = -F.linear(v_neg, self.W.t(), self.h_bias) * neg_hidden_act
        neg_energy = neg_energy.sum(dim=1).mean()

        # 返回对比散度
        return pos_energy - neg_energy, h_sample


class HierarchicalDRBM(nn.Module):
    """分层DRBM模型"""

    def __init__(self, classes, dim, hidden_per_class=128,
                 top_hidden=128, normal_class_idx=0):
        """
        Notes:
            DLDRBM分类器
        Args:
            classes (): 类别数
            dim (): 每个类别向量的维度数
            hidden_per_class (): 每个类别DRBM的节点数
            top_hidden (): 所有类别合并后分类时隐藏层的节点数
            normal_class_idx (): 正常类别的向量所在的索引
        """
        super().__init__()
        self.classes = classes
        self.dim = dim
        self.normal_class_idx = normal_class_idx
        self.hidden_per_class = hidden_per_class

        # 每个类别独立的DRBM
        self.class_drbms = nn.ModuleList([
            DRBMLayer(n_visible=dim, n_hidden=hidden_per_class)
            for _ in range(classes)
        ])

        # 不正常部分的DRBM
        self.abnormal_drbm1 = DRBMLayer(n_visible=(classes - 1) * hidden_per_class,
                                        n_hidden=(classes - 1) // 2 * hidden_per_class)
        self.abnormal_drbm2 = DRBMLayer(n_visible=(classes - 1) // 2 * hidden_per_class, n_hidden=hidden_per_class)

        # 不正常部分的全连接层
        self.mlp_abnormal = Mlp((classes - 1) * hidden_per_class, (classes - 1) // 2 * hidden_per_class,
                                hidden_per_class, drop=0.1)

        self.binary_mlp = Mlp(2 * hidden_per_class, top_hidden // (classes // 2), 2)
        self.all_mlp = Mlp(classes * hidden_per_class, top_hidden, classes)

    def encode_classes(self, x):
        """逐类编码：x.shape = [batch, classes, dim]"""
        hidden_states = []

        # 对每个类别并行处理
        for i in range(self.classes):
            class_input = x[:, i, :]  # [batch, dim]
            drbm = self.class_drbms[i]

            # 获取隐藏层表示
            h_prob = drbm.sample_h(class_input)
            hidden_states.append(h_prob)

        # 拼接所有类别的隐藏状态
        return torch.cat(hidden_states, dim=1)  # [batch, classes*hidden_per_class]

    def binary_clf(self, combined_h):
        """二分类预测"""
        # 分割为正常与不正常部分
        normal_part = combined_h[:, :self.hidden_per_class]  # 正常部分
        abnormal_part = combined_h[:, self.hidden_per_class:]  # 不正常部分

        abnormal_part = self.mlp_abnormal(abnormal_part)  # [batch, hidden_per_class]
        # abnormal_part = self.abnormal_drbm1.sample_h(abnormal_part)
        # abnormal_part = self.abnormal_drbm2.sample_h(abnormal_part)

        # 将正常与不正常部分重新拼接
        combined_h = torch.cat([normal_part, abnormal_part], dim=1)
        binary_probs = self.binary_mlp(combined_h)

        return torch.softmax(binary_probs, dim=1)

    def forward(self, x):
        """多标签全分类预测"""
        combined_h = self.encode_classes(x)

        # 二分类预测
        binary_probs = self.binary_clf(combined_h)  # [batch, 2]

        normal_gate = binary_probs[:, 0].unsqueeze(1)
        abnormal_gate = binary_probs[:, 1].unsqueeze(1)

        # Gating 不正常部分（乘以异常概率）
        abnormal_part = combined_h[:, self.hidden_per_class:]
        gated_abnormal = abnormal_part * abnormal_gate

        # Gating 正常部分（乘以正常概率）
        normal_part = combined_h[:, :self.hidden_per_class]
        gated_normal = normal_part * normal_gate

        # 拼接正常与 gated 的不正常部分
        combined_h = torch.cat([
            gated_normal,
            gated_abnormal
        ], dim=1)

        probs = self.all_mlp(combined_h)

        return torch.sigmoid(probs), binary_probs


def train_class_drbms(model, optimizer, label_emb):
    """
    DRBM训练函数，使用该函数基于对比散度对DRBM部分进行独立训练
    Args:
        model (): 模型包含DRBM的部分
        optimizer (): DRBM部分的优化器
        label_emb (): 特征提取部分输出的标签向量
    """
    label_emb = label_emb.detach()

    # 类别 DRBM 联合训练
    optimizer.zero_grad()
    total_loss = 0
    output = []
    for class_idx, drbm in enumerate(model.class_drbms):
        class_x = label_emb[:, class_idx, :]
        loss, out = drbm.contrastive_divergence_loss(class_x)
        total_loss += loss
        output.append(out)

    total_loss.backward(retain_graph=True)
    optimizer.step()

    # abnormal DRBM 训练
    abnormal_part = torch.cat(output[1:], dim=1)
    loss1, drbm1_out = model.abnormal_drbm1.contrastive_divergence_loss(abnormal_part)
    loss2, _ = model.abnormal_drbm2.contrastive_divergence_loss(drbm1_out)

    total_loss = loss1 + loss2  # 合并两个损失
    optimizer.zero_grad()
    total_loss.backward()  # 一次反向传播
    optimizer.step()

# -*- coding: utf-8 -*-
# @annotation    : 损失函数实现版本1（MultiLabelContrastiveLoss实现略有不同，FocalLoss完全相同）

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        接受(batch,num_classes)的张量，该损失类不含sigmoid，需要在该类之前计算sigmoid。
        Args:
            alpha ():
            gamma ():
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, probs, targets):
        bce_loss = self.bce(probs, targets)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return loss.mean()

class MultiLabelContrastiveLoss(nn.Module):
    def __init__(self, temperature=2):
        '''
        有监督多标签对比学习损失函数，接受一个(batch,num_classes,embedding_dim)的向量
        Args:
            temperature (): 温度参数
        '''
        super().__init__()
        self.temperature = temperature

    def forward(self, label_embeddings, labels):
        '''
        Args:
            label_embeddings (): 标签嵌入向量[batch_size, num_classes, embed_dim]
            labels (): 标签[batch_size, num_classes]
        '''
        batch_size, num_classes, embed_dim = label_embeddings.shape
        device = label_embeddings.device

        # 获取活跃标签的嵌入（正样本锚点）
        active_mask = (labels == 1)  # [batch_size, num_classes]
        active_embeddings = label_embeddings[active_mask]  # [num_active, embed_dim]
        active_labels = torch.where(active_mask)  # (batch_idx, class_idx)

        # 如果没有活跃标签，仅返回 0 损失
        if active_embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算相似度矩阵
        similarity = torch.matmul(active_embeddings, active_embeddings.T) / self.temperature
        similarity = similarity - similarity.max().detach()  # 防止溢出

        # 构建正样本掩码：相同类别但不同样本
        pos_mask = torch.zeros_like(similarity, dtype=torch.bool)
        for i, (b_i, c_i) in enumerate(zip(*active_labels)):
            same_class = (active_labels[1] == c_i) & (active_labels[0] != b_i)
            pos_mask[i, same_class] = True

        # 检查是否存在正样本
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算对比损失分子（正样本相似度）
        pos_pairs = similarity[pos_mask]  # [total_pos_pairs]
        num_pos_per_anchor = pos_mask.sum(dim=1)  # [num_active]

        # 过滤掉没有正样本的锚点
        valid_anchors = num_pos_per_anchor > 0  # [num_active]
        num_valid_anchors = valid_anchors.sum()

        if num_valid_anchors == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 计算对比损失分母（所有样本相似度）
        exp_sim = torch.exp(similarity).clamp(max=1e8)  # 避免过大
        denominator = exp_sim.sum(dim=1, keepdim=True) + 1e-8  # 避免 log(0)

        # 计算 log_prob
        log_prob = pos_pairs - torch.log(denominator[valid_anchors])

        # 计算最终损失
        contrast_loss = -log_prob.mean()

        return contrast_loss

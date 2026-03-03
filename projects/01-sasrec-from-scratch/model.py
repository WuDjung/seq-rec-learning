"""
model.py - SASRec 模型实现
参考: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018
"""

import torch
import torch.nn as nn
import math


class SASRec(nn.Module):
    """
    SASRec: 基于单向自注意力的序列推荐模型。

    参数:
        num_items: int, 物品总数（不含 padding 的 0）
        max_len: int, 最大序列长度
        embed_dim: int, 嵌入维度（同时也是 Transformer 的 d_model）
        num_heads: int, 多头注意力的头数
        num_layers: int, Transformer Encoder 的层数
        dropout: float, Dropout 比率
    """

    def __init__(self, num_items, max_len, embed_dim, num_heads, num_layers, dropout):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.embed_dim = embed_dim

        # ---- Embedding 层 ----
        # padding_idx=0: ID 为 0 的位置（填充）嵌入向量恒为零向量，不参与梯度更新
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        # Embedding 之后的 LayerNorm 和 Dropout
        # 这是 SASRec 原始实现中的做法，有助于稳定训练
        self.emb_layernorm = nn.LayerNorm(embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # ---- Transformer Encoder 层 ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,  # 前馈网络的隐藏层维度，通常是 4 倍 embed_dim
            dropout=dropout,
            activation="gelu",             # SASRec 原文用 ReLU，GELU 是更现代的选择
            batch_first=True,              # 输入形状: (batch, seq_len, embed_dim)
            norm_first=True,               # Pre-Norm: 先 LayerNorm 再注意力（训练更稳定）
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ---- 输出层 ----
        self.output_layernorm = nn.LayerNorm(embed_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """使用截断正态分布初始化嵌入层权重"""
        nn.init.trunc_normal_(self.item_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)
        # padding 位置的嵌入必须保持为零
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0)

    def forward(self, input_seq):
        """
        前向传播。

        参数:
            input_seq: (batch_size, max_len), 物品 ID 序列，0 表示 padding

        返回:
            (batch_size, max_len, embed_dim), 每个位置的隐藏表示
        """
        batch_size, seq_len = input_seq.shape

        # 1. Embedding: 物品嵌入 + 位置嵌入
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        x = self.item_embedding(input_seq) + self.position_embedding(positions)
        x = self.emb_layernorm(x)
        x = self.emb_dropout(x)

        # 2. 构造因果 mask
        # causal_mask[i][j] = True 表示位置 i 不能看到位置 j
        # 上三角矩阵: 对角线以上为 True，确保每个位置只能看到自己和之前的位置
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_seq.device),
            diagonal=1
        ).bool()

        # 3. Padding mask: 物品 ID 为 0 的位置不应被关注
        padding_mask = (input_seq == 0)

        # 4. Transformer 编码
        x = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        x = self.output_layernorm(x)
        return x

    def predict(self, input_seq, candidate_items=None):
        """
        预测: 给定输入序列，计算候选物品的分数。

        参数:
            input_seq: (batch_size, max_len), 输入序列
            candidate_items: (batch_size, num_candidates), 候选物品 ID
                             如果为 None, 则对所有物品打分

        返回:
            如果 candidate_items 不为 None: (batch_size, num_candidates)
            如果 candidate_items 为 None: (batch_size, num_items)
        """
        hidden = self.forward(input_seq)

        # 取最后一个非 padding 位置的表示作为用户兴趣向量
        # 由于我们使用左侧填充，最后一个位置一定是有效的
        user_repr = hidden[:, -1, :]  # (batch_size, embed_dim)

        if candidate_items is not None:
            # 只对候选物品打分
            candidate_emb = self.item_embedding(candidate_items)  # (batch, num_candidates, embed_dim)
            scores = torch.bmm(candidate_emb, user_repr.unsqueeze(-1)).squeeze(-1)
        else:
            # 对所有物品打分（去掉 padding 的第 0 个）
            all_item_emb = self.item_embedding.weight[1:]  # (num_items, embed_dim)
            scores = torch.matmul(user_repr, all_item_emb.T)  # (batch, num_items)

        return scores

    def compute_loss(self, input_seq, target_seq):
        """
        计算训练损失（Binary Cross-Entropy）。

        SASRec 原文使用 BCE loss + 负采样:
        - 正样本: 目标序列中的真实物品
        - 负样本: 随机采样的物品

        但这里我们使用更简单且通常效果更好的 Cross-Entropy loss:
        在每个位置上，把预测下一个物品看作一个多分类问题。

        参数:
            input_seq: (batch_size, max_len), 输入序列
            target_seq: (batch_size, max_len), 目标序列

        返回:
            loss: 标量
        """
        hidden = self.forward(input_seq)  # (batch, max_len, embed_dim)

        # 与所有物品嵌入计算分数
        all_item_emb = self.item_embedding.weight[1:]  # (num_items, embed_dim)
        logits = torch.matmul(hidden, all_item_emb.T)  # (batch, max_len, num_items)

        # 将 logits 和 target 展平，方便计算 Cross-Entropy
        # logits: (batch * max_len, num_items)
        # target: (batch * max_len,)
        logits = logits.view(-1, self.num_items)
        target = target_seq.view(-1)

        # target 中为 0 的位置是 padding，不参与 loss 计算
        # 由于物品 ID 从 1 开始，而 CrossEntropyLoss 期望类别从 0 开始
        # 所以 target 需要减 1，padding 位置用 ignore_index 忽略
        target = target - 1  # padding 的 0 变为 -1

        loss = nn.functional.cross_entropy(
            logits,
            target,
            ignore_index=-1,  # 忽略 padding 位置
        )

        return loss

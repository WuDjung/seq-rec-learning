"""
model.py - BERT4Rec 模型实现
参考: Sun et al., "BERT4Rec: Sequential Recommendation with
      Bidirectional Encoder Representations from Transformer", CIKM 2019
"""

import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    """
    BERT4Rec: 基于双向自注意力 + Masked Item Prediction 的序列推荐模型。

    与 SASRec 的核心区别:
    1. 无因果 mask（双向注意力，每个位置可以看到所有其他位置）
    2. 嵌入表包含 mask token（ID = num_items + 1）
    3. 训练时只在被 mask 的位置计算损失

    参数:
        num_items: int, 物品总数（不含 padding 和 mask token）
        max_len: int, 最大序列长度
        embed_dim: int, 嵌入维度
        num_heads: int, 多头注意力头数
        num_layers: int, Transformer 层数
        dropout: float, Dropout 比率
    """

    def __init__(self, num_items, max_len, embed_dim, num_heads, num_layers, dropout):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.mask_token = num_items + 1

        # ---- Embedding 层 ----
        # 词表大小: 0(padding) + num_items(物品) + 1(mask token) = num_items + 2
        self.item_embedding = nn.Embedding(num_items + 2, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        self.emb_layernorm = nn.LayerNorm(embed_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # ---- Transformer Encoder 层 ----
        # 注意: BERT4Rec 不使用因果 mask，所以这里和 SASRec 的唯一区别
        # 是在 forward 中不传入因果 mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",      # BERT4Rec 原文使用 GELU
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ---- 输出层 ----
        self.output_layernorm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.item_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0)

    def forward(self, input_seq):
        """
        前向传播（双向，不使用因果 mask）。

        参数:
            input_seq: (batch_size, max_len), 包含 mask token 的序列

        返回:
            (batch_size, max_len, embed_dim), 每个位置的隐藏表示
        """
        batch_size, seq_len = input_seq.shape

        # 1. Embedding
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        x = self.item_embedding(input_seq) + self.position_embedding(positions)
        x = self.emb_layernorm(x)
        x = self.emb_dropout(x)

        # 2. Padding mask（padding 位置不应被关注）
        # 注意: 这里不使用因果 mask，这是和 SASRec 的关键区别
        padding_mask = (input_seq == 0)

        # 3. Transformer 编码（双向）
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.output_layernorm(x)

        return x

    def predict(self, input_seq):
        """
        预测: 对末尾 [mask] 位置的物品打分。

        推理时，输入序列的最后一个 token 是 [mask]，
        我们取该位置的隐藏表示与所有物品嵌入做内积。
        """
        hidden = self.forward(input_seq)  # (batch, max_len, embed_dim)

        # 取最后一个位置（[mask] 位置）
        mask_repr = hidden[:, -1, :]  # (batch, embed_dim)

        # 与所有物品嵌入做内积（不包括 padding 和 mask token）
        all_item_emb = self.item_embedding.weight[1:self.num_items + 1]  # (num_items, embed_dim)
        scores = torch.matmul(mask_repr, all_item_emb.T)  # (batch, num_items)

        return scores

    def compute_loss(self, masked_seq, labels):
        """
        计算训练损失: 只在被 mask 的位置计算 Cross-Entropy。

        参数:
            masked_seq: (batch_size, max_len), 包含 mask token 的输入序列
            labels: (batch_size, max_len), 被 mask 位置的真实物品 ID，
                    非 mask 位置为 0
        """
        hidden = self.forward(masked_seq)  # (batch, max_len, embed_dim)

        # 与所有物品嵌入做内积
        all_item_emb = self.item_embedding.weight[1:self.num_items + 1]  # (num_items, embed_dim)
        logits = torch.matmul(hidden, all_item_emb.T)  # (batch, max_len, num_items)

        # 展平
        logits = logits.view(-1, self.num_items)  # (batch * max_len, num_items)
        labels = labels.view(-1)                   # (batch * max_len,)

        # labels 中非 0 的位置才是需要预测的（被 mask 的位置）
        # 物品 ID 从 1 开始，CrossEntropyLoss 期望类别从 0 开始，所以减 1
        labels = labels - 1  # 非 mask 位置从 0 变为 -1

        loss = nn.functional.cross_entropy(
            logits,
            labels,
            ignore_index=-1,
        )

        return loss
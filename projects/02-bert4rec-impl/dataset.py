"""
dataset.py - BERT4Rec 训练用的 PyTorch Dataset
核心区别: 训练时随机 mask 物品，让模型根据上下文预测被 mask 的物品
"""

import copy
import random
import torch
from torch.utils.data import Dataset


# 特殊 token ID
# 0 = padding, 物品 ID 从 1 开始, mask token 的 ID 放在物品之后
# 具体的 mask_token_id 在初始化时根据 num_items 确定


class BERT4RecDataset(Dataset):
    """
    BERT4Rec 训练数据集。

    每次 __getitem__ 被调用时，随机 mask 输入序列中的部分物品。
    这意味着同一个用户在不同 epoch 中看到的训练样本是不同的，
    大幅增加了有效训练样本数量。
    """

    def __init__(self, train_seqs, max_len, mask_prob, num_items):
        """
        参数:
            train_seqs: dict, {user_id: [item1, item2, ...]}
            max_len: int, 最大序列长度
            mask_prob: float, 每个物品被 mask 的概率
            num_items: int, 物品总数
        """
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = num_items + 1  # mask token 的 ID，排在所有物品之后
        self.num_items = num_items

        # 将 dict 转为 list，方便按索引访问
        self.users = list(train_seqs.keys())
        self.seqs = [train_seqs[u] for u in self.users]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        seq = self.seqs[index]

        # 截断到最近的 max_len 个行为
        seq = seq[-self.max_len:]

        # 左侧填充
        pad_len = self.max_len - len(seq)
        padded_seq = [0] * pad_len + seq

        # 随机 mask: 对每个非 padding 位置，以 mask_prob 的概率替换为 mask token
        masked_seq = padded_seq.copy()
        labels = [0] * self.max_len  # 0 表示该位置不需要预测

        for i in range(pad_len, self.max_len):
            if random.random() < self.mask_prob:
                labels[i] = masked_seq[i]     # 记录真实物品 ID 作为预测目标
                masked_seq[i] = self.mask_token  # 替换为 mask token

        # 安全检查: 确保至少有一个位置被 mask
        # 如果一个都没 mask（序列很短且概率不幸），强制 mask 最后一个
        if sum(1 for l in labels if l != 0) == 0:
            labels[-1] = masked_seq[-1]
            masked_seq[-1] = self.mask_token

        return (
            torch.tensor(masked_seq, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class BERT4RecEvalDataset(Dataset):
    """
    BERT4Rec 评估数据集。

    推理时，在用户历史序列的末尾追加一个 [mask] token，
    让模型预测这个位置的物品（即下一个物品）。
    """

    def __init__(self, eval_data, max_len, mask_token):
        """
        参数:
            eval_data: dict, {user_id: (历史序列, 目标物品)}
            max_len: int, 最大序列长度
            mask_token: int, mask token 的 ID
        """
        self.max_len = max_len
        self.mask_token = mask_token
        self.users = []
        self.input_seqs = []
        self.targets = []

        for user, (seq, target) in eval_data.items():
            # 截断: 留一个位置给末尾的 [mask]
            seq = seq[-(max_len - 1):]

            # 在末尾追加 [mask] token
            seq = seq + [mask_token]

            # 左侧填充到 max_len
            pad_len = max_len - len(seq)
            seq = [0] * pad_len + seq

            self.users.append(user)
            self.input_seqs.append(seq)
            self.targets.append(target)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_seqs[index], dtype=torch.long),
            torch.tensor(self.targets[index], dtype=torch.long),
        )
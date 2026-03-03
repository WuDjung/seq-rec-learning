"""
dataset.py - SASRec 训练用的 PyTorch Dataset
功能: 将用户行为序列转化为模型可用的(输入序列, 目标序列)对
"""

import torch
from torch.utils.data import Dataset


class SASRecDataset(Dataset):
    """
    SASRec 训练数据集。

    对每个用户的训练序列:
    - 截断到 max_len + 1（需要多一个位置来构造目标）
    - 左侧填充 0 至等长
    - 输入 = 序列[:-1], 目标 = 序列[1:]
    """

    def __init__(self, train_seqs, max_len):
        """
        参数:
            train_seqs: dict, {user_id: [item1, item2, ...]}
            max_len: int, 模型能处理的最大序列长度
        """
        self.max_len = max_len
        self.users = []
        self.input_seqs = []
        self.target_seqs = []

        for user, seq in train_seqs.items():
            # 截断: 只保留最近的 max_len + 1 个行为
            # +1 是因为需要从中拆出输入和目标
            seq = seq[-(max_len + 1):]

            # 填充: 如果长度不足 max_len + 1, 在左侧补0
            pad_len = (max_len + 1) - len(seq)
            seq = [0] * pad_len + seq

            # 拆分输入和目标
            # 输入: seq[0] 到 seq[-2], 长度 = max_len
            # 目标: seq[1] 到 seq[-1], 长度 = max_len
            input_seq = seq[:-1]
            target_seq = seq[1:]

            self.users.append(user)
            self.input_seqs.append(input_seq)
            self.target_seqs.append(target_seq)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return (
            torch.tensor(self.input_seqs[index], dtype=torch.long),
            torch.tensor(self.target_seqs[index], dtype=torch.long),
        )


class EvalDataset(Dataset):
    """
    评估数据集（验证/测试通用）。

    对每个用户:
    - 输入序列: 该用户在当前阶段可见的历史行为（截断 + 左侧填充）
    - 目标物品: 需要预测的那一个物品（验证集是倒数第2个, 测试集是最后1个）
    """

    def __init__(self, eval_data, max_len):
        """
        参数:
            eval_data: dict, {user_id: (历史序列, 目标物品)}
            max_len: int, 模型能处理的最大序列长度
        """
        self.max_len = max_len
        self.users = []
        self.input_seqs = []
        self.targets = []

        for user, (seq, target) in eval_data.items():
            # 截断到最近 max_len 个行为
            seq = seq[-max_len:]

            # 左侧填充
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

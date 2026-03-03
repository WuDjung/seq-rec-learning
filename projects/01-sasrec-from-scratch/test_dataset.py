"""
test_dataset.py - 验证 Dataset 实现是否正确
运行: python test_dataset.py
"""

import os
import pickle
from torch.utils.data import DataLoader
from dataset import SASRecDataset, EvalDataset


def main():
    # 加载预处理数据
    data_path = os.path.join("data", "ml-1m", "processed.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    max_len = 200

    # 构建训练集
    train_dataset = SASRecDataset(data["train_seqs"], max_len)
    print(f"训练样本数: {len(train_dataset)}")

    # 取一个样本看看
    input_seq, target_seq = train_dataset[0]
    print(f"\n样例 - 训练集第1个用户:")
    print(f"  输入序列形状: {input_seq.shape}")
    print(f"  目标序列形状: {target_seq.shape}")
    print(f"  输入序列前10个: {input_seq[:10].tolist()}")
    print(f"  目标序列前10个: {target_seq[:10].tolist()}")

    # 验证: 输入的第2个元素 == 目标的第1个元素（错位一位）
    assert input_seq[1].item() == target_seq[0].item(), "输入和目标没有正确错位!"
    print("  ✓ 输入/目标错位关系正确")

    # 验证: 填充部分为0
    num_padding = (input_seq == 0).sum().item()
    print(f"  ✓ 填充数量: {num_padding}")

    # 构建验证集
    valid_dataset = EvalDataset(data["valid_data"], max_len)
    print(f"\n验证样本数: {len(valid_dataset)}")

    input_seq, target = valid_dataset[0]
    print(f"样例 - 验证集第1个用户:")
    print(f"  输入序列形状: {input_seq.shape}")
    print(f"  目标物品: {target.item()}")

    # 测试 DataLoader 能否正常打包 batch
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    batch_input, batch_target = next(iter(train_loader))
    print(f"\nDataLoader 测试:")
    print(f"  batch 输入形状: {batch_input.shape}")
    print(f"  batch 目标形状: {batch_target.shape}")
    print(f"  ✓ DataLoader 工作正常")


if __name__ == "__main__":
    main()

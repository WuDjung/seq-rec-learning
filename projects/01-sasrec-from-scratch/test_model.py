"""
test_model.py - 验证 SASRec 模型是否能正常前向传播和计算损失
运行: python test_model.py
"""

import os
import pickle
import torch
from dataset import SASRecDataset
from torch.utils.data import DataLoader
from model import SASRec


def main():
    # 加载数据
    data_path = os.path.join("data", "ml-1m", "processed.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # 超参数
    max_len = 200
    embed_dim = 64
    num_heads = 2
    num_layers = 2
    dropout = 0.2
    batch_size = 64
    num_items = data["num_items"]

    print(f"模型参数: num_items={num_items}, max_len={max_len}, "
          f"embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}")

    # 创建模型
    model = SASRec(num_items, max_len, embed_dim, num_heads, num_layers, dropout)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 构建一个 batch
    train_dataset = SASRecDataset(data["train_seqs"], max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_seq, target_seq = next(iter(train_loader))

    print(f"\n输入形状: {input_seq.shape}")
    print(f"目标形状: {target_seq.shape}")

    # 测试前向传播
    model.eval()
    with torch.no_grad():
        # 测试 forward
        hidden = model(input_seq)
        print(f"隐藏层输出形状: {hidden.shape}")

        # 测试 compute_loss
        model.train()
        loss = model.compute_loss(input_seq, target_seq)
        print(f"损失值: {loss.item():.4f}")

        # 测试 predict（全量物品打分）
        model.eval()
        scores = model.predict(input_seq)
        print(f"预测分数形状: {scores.shape}")
        print(f"Top-5 推荐: {torch.topk(scores[0], 5).indices.tolist()}")

    print("\n✓ 模型测试全部通过")


if __name__ == "__main__":
    main()

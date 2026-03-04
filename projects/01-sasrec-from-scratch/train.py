"""
train.py - SASRec 训练与评估脚本
功能: 训练模型，每个 epoch 后在验证集上评估，最后在测试集上报告最终结果
"""

import os
import pickle
import time
import math
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import SASRecDataset, EvalDataset
from model import SASRec


# ============================================================
# 超参数配置
# ============================================================
class Config:
    # 数据
    data_path = os.path.join("data", "ml-1m", "processed.pkl")
    max_len = 200

    # 模型
    embed_dim = 64
    num_heads = 2
    num_layers = 2
    dropout = 0.2

    # 训练
    batch_size = 128
    lr = 1e-3
    epochs = 30
    device = "cpu"

    # 评估
    topk_list = [5, 10, 20]


# ============================================================
# 评估函数
# ============================================================
def evaluate(model, eval_dataset, num_items, config, train_seqs):
    """
    在验证集或测试集上评估模型。

    【修改】新增 train_seqs 参数，用于在评估时排除用户已交互的物品。
    """
    model.eval()
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    all_ranks = []

    with torch.no_grad():
        for input_seq, target in eval_loader:
            input_seq = input_seq.to(config.device)
            target = target.to(config.device)

            # 对所有物品打分: (batch_size, num_items)
            scores = model.predict(input_seq)

            # 【新增】把用户已交互的物品分数设为负无穷，排除它们对排名的干扰
            # 获取当前 batch 中每个用户的 ID
            batch_start = len(all_ranks)
            for i in range(input_seq.size(0)):
                user_idx = batch_start + i
                if user_idx < len(eval_dataset.users):
                    user = eval_dataset.users[user_idx]
                    if user in train_seqs:
                        # 已交互的物品 ID 从 1 开始，scores 列索引从 0 开始
                        interacted = torch.tensor(
                            [item - 1 for item in train_seqs[user]],
                            dtype=torch.long,
                            device=config.device,
                        )
                        scores[i, interacted] = float("-inf")

            # 获取目标物品的分数
            target_idx = target - 1
            target_scores = scores.gather(1, target_idx.unsqueeze(1))

            # 计算排名
            ranks = (scores >= target_scores).sum(dim=1)

            # 【新增】排名保护: 确保排名至少为 1（防止异常情况）
            ranks = ranks.clamp(min=1)

            all_ranks.extend(ranks.cpu().tolist())

    all_ranks = np.array(all_ranks, dtype=np.float64)

    # 计算各个 K 值下的 HR 和 NDCG
    metrics = {}
    for k in config.topk_list:
        hit = (all_ranks <= k).astype(float)
        hr = hit.mean()
        # 【修复】NDCG 计算: all_ranks 最小为 1, log2(1+1)=1.0, 不会出现除以零
        ndcg = (hit / np.log2(all_ranks + 1)).mean()

        metrics[f"HR@{k}"] = hr
        metrics[f"NDCG@{k}"] = ndcg

    return metrics


# ============================================================
# 格式化输出指标
# ============================================================
def format_metrics(metrics):
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    return "  ".join(parts)


# ============================================================
# 主训练流程
# ============================================================
def train(config):
    # ---- 1. 加载数据 ----
    print("加载数据...")
    with open(config.data_path, "rb") as f:
        data = pickle.load(f)

    num_items = data["num_items"]
    train_seqs = data["train_seqs"]  # 【新增】保存训练序列，评估时需要
    train_dataset = SASRecDataset(train_seqs, config.max_len)
    valid_dataset = EvalDataset(data["valid_data"], config.max_len)
    test_dataset = EvalDataset(data["test_data"], config.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(valid_dataset)}, 测试样本: {len(test_dataset)}")
    print(f"物品数: {num_items}, 最大序列长度: {config.max_len}")

    # ---- 2. 创建模型和优化器 ----
    model = SASRec(
        num_items=num_items,
        max_len=config.max_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"设备: {config.device}")
    print(f"{'='*70}")

    # ---- 3. 训练循环 ----
    best_hr10 = 0.0
    best_epoch = 0
    best_metrics = {}

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for input_seq, target_seq in train_loader:
            input_seq = input_seq.to(config.device)
            target_seq = target_seq.to(config.device)

            loss = model.compute_loss(input_seq, target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start

        # ---- 4. 验证集评估 ----
        # 【修改】传入 train_seqs
        valid_metrics = evaluate(model, valid_dataset, num_items, config, train_seqs)

        print(f"Epoch {epoch:3d}/{config.epochs} | "
              f"Loss={avg_loss:.4f} | "
              f"Time={epoch_time:.1f}s | "
              f"{format_metrics(valid_metrics)}")

        if valid_metrics["HR@10"] > best_hr10:
            best_hr10 = valid_metrics["HR@10"]
            best_epoch = epoch
            best_metrics = valid_metrics.copy()
            torch.save(model.state_dict(), "best_model.pt")

    print(f"{'='*70}")
    print(f"最佳验证结果 (Epoch {best_epoch}): {format_metrics(best_metrics)}")

    # ---- 5. 测试集评估 ----
    # 【注意】测试时，用户可见的历史包括训练序列 + 验证物品
    # 所以需要把验证物品也加入已交互集合
    test_train_seqs = {}
    for user, seq in train_seqs.items():
        test_train_seqs[user] = seq.copy()
        if user in data["valid_data"]:
            test_train_seqs[user].append(data["valid_data"][user][1])

    print(f"\n加载最佳模型，在测试集上评估...")
    model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    test_metrics = evaluate(model, test_dataset, num_items, config, test_train_seqs)
    print(f"测试集结果: {format_metrics(test_metrics)}")


if __name__ == "__main__":
    config = Config()
    train(config)

"""
train.py - BERT4Rec 训练与评估脚本
"""

import os
import pickle
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import BERT4RecDataset, BERT4RecEvalDataset
from model import BERT4Rec


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

    # BERT4Rec 特有
    mask_prob = 0.2  # ML-1m 是稠密数据集，论文推荐 0.2

    # 训练
    batch_size = 128
    lr = 1e-3
    epochs = 100
    device = "cpu"

    # 评估
    topk_list = [5, 10, 20]


# ============================================================
# 评估函数
# ============================================================
def evaluate(model, eval_dataset, num_items, config, train_seqs):
    """在验证集或测试集上评估模型（与 SASRec 的评估逻辑相同）"""
    model.eval()
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    all_ranks = []

    with torch.no_grad():
        for input_seq, target in eval_loader:
            input_seq = input_seq.to(config.device)
            target = target.to(config.device)

            scores = model.predict(input_seq)

            # 排除已交互的物品
            batch_start = len(all_ranks)
            for i in range(input_seq.size(0)):
                user_idx = batch_start + i
                if user_idx < len(eval_dataset.users):
                    user = eval_dataset.users[user_idx]
                    if user in train_seqs:
                        interacted = torch.tensor(
                            [item - 1 for item in train_seqs[user]],
                            dtype=torch.long,
                            device=config.device,
                        )
                        scores[i, interacted] = float("-inf")

            target_idx = target - 1
            target_scores = scores.gather(1, target_idx.unsqueeze(1))
            ranks = (scores >= target_scores).sum(dim=1)
            ranks = ranks.clamp(min=1)
            all_ranks.extend(ranks.cpu().tolist())

    all_ranks = np.array(all_ranks, dtype=np.float64)

    metrics = {}
    for k in config.topk_list:
        hit = (all_ranks <= k).astype(float)
        hr = hit.mean()
        ndcg = (hit / np.log2(all_ranks + 1)).mean()
        metrics[f"HR@{k}"] = hr
        metrics[f"NDCG@{k}"] = ndcg

    return metrics


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
    train_seqs = data["train_seqs"]
    mask_token = num_items + 1

    train_dataset = BERT4RecDataset(train_seqs, config.max_len, config.mask_prob, num_items)
    valid_dataset = BERT4RecEvalDataset(data["valid_data"], config.max_len, mask_token)
    test_dataset = BERT4RecEvalDataset(data["test_data"], config.max_len, mask_token)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(valid_dataset)}, 测试样本: {len(test_dataset)}")
    print(f"物品数: {num_items}, Mask Token ID: {mask_token}, 最大序列长度: {config.max_len}")
    print(f"Mask 概率: {config.mask_prob}")

    # ---- 2. 创建模型和优化器 ----
    model = BERT4Rec(
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

        for masked_seq, labels in train_loader:
            masked_seq = masked_seq.to(config.device)
            labels = labels.to(config.device)

            loss = model.compute_loss(masked_seq, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start

        # ---- 4. 验证 ----
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

    # ---- 5. 测试 ----
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
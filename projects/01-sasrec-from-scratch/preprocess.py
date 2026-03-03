"""
preprocess.py - MovieLens-1M 数据预处理
功能: 读取原始评分数据 → 转化为用户行为序列 → 划分训练/验证/测试集
"""

import os
import pickle
from collections import defaultdict


def load_ratings(data_dir):
    """
    读取 ratings.dat 文件，返回原始交互记录列表。
    每条记录格式: (user_id, item_id, timestamp)
    评分值被丢弃（转化为隐式反馈）。
    """
    ratings_path = os.path.join(data_dir, "ratings.dat")
    interactions = []

    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            user_id = int(parts[0])
            item_id = int(parts[1])
            # parts[2] 是评分值，这里丢弃
            timestamp = int(parts[3])
            interactions.append((user_id, item_id, timestamp))

    print(f"原始交互记录数: {len(interactions)}")
    return interactions


def filter_kcore(interactions, k=5):
    """
    k-core 过滤: 迭代地移除交互次数少于 k 次的用户和物品。
    为什么要迭代？因为移除某些用户后，某些物品的交互次数可能也会降到 k 以下，
    反之亦然，所以需要反复过滤直到稳定。
    """
    while True:
        # 统计每个用户和物品的交互次数
        user_count = defaultdict(int)
        item_count = defaultdict(int)
        for user, item, _ in interactions:
            user_count[user] += 1
            item_count[item] += 1

        # 过滤
        filtered = [
            (u, i, t) for u, i, t in interactions
            if user_count[u] >= k and item_count[i] >= k
        ]

        # 如果没有变化，说明已经稳定
        if len(filtered) == len(interactions):
            break
        interactions = filtered

    print(f"5-core 过滤后交互记录数: {len(interactions)}")
    return interactions


def build_user_sequences(interactions):
    """
    将交互记录转化为每个用户的行为序列（按时间排序）。
    同时将用户ID和物品ID重新映射为从1开始的连续整数（0留给padding）。
    """
    # 按用户分组，每个用户的交互按时间排序
    user_interactions = defaultdict(list)
    for user, item, timestamp in interactions:
        user_interactions[user].append((item, timestamp))

    for user in user_interactions:
        user_interactions[user].sort(key=lambda x: x[1])  # 按时间排序

    # 重新映射物品ID: 原始ID可能不连续（如1, 5, 100），映射为连续的1, 2, 3...
    # 0 保留给 padding（序列填充）
    all_items = set()
    for user in user_interactions:
        for item, _ in user_interactions[user]:
            all_items.add(item)

    item_to_idx = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
    num_items = len(item_to_idx)

    # 重新映射用户ID
    user_to_idx = {user: idx + 1 for idx, user in enumerate(sorted(user_interactions.keys()))}
    num_users = len(user_to_idx)

    # 构建最终的用户行为序列（只保留物品ID序列，不再需要时间戳）
    user_sequences = {}
    for user, items_with_time in user_interactions.items():
        user_idx = user_to_idx[user]
        item_seq = [item_to_idx[item] for item, _ in items_with_time]
        user_sequences[user_idx] = item_seq

    print(f"用户数: {num_users}, 物品数: {num_items}")
    print(f"平均序列长度: {sum(len(s) for s in user_sequences.values()) / len(user_sequences):.1f}")

    return user_sequences, num_users, num_items


def split_dataset(user_sequences):
    """
    Leave-One-Out 划分:
    - 训练集: 每个用户序列的 [0, ..., -3] (即去掉最后两个)
    - 验证集: 每个用户序列的倒数第二个物品
    - 测试集: 每个用户序列的最后一个物品

    注意: 序列长度不足3的用户无法划分，会被跳过。
    """
    train_seqs = {}
    valid_data = {}  # {user: (训练部分序列, 验证目标物品)}
    test_data = {}   # {user: (训练+验证部分序列, 测试目标物品)}

    skipped = 0
    for user, seq in user_sequences.items():
        if len(seq) < 3:
            skipped += 1
            continue

        train_seqs[user] = seq[:-2]              # 训练用的序列
        valid_data[user] = (seq[:-2], seq[-2])    # 验证: 输入训练序列, 预测倒数第2个
        test_data[user] = (seq[:-1], seq[-1])     # 测试: 输入训练+验证, 预测最后1个

    print(f"有效用户数: {len(train_seqs)}, 跳过(序列过短): {skipped}")
    return train_seqs, valid_data, test_data


def main():
    data_dir = os.path.join("data", "ml-1m")

    # 1. 读取原始数据
    interactions = load_ratings(data_dir)

    # 2. 5-core 过滤
    interactions = filter_kcore(interactions, k=5)

    # 3. 构建用户行为序列
    user_sequences, num_users, num_items = build_user_sequences(interactions)

    # 4. 划分数据集
    train_seqs, valid_data, test_data = split_dataset(user_sequences)

    # 5. 保存处理结果
    output = {
        "train_seqs": train_seqs,
        "valid_data": valid_data,
        "test_data": test_data,
        "num_users": num_users,
        "num_items": num_items,
    }

    output_path = os.path.join(data_dir, "processed.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\n处理完成! 已保存到 {output_path}")
    print(f"数据概览:")
    print(f"  用户数: {num_users}")
    print(f"  物品数: {num_items}")
    print(f"  训练用户数: {len(train_seqs)}")

    # 打印一个样例，帮助你直观理解数据格式
    sample_user = list(train_seqs.keys())[0]
    print(f"\n样例 - 用户 {sample_user}:")
    print(f"  训练序列: {train_seqs[sample_user][:10]}... (长度 {len(train_seqs[sample_user])})")
    print(f"  验证目标: {valid_data[sample_user][1]}")
    print(f"  测试目标: {test_data[sample_user][1]}")


if __name__ == "__main__":
    main()
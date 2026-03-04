# 01 - SASRec 从零实现

## 项目目标

在 MovieLens-1M 数据集上，从零实现 SASRec（Self-Attentive Sequential Recommendation），跑通完整的"数据预处理 → 模型定义 → 训练 → 评估"流程。

## 数据集

- **名称**: MovieLens-1M
- **下载地址**: https://grouplens.org/datasets/movielens/1m/
- **规模**: 约 100 万条评分记录，6000+ 用户，3700+ 电影
- **预处理**: 隐式反馈（丢弃评分值）→ 5-core 过滤 → 按时间排序 → Leave-One-Out 划分

### 数据准备

```bash
# 在项目目录下执行
mkdir -p data/ml-1m
cd data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
cd ..
python preprocess.py
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `preprocess.py` | 数据预处理：读取原始数据 → 5-core 过滤 → 构建用户序列 → 划分数据集 |
| `dataset.py` | PyTorch Dataset 定义（待实现） |
| `model.py` | SASRec 模型实现（待实现） |
| `train.py` | 训练脚本（待实现） |
| `evaluate.py` | 评估脚本：HR@K, NDCG@K（待实现） |

## 运行方式

```bash
# 1. 数据预处理
python preprocess.py

# 2. 训练（自动在每个 epoch 后评估）
python train.py
```

## 预处理结果

始交互记录数: 1,000,209
5-core 过滤后: 999,611
用户数: 6,040 | 物品数: 3,416 | 平均序列长度: 165.5

## 实验结果

| 指标 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|------|------|--------|-------|---------|-------|---------|
| 验证集 | 0.7826 | 0.7687 | 0.8036 | 0.7755 | 0.8280 | 0.7816 |
| 测试集 | 0.7743 | 0.7636 | 0.7934 | 0.7697 | 0.8161 | 0.7755 |

超参数: embed_dim=64, num_heads=2, num_layers=2, dropout=0.2, lr=1e-3, max_len=200, epochs=30

说明: 采用全量评估（非采样），评估时排除用户已交互物品。
由于 MovieLens-1M 非常稠密（平均每用户交互 165/3416 部电影），
全量评估下的指标数值会高于采样评估，不宜与采用采样评估的论文结果直接比较。

## 参考

- Kang, W. C., & McAuley, J. (2018). Self-attentive sequential recommendation. *In Proceedings of the IEEE International Conference on Data Mining (ICDM)*.

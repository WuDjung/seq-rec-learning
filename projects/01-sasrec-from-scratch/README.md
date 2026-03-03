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

> 待补充（运行 preprocess.py 后在此记录输出）

## 实验结果

> 待补充

## 参考

- Kang, W. C., & McAuley, J. (2018). Self-attentive sequential recommendation. *In Proceedings of the IEEE International Conference on Data Mining (ICDM)*.

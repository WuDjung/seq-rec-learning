### 主仓库 `README.md`

# 序列推荐学习之路

从零开始学习序列推荐（Sequential Recommendation），涵盖理论笔记、论文精读和代码实现。

## 学习路线

### 第一阶段：基础入门（进行中）

- [x] 理解序列推荐任务定义
- [ ] 从零实现 SASRec（MovieLens-1M）
- [ ] 精读 SASRec 论文
- [ ] 精读 BERT4Rec 论文

### 第二阶段：生成式推荐

- [ ] 精读 HSTU 论文
- [ ] 精读 TIGER 论文
- [ ] 理解语义 ID（Semantic ID）

### 第三阶段：LLM + 推荐

- [ ] 调研 LLM 增强推荐的主流方案
- [ ] 实现 LLM 特征增强的序列推荐

## 项目列表

| # | 项目 | 说明 | 状态 |
|---|------|------|------|
| 01 | [SASRec 从零实现](projects/01-sasrec-from-scratch/) | 在 MovieLens-1M 上实现完整的 SASRec 训练和评估流程 | 进行中 |
| 02 | [BERT4Rec 实现](projects/02-bert4rec-impl/) | 待开始 | 未开始 |

## 笔记列表

| # | 笔记 | 主题 |
|---|------|------|
| 01 | [序列推荐概述](notes/01-序列推荐概述.md) | 任务定义、经典模型演进、评估指标 |
| 02 | [SASRec 论文精读](notes/02-SASRec论文精读.md) | 模型结构、训练策略、实验分析 |
| 03 | [BERT4Rec 论文精读](notes/03-BERT4Rec论文精读.md) | 双向注意力、Masked Item Prediction |

## 环境配置

```bash
# 克隆仓库
git clone https://github.com/WuDjung/seq-rec-learning.git
cd seq-rec-learning

# 安装依赖
pip install -r requirements.txt
```

## 参考资料

- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (ICDM 2018)
- [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690) (CIKM 2019)
- [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) (ICML 2024)
- [MovieLens-1M 数据集](https://grouplens.org/datasets/movielens/1m/)



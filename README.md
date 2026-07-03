# ml-project-power-forecast

本项目为机器学习课程作业，聚焦于家庭电力消耗预测任务。项目基于UCI公开数据集，实现了LSTM、Transformer以及自行设计的PEAT（Periodic-Enhanced Attentional Transformer）模型，并对各模型的预测性能进行了系统性的对比与分析。项目代码涵盖完整的数据预处理、模型训练、评估与可视化流程，所有实验均支持五轮随机种子重复验证。


## 项目结构
```

ml-project-power-forecast/
├── data/                          # 数据集
│   ├── train.csv                  # 训练集（前987天）
│   └── test.csv                   # 测试集（后455天）
│
├── logs/                          # 训练日志
│   ├── lstm_90_log.png            # LSTM-90天损失日志
│   ├── lstm_365_log.png           # LSTM-365天损失日志
│   ├── lstm_90_add_log.png        # LSTM+TimeFeat-90天损失日志
│   ├── lstm_365_add_log.png       # LSTM+TimeFeat-365天损失日志
│   ├── transform_90_log.png       # Transformer-90天损失日志
│   ├── transform_365_log.png      # Transformer-365天损失日志
│   ├── transform_90_add_log.png   # Transformer+TimeFeat-90天损失日志
│   ├── transform_365_add_log.png  # Transformer+TimeFeat-365天损失日志
│   ├── cnn_transformer_90_log.png # CNN+Transformer-90天损失日志
│   ├── cnn_transformer_365_log.png# CNN+Transformer-365天损失日志
│   ├── cnn_transformer_timefeat_90_log.png   # CNN+Transformer+TimeFeat-90天
│   ├── cnn_transformer_timefeat_365_log.png  # CNN+Transformer+TimeFeat-365天
│   ├── attncnn_transformer_90_log.png        # AttnCNN+Transformer-90天
│   └── attncnn_transformer_365_log.png       # AttnCNN+Transformer-365天
│
├── results/                       # 预测结果曲线
│   ├── lstm_prediction.png
│   ├── lstm_timefeat_prediction.png
│   ├── transformer_prediction.png
│   ├── transformer_timefeat_prediction.png
│   ├── cnn_only_prediction.png
│   ├── one_cnn_periodic_transformer.png
│   └── attn_cnn_periodic_transformer.png
│
├── structures/                    # 模型结构图
│   └── ours.png                   # PEAT模型结构图
│
├── train_lstm.py                  # LSTM基准模型
├── train_lstm_timefeat.py         # LSTM + 时间特征
├── train_transformer.py           # Transformer基准模型
├── train_transformer_timefeat.py  # Transformer + 时间特征
├── train_cnn_only.py              # CNN + Transformer（基准）
├── train_cnn_timefeat.py          # CNN + Transformer + TimeFeat
├── train_attn_only.py             # AttnCNN + Transformer（无TimeFeat）
├── train_cnn_attention_timefeat.py # PEAT完整模型（AttnCNN + TimeFeat + Transformer）
├── train_only_attn.py             # 注意力消融实验
├── data_preprocessing.ipynb       # 数据预处理Jupyter Notebook
├── train.sh                       # 批量训练脚本
├── .gitignore
└── README.md

```


## 环境配置

### 推荐环境

- Python 3.10+
- CUDA 11.8+（如使用GPU）
- 建议使用虚拟环境：

```bash
conda create -n ml_power python=3.10
conda activate ml_power
```

### 依赖安装

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## 数据集

本实验采用UCI公开数据集：**Individual Household Electric Power Consumption**。

原始数据为分钟级粒度，记录了法国一户家庭从2006年12月至2010年11月的用电数据。经日级别汇总后，共得到1442条样本。

**数据划分：**

- 训练集：前987天（2006-12-16 至 2009-08-28）
- 测试集：后455天（2009-08-29 至 2010-11-26）

**预处理说明：** 请参考 `data_preprocessing.ipynb` 将原始UCI数据转换为 `train.csv` 和 `test.csv`，并确保数据目录结构正确。主要处理步骤包括日期解析、按天聚合、特征筛选与归一化。

## 模型说明

### 1. LSTM基准模型

- 网络结构：2层LSTM，隐藏状态128维，Dropout 0.2
- 输入：过去90天的12维原始特征
- 输出：未来90天或365天的全局有功功率
- 训练脚本：`train_lstm.py`

### 2. Transformer基准模型

- 网络结构：2层Transformer编码器，4头自注意力，64维嵌入
- 位置编码：标准正弦余弦位置编码
- 输入：过去90天的12维原始特征
- 输出：未来90天或365天的全局有功功率
- 训练脚本：`train_transformer.py`

### 3. PEAT（Periodic-Enhanced Attentional Transformer）

- 核心设计：
  - **周期感知嵌入**：星期和月份分别通过可学习嵌入表编码为16维向量
  - **注意力CNN模块**：一维卷积提取局部时序特征，注意力机制自适应加权聚合
  - **融合机制**：CNN聚合特征与周期嵌入拼接融合后送入Transformer编码器
- 训练脚本：`train_peat.py`

### 4. 消融实验

| 实验配置                            | 脚本                                            |
| ----------------------------------- | ----------------------------------------------- |
| CNN + Transformer（基准）           | `train_cnn_only.py`                           |
| CNN + Transformer + TimeFeat        | `train_cnn_timefeat.py`                       |
| AttnCNN + Transformer（无TimeFeat） | `train_attn_only.py` / `train_only_attn.py` |
| PEAT（完整模型）                    | `train_peat.py`                               |

## 快速开始

### 1. 准备数据

运行 `data_preprocessing.ipynb` 对原始数据进行预处理，生成 `train.csv` 和 `test.csv`。

### 2. 训练模型

使用 `train.sh` 批量运行所有实验，或单独运行某个模型：

```bash
# 训练LSTM基准模型
python train_lstm.py

# 训练Transformer基准模型
python train_transformer.py

# 训练PEAT完整模型
python train_peat.py
```

### 3. 输出结果

- 训练日志：自动保存至 `logs/` 目录
- 预测曲线：自动保存至 `results/` 目录
- 控制台输出：包含五轮实验的MSE/MAE均值与标准差

## 训练策略

所有模型采用统一的训练策略：

- 损失函数：均方误差（MSE）
- 优化器：Adam，学习率 5×10⁻⁴
- 学习率调度：ReduceLROnPlateau（patience=4，因子0.5）
- 早停机制：patience=12
- 梯度裁剪：max_norm=1.0
- 批大小：32
- 最大轮数：120
- 重复实验：5个随机种子（42, 123, 2024, 7, 99）

## 实验结果

各模型的详细性能对比与分析，请参见实验报告。

| 模型                   | 90天 MAE                 | 365天 MAE                |
| ---------------------- | ------------------------ | ------------------------ |
| LSTM                   | 407.84 ± 38.06          | 366.56 ± 28.45          |
| Transformer            | 420.43 ± 41.05          | 388.32 ± 32.47          |
| **PEAT（Ours）** | **294.59 ± 5.14** | **313.86 ± 8.96** |

## 引用

本实验使用的数据集：

```
Individual Household Electric Power Consumption
UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
```

## 作者与致谢

- 课程：机器学习
- 数据来源：UCI Machine Learning Repository

---

**如有任何问题，欢迎通过GitHub Issues联系。**

```

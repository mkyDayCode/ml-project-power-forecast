# ml-project-power-forecast

本项目为机器学习课程作业，聚焦于家庭电力消耗预测任务。项目基于UCI公开数据集，实现了LSTM、Transformer以及自行设计的PEAT（Periodic-Enhanced Attentional Transformer）模型，并对各模型的预测性能进行了系统性的对比与分析。项目代码涵盖完整的数据预处理、模型训练、评估与可视化流程，所有实验均支持五轮随机种子重复验证。

## 项目结构
ml-project-power-forecast/
├── data/ # 数据集
│ ├── train.csv # 训练集（前987天）
│ └── test.csv # 测试集（后455天）
│
├── logs/ # 训练日志
│ ├── lstm_90_log.png # LSTM-90天损失日志
│ ├── lstm_365_log.png # LSTM-365天损失日志
│ ├── lstm_90_add_log.png # LSTM+TimeFeat-90天损失日志
│ ├── lstm_365_add_log.png # LSTM+TimeFeat-365天损失日志
│ ├── transform_90_log.png # Transformer-90天损失日志
│ ├── transform_365_log.png # Transformer-365天损失日志
│ ├── transform_90_add_log.png # Transformer+TimeFeat-90天损失日志
│ ├── transform_365_add_log.png # Transformer+TimeFeat-365天损失日志
│ ├── cnn_transformer_90_log.png # CNN+Transformer-90天损失日志
│ ├── cnn_transformer_365_log.png# CNN+Transformer-365天损失日志
│ ├── cnn_transformer_timefeat_90_log.png # CNN+Transformer+TimeFeat-90天
│ ├── cnn_transformer_timefeat_365_log.png # CNN+Transformer+TimeFeat-365天
│ ├── attncnn_transformer_90_log.png # AttnCNN+Transformer-90天
│ └── attncnn_transformer_365_log.png # AttnCNN+Transformer-365天
│
├── results/ # 预测结果曲线
│ ├── lstm_prediction.png
│ ├── lstm_timefeat_prediction.png
│ ├── transformer_prediction.png
│ ├── transformer_timefeat_prediction.png
│ ├── cnn_only_prediction.png
│ ├── one_cnn_periodic_transformer.png
│ └── attn_cnn_periodic_transformer.png
│
├── structures/ # 模型结构图
│ └── ours.png # PEAT模型结构图
│
├── train_lstm.py # LSTM基准模型
├── train_lstm_timefeat.py # LSTM + 时间特征
├── train_transformer.py # Transformer基准模型
├── train_transformer_timefeat.py # Transformer + 时间特征
├── train_cnn_only.py # CNN + Transformer（基准）
├── train_cnn_timefeat.py # CNN + Transformer + TimeFeat
├── train_attn_only.py # AttnCNN + Transformer（无TimeFeat）
├── train_cnn_attention_timefeat.py # PEAT完整模型（AttnCNN + TimeFeat + Transformer）
├── train_only_attn.py # 注意力消融实验
├── data_preprocessing.ipynb # 数据预处理Jupyter Notebook
├── train.sh # 批量训练脚本
├── .gitignore
└── README.md

text

## 环境配置

```bash
# 创建conda环境
conda create -n ml_power python=3.8
conda activate ml_power

# 安装依赖
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib
pip install scikit-learn
使用说明
bash
# 训练LSTM基准模型
python train_lstm.py

# 训练Transformer
python train_transformer.py

# 训练PEAT完整模型
python train_cnn_attention_timefeat.py
模型列表
模型	描述
LSTM	LSTM基准模型
LSTM + TimeFeat	LSTM + 时间特征
Transformer	Transformer基准模型
Transformer + TimeFeat	Transformer + 时间特征
CNN + Transformer	CNN + Transformer（基准）
CNN + Transformer + TimeFeat	PEAT完整模型
AttnCNN + Transformer	Attention增强CNN + Transformer
PEAT	Periodic-Enhanced Attentional Transformer
实验结果
[待补充]

许可证
MIT License

text

---

## 如何正确保存为 UTF-8

### 在 VSCode 中：
1. 点击右下角的编码（比如显示 "UTF-8" 或 "GBK"）
2. 选择 **"Save with Encoding"**
3. 选择 **"UTF-8"**

### 在命令行中：
```bash
# 查看当前编码
file -i README.md

# 如果不是 UTF-8，转换
iconv -f GBK -t UTF-8 README.md > README_new.md
mv README_new.md README.md
提交到 GitHub
bash
git add README.md
git commit -m "更新README文档"
git push origin master

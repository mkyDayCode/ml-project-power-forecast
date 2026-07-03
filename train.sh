#!/bin/bash
#SBATCH --job-name=machine_learning  # 修改为你的项目名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=72:00:00
#SBATCH -o logs/0626_all_%j.log              # 使用作业ID

echo "=== 启动时间: $(date) ==="
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA 可用，GPU 数量：{torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    显存：{torch.cuda.get_device_properties(i).total_memory/1e9:.2f} GB')
        print(f'    计算能力：{torch.cuda.get_device_capability(i)}')
else:
    print('CUDA 不可用')
"
echo "================================"
# 禁用 Python 字节码缓存
export PYTHONDONTWRITEBYTECODE=1
# 1. 加载Conda
source /public/home/yangzhe/miniconda3/etc/profile.d/conda.sh

# 2. 激活你的hllm环境
conda activate llm2rec
# 3. 验证环境
echo "=== 环境信息 ==="
echo "Python路径: $(which python3)"
echo "Python版本: $(python3 --version)"
echo "Conda环境: $(conda info --envs | grep '*')"
echo ""


echo "running at $(date)" 
# echo "using lstm"
# python train_lstm.py

# echo "test my model"
# python train_embed_transformer.py

python train_transformer.py
python train_transformer_add.py

python train_lstm.py
python train_lstm_add.py

python train_embed_transformer.py
python train_embed_transformer_add.py

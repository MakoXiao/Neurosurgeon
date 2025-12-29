#!/bin/bash

# 实验启动脚本 - 后台运行版本
# 自动记录所有日志和进度

set -e

PROJECT_ROOT="/opt/03-ai/01-proj/Neurosurgeon"
cd $PROJECT_ROOT

# 激活虚拟环境
source neurosurgeon_env/bin/activate

# 创建日志目录
mkdir -p logs

echo "=================================="
echo "实验开始时间: $(date)"
echo "=================================="

# 记录环境信息
echo "环境信息:" | tee -a logs/experiment.log
nvidia-smi | tee -a logs/experiment.log
echo "" | tee -a logs/experiment.log

# 步骤1: 训练所有模型 (预计8小时)
echo "=================================="
echo "步骤 1/4: 训练分类模型"
echo "开始时间: $(date)"
echo "=================================="

python train_models.py \
    --model all \
    --data_dir data/caltech-101 \
    --save_dir checkpoints \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --device cuda \
    2>&1 | tee logs/train_models.log

echo "模型训练完成时间: $(date)"
echo ""

# 步骤2: 训练RL智能体 (预计12小时)
echo "=================================="
echo "步骤 2/4: 训练RL智能体"
echo "开始时间: $(date)"
echo "=================================="

python train_rl_agent.py \
    --model all \
    --data_dir data/caltech-101 \
    --checkpoint_dir checkpoints \
    --save_dir rl_agents \
    --episodes 1000 \
    --max_steps 100 \
    --update_interval 10 \
    --device cuda \
    2>&1 | tee logs/train_rl_agents.log

echo "RL训练完成时间: $(date)"
echo ""

# 步骤3: 运行对比实验 (预计2小时)
echo "=================================="
echo "步骤 3/4: 运行对比实验"
echo "开始时间: $(date)"
echo "=================================="

python experiments/compare_methods.py \
    --model all \
    --data_dir data/caltech-101 \
    --checkpoint_dir checkpoints \
    --rl_agent_dir rl_agents \
    --save_dir results \
    --num_samples 200 \
    --device cuda \
    2>&1 | tee logs/compare_methods.log

echo "对比实验完成时间: $(date)"
echo ""

# 步骤4: 生成可视化
echo "=================================="
echo "步骤 4/4: 生成可视化图表"
echo "开始时间: $(date)"
echo "=================================="

python experiments/visualize_results.py \
    --results_dir results \
    --save_dir figures \
    2>&1 | tee logs/visualize.log

echo "可视化完成时间: $(date)"
echo ""

# 完成
echo "=================================="
echo "所有实验完成!"
echo "完成时间: $(date)"
echo "=================================="
echo ""
echo "结果位置:"
echo "  - 模型检查点: checkpoints/"
echo "  - RL智能体: rl_agents/"
echo "  - 实验结果: results/"
echo "  - 可视化图表: figures/"
echo ""
echo "查看详细日志:"
echo "  - 模型训练: logs/train_models.log"
echo "  - RL训练: logs/train_rl_agents.log"
echo "  - 对比实验: logs/compare_methods.log"
echo "  - 可视化: logs/visualize.log"
echo ""



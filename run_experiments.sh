#!/bin/bash

# 完整实验流程脚本
# 包括模型训练、RL智能体训练、对比实验和结果可视化

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始完整实验流程"
echo "=========================================="

# 项目根目录
PROJECT_ROOT="/opt/03-ai/01-proj/Neurosurgeon"
VENV_PATH="$PROJECT_ROOT/neurosurgeon_env"

# 激活虚拟环境
echo "激活虚拟环境: $VENV_PATH"
source $VENV_PATH/bin/activate

# 配置参数
DATA_DIR="$PROJECT_ROOT/data/caltech-101"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
RL_AGENT_DIR="$PROJECT_ROOT/rl_agents"
RESULTS_DIR="$PROJECT_ROOT/results"
FIGURES_DIR="$PROJECT_ROOT/figures"

# 设备配置
DEVICE="cuda"  # 或 "cpu"

# 训练参数
TRAIN_EPOCHS=50
TRAIN_BATCH_SIZE=32
TRAIN_LR=0.001

# RL训练参数
RL_EPISODES=1000
RL_MAX_STEPS=100
RL_UPDATE_INTERVAL=10

# 对比实验参数
NUM_SAMPLES=200

# 创建必要的目录
mkdir -p $CHECKPOINT_DIR
mkdir -p $RL_AGENT_DIR
mkdir -p $RESULTS_DIR
mkdir -p $FIGURES_DIR

echo ""
echo "=========================================="
echo "步骤 1: 训练分类模型"
echo "=========================================="
echo "训练 ResNet18, VGG11, MobileNetV2, AlexNet"
echo ""

cd /opt/03-ai/01-proj/Neurosurgeon

python train_models.py \
    --model all \
    --data_dir $DATA_DIR \
    --save_dir $CHECKPOINT_DIR \
    --epochs $TRAIN_EPOCHS \
    --batch_size $TRAIN_BATCH_SIZE \
    --lr $TRAIN_LR \
    --device $DEVICE

echo ""
echo "模型训练完成!"
echo ""

echo "=========================================="
echo "步骤 2: 训练RL智能体"
echo "=========================================="
echo "为每个模型训练混合动作空间PPO智能体"
echo ""

python train_rl_agent.py \
    --model all \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --save_dir $RL_AGENT_DIR \
    --episodes $RL_EPISODES \
    --max_steps $RL_MAX_STEPS \
    --update_interval $RL_UPDATE_INTERVAL \
    --device $DEVICE

echo ""
echo "RL智能体训练完成!"
echo ""

echo "=========================================="
echo "步骤 3: 运行对比实验"
echo "=========================================="
echo "对比不同方法和创新点的性能"
echo ""

python experiments/compare_methods.py \
    --model all \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --rl_agent_dir $RL_AGENT_DIR \
    --save_dir $RESULTS_DIR \
    --num_samples $NUM_SAMPLES \
    --device $DEVICE

echo ""
echo "对比实验完成!"
echo ""

echo "=========================================="
echo "步骤 4: 生成可视化图表"
echo "=========================================="
echo "生成论文级别的效果对比图表"
echo ""

python experiments/visualize_results.py \
    --results_dir $RESULTS_DIR \
    --save_dir $FIGURES_DIR

echo ""
echo "可视化完成!"
echo ""

echo "=========================================="
echo "实验流程全部完成!"
echo "=========================================="
echo ""
echo "结果位置:"
echo "  - 模型检查点: $CHECKPOINT_DIR"
echo "  - RL智能体: $RL_AGENT_DIR"
echo "  - 实验结果: $RESULTS_DIR"
echo "  - 可视化图表: $FIGURES_DIR"
echo ""
echo "=========================================="


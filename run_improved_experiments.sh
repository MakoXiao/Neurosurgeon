#!/bin/bash

################################################################################
#                    改进实验自动化脚本
################################################################################

set -e  # 出错时退出

PROJECT_DIR="/opt/03-ai/01-proj/Neurosurgeon"
cd "$PROJECT_DIR"

# 激活虚拟环境
source neurosurgeon_env/bin/activate

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    改进实验自动化流程${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# 创建日志目录
mkdir -p logs/improved_experiments
LOG_DIR="logs/improved_experiments"

################################################################################
# 阶段1: 检查ResNet18训练状态
################################################################################
echo -e "${YELLOW}[阶段1] 检查ResNet18训练状态...${NC}"

if pgrep -f "train_models.py.*resnet18" > /dev/null; then
    echo -e "${GREEN}  ✓ ResNet18正在训练中${NC}"
    echo -e "  等待训练完成（按Ctrl+C跳过等待）..."
    
    # 等待训练完成（最多等待4小时）
    timeout 14400 bash -c '
        while pgrep -f "train_models.py.*resnet18" > /dev/null; do
            sleep 60
            echo "  仍在训练中..."
        done
    ' || echo -e "${YELLOW}  等待超时或被中断${NC}"
else
    echo -e "${YELLOW}  ⚠ ResNet18未在训练，检查是否已完成...${NC}"
    
    if [ -f "checkpoints/resnet18/best_model.pth" ]; then
        echo -e "${GREEN}  ✓ 找到ResNet18模型${NC}"
    else
        echo -e "${RED}  ✗ 未找到ResNet18模型，开始训练...${NC}"
        
        nohup python train_models.py \
            --model resnet18 \
            --data_dir data/caltech-101 \
            --save_dir checkpoints \
            --epochs 100 \
            --batch_size 64 \
            --lr 0.0001 \
            --device cuda \
            > "$LOG_DIR/train_resnet18.log" 2>&1 &
        
        echo -e "${GREEN}  训练已启动（后台进程）${NC}"
        echo -e "  查看日志: tail -f $LOG_DIR/train_resnet18.log"
        echo -e "${YELLOW}  请等待训练完成后再运行后续步骤${NC}"
        exit 1
    fi
fi

################################################################################
# 阶段2: 使用VGG11进行多场景测试（VGG11已训练完成）
################################################################################
echo ""
echo -e "${YELLOW}[阶段2] VGG11多场景测试...${NC}"

if [ -f "results/multi_scenario/vgg11_summary.json" ]; then
    echo -e "${GREEN}  ✓ VGG11多场景测试已完成${NC}"
else
    echo -e "${BLUE}  开始VGG11多场景测试（500样本）...${NC}"
    
    python experiments/multi_scenario_test.py \
        --model vgg11 \
        --data_dir data/caltech-101 \
        --checkpoint_dir checkpoints \
        --rl_agent_dir rl_agents \
        --save_dir results/multi_scenario \
        --num_samples 500 \
        --device cuda \
        2>&1 | tee "$LOG_DIR/vgg11_multi_scenario.log"
    
    echo -e "${GREEN}  ✓ VGG11多场景测试完成${NC}"
fi

################################################################################
# 阶段3: ResNet18改进的RL训练（3000 episodes）
################################################################################
echo ""
echo -e "${YELLOW}[阶段3] ResNet18改进的RL训练...${NC}"

if [ -f "checkpoints/resnet18/best_model.pth" ]; then
    echo -e "${BLUE}  开始ResNet18的RL训练（3000 episodes，多场景）...${NC}"
    
    # 弱网场景（低带宽、高延迟）
    python train_rl_agent.py \
        --model resnet18 \
        --data_dir data/caltech-101 \
        --checkpoint_dir checkpoints \
        --save_dir rl_agents_improved \
        --episodes 3000 \
        --max_steps 200 \
        --update_interval 10 \
        --device cuda \
        2>&1 | tee "$LOG_DIR/train_rl_resnet18.log"
    
    echo -e "${GREEN}  ✓ ResNet18 RL训练完成${NC}"
else
    echo -e "${RED}  ✗ ResNet18模型未就绪，跳过${NC}"
fi

################################################################################
# 阶段4: ResNet18多场景测试
################################################################################
echo ""
echo -e "${YELLOW}[阶段4] ResNet18多场景测试...${NC}"

if [ -f "rl_agents_improved/rl_agent_resnet18/best_agent.pth" ]; then
    echo -e "${BLUE}  开始ResNet18多场景测试...${NC}"
    
    python experiments/multi_scenario_test.py \
        --model resnet18 \
        --data_dir data/caltech-101 \
        --checkpoint_dir checkpoints \
        --rl_agent_dir rl_agents_improved \
        --save_dir results/multi_scenario \
        --num_samples 500 \
        --device cuda \
        2>&1 | tee "$LOG_DIR/resnet18_multi_scenario.log"
    
    echo -e "${GREEN}  ✓ ResNet18多场景测试完成${NC}"
else
    echo -e "${RED}  ✗ ResNet18 RL Agent未就绪，跳过${NC}"
fi

################################################################################
# 阶段5: 生成详细对比图表
################################################################################
echo ""
echo -e "${YELLOW}[阶段5] 生成详细对比图表...${NC}"

# VGG11图表
if [ -f "results/multi_scenario/vgg11_summary.json" ]; then
    echo -e "${BLUE}  生成VGG11图表...${NC}"
    
    python experiments/visualize_multi_scenario.py \
        --results_dir results/multi_scenario \
        --model vgg11 \
        --save_dir figures/multi_scenario \
        2>&1 | tee "$LOG_DIR/visualize_vgg11.log"
    
    echo -e "${GREEN}  ✓ VGG11图表生成完成${NC}"
fi

# ResNet18图表
if [ -f "results/multi_scenario/resnet18_summary.json" ]; then
    echo -e "${BLUE}  生成ResNet18图表...${NC}"
    
    python experiments/visualize_multi_scenario.py \
        --results_dir results/multi_scenario \
        --model resnet18 \
        --save_dir figures/multi_scenario \
        2>&1 | tee "$LOG_DIR/visualize_resnet18.log"
    
    echo -e "${GREEN}  ✓ ResNet18图表生成完成${NC}"
fi

################################################################################
# 完成总结
################################################################################
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    改进实验完成！${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}生成的结果文件：${NC}"
echo "  • 多场景数据: results/multi_scenario/"
echo "  • 对比图表: figures/multi_scenario/"
echo "  • 训练日志: $LOG_DIR/"
echo ""
echo -e "${YELLOW}后续步骤：${NC}"
echo "  1. 查看多场景对比图表: ls -lh figures/multi_scenario/"
echo "  2. 分析结果数据: cat results/multi_scenario/*_summary.json"
echo "  3. 查看汇总表格: cat figures/multi_scenario/*_summary.csv"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"


#!/bin/bash

################################################################################
#                    实验进度监控脚本
################################################################################

PROJECT_DIR="/opt/03-ai/01-proj/Neurosurgeon"
cd "$PROJECT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear

echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    实验进度监控面板${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

################################################################################
# 1. ResNet18训练进度
################################################################################
echo -e "${YELLOW}[1] ResNet18训练状态${NC}"
echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

if pgrep -f "train_models.py.*resnet18" > /dev/null; then
    echo -e "${GREEN}  ✓ 正在训练中...${NC}"
    
    if [ -f "logs/retrain_resnet18.log" ]; then
        # 获取最新的epoch信息
        LATEST_EPOCH=$(grep -E "Epoch [0-9]+/100" logs/retrain_resnet18.log | tail -1)
        LATEST_ACC=$(grep "测试准确率" logs/retrain_resnet18.log | tail -1)
        
        echo -e "  ${LATEST_EPOCH}"
        echo -e "  ${LATEST_ACC}"
        
        # 计算进度百分比
        CURRENT_EPOCH=$(echo "$LATEST_EPOCH" | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+")
        if [ ! -z "$CURRENT_EPOCH" ]; then
            PROGRESS=$((CURRENT_EPOCH * 100 / 100))
            echo -e "  进度: ${PROGRESS}%"
        fi
    fi
else
    if [ -f "checkpoints/resnet18/best_model.pth" ]; then
        echo -e "${GREEN}  ✓ 训练已完成${NC}"
        FINAL_ACC=$(grep "测试准确率" logs/retrain_resnet18.log | tail -1 | grep -oE "[0-9]+\.[0-9]+%")
        echo -e "  最终准确率: ${FINAL_ACC}"
    else
        echo -e "${RED}  ✗ 未在训练中${NC}"
    fi
fi
echo ""

################################################################################
# 2. VGG11多场景测试进度
################################################################################
echo -e "${YELLOW}[2] VGG11多场景测试状态${NC}"
echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

if pgrep -f "multi_scenario_test.py.*vgg11" > /dev/null; then
    echo -e "${GREEN}  ✓ 正在测试中...${NC}"
    
    if [ -f "logs/improved_experiments/vgg11_multi_scenario.log" ]; then
        # 获取当前场景和进度
        CURRENT_SCENARIO=$(grep -E "测试场景:" logs/improved_experiments/vgg11_multi_scenario.log | tail -1)
        echo -e "  ${CURRENT_SCENARIO}"
        
        # 获取进度百分比
        PROGRESS_LINE=$(tail -50 logs/improved_experiments/vgg11_multi_scenario.log | grep "%" | tail -1)
        if [ ! -z "$PROGRESS_LINE" ]; then
            PROGRESS_PCT=$(echo "$PROGRESS_LINE" | grep -oE "[0-9]+%" | head -1)
            PROGRESS_COUNT=$(echo "$PROGRESS_LINE" | grep -oE "[0-9]+/[0-9]+" | head -1)
            echo -e "  进度: ${PROGRESS_PCT} (${PROGRESS_COUNT})"
        fi
    fi
else
    if [ -f "results/multi_scenario/vgg11_summary.json" ]; then
        echo -e "${GREEN}  ✓ 测试已完成${NC}"
        
        # 统计场景数量
        SCENARIO_COUNT=$(cat results/multi_scenario/vgg11_summary.json | grep -c "config")
        echo -e "  已完成 ${SCENARIO_COUNT} 个场景的测试"
    else
        echo -e "${RED}  ✗ 未在测试中${NC}"
    fi
fi
echo ""

################################################################################
# 3. ResNet18 RL训练状态
################################################################################
echo -e "${YELLOW}[3] ResNet18 RL训练状态${NC}"
echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

if pgrep -f "train_rl_agent.py.*resnet18" > /dev/null; then
    echo -e "${GREEN}  ✓ 正在训练中...${NC}"
    
    if [ -f "logs/improved_experiments/train_rl_resnet18.log" ]; then
        EPISODE_INFO=$(grep -E "Episode [0-9]+/3000" logs/improved_experiments/train_rl_resnet18.log | tail -1)
        echo -e "  ${EPISODE_INFO}"
    fi
else
    if [ -f "rl_agents_improved/rl_agent_resnet18/best_agent.pth" ]; then
        echo -e "${GREEN}  ✓ RL训练已完成${NC}"
    else
        echo -e "${YELLOW}  ⏳ 等待ResNet18模型训练完成${NC}"
    fi
fi
echo ""

################################################################################
# 4. ResNet18多场景测试状态
################################################################################
echo -e "${YELLOW}[4] ResNet18多场景测试状态${NC}"
echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

if pgrep -f "multi_scenario_test.py.*resnet18" > /dev/null; then
    echo -e "${GREEN}  ✓ 正在测试中...${NC}"
else
    if [ -f "results/multi_scenario/resnet18_summary.json" ]; then
        echo -e "${GREEN}  ✓ 测试已完成${NC}"
    else
        echo -e "${YELLOW}  ⏳ 等待RL训练完成${NC}"
    fi
fi
echo ""

################################################################################
# 5. 生成的结果文件
################################################################################
echo -e "${YELLOW}[5] 已生成的结果文件${NC}"
echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

# 检查结果目录
if [ -d "results/multi_scenario" ]; then
    FILE_COUNT=$(ls results/multi_scenario/*.json 2>/dev/null | wc -l)
    echo -e "  多场景结果: ${FILE_COUNT} 个JSON文件"
    ls -lh results/multi_scenario/*.json 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
fi
echo ""

if [ -d "figures/multi_scenario" ]; then
    FIG_COUNT=$(ls figures/multi_scenario/*.png 2>/dev/null | wc -l)
    echo -e "  可视化图表: ${FIG_COUNT} 个PNG文件"
    ls -lh figures/multi_scenario/*.png 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
fi
echo ""

################################################################################
# 6. 快速操作命令
################################################################################
echo -e "${YELLOW}[6] 快速操作命令${NC}"
echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "  查看ResNet18训练:    ${GREEN}tail -f logs/retrain_resnet18.log${NC}"
echo -e "  查看VGG11多场景:     ${GREEN}tail -f logs/improved_experiments/vgg11_multi_scenario.log${NC}"
echo -e "  查看VGG11结果:       ${GREEN}cat results/multi_scenario/vgg11_summary.json | jq .${NC}"
echo -e "  生成VGG11图表:       ${GREEN}python experiments/visualize_multi_scenario.py --model vgg11${NC}"
echo -e "  运行完整流程:        ${GREEN}./run_improved_experiments.sh${NC}"
echo ""

################################################################################
# 7. 系统资源使用
################################################################################
echo -e "${YELLOW}[7] 系统资源使用${NC}"
echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

# GPU使用
if command -v nvidia-smi &> /dev/null; then
    echo -e "  GPU状态:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F, '{printf "    GPU%s: %s | 使用率: %s%% | 显存: %sMB/%sMB\n", $1, $2, $3, $4, $5}'
fi

# CPU和内存
echo -e "\n  CPU & 内存:"
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "    CPU使用率: " 100 - $1 "%"}'
free -h | grep "Mem:" | awk '{print "    内存使用: " $3 " / " $2}'

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "  刷新时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "  重新运行此脚本: ${GREEN}./check_progress.sh${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"


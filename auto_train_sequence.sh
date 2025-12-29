#!/bin/bash

################################################################################
#                    自动化顺序训练脚本
#                    AlexNet完成后自动启动MobileNetV2
################################################################################

set -e

PROJECT_DIR="/opt/03-ai/01-proj/Neurosurgeon"
cd "$PROJECT_DIR"

# 激活虚拟环境
source neurosurgeon_env/bin/activate

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    自动化训练序列监控${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

################################################################################
# 等待AlexNet训练完成
################################################################################
echo -e "${YELLOW}等待AlexNet训练完成...${NC}"
echo ""

while true; do
    # 检查AlexNet训练进程是否还在运行
    if ! pgrep -f "train_models.py.*alexnet" > /dev/null; then
        # 进程已结束，检查是否成功完成
        if [ -f "checkpoints/alexnet/best_model.pth" ]; then
            echo -e "${GREEN}✅ AlexNet训练已完成！${NC}"
            
            # 获取最终准确率
            if [ -f "checkpoints/alexnet/training_history.json" ]; then
                ALEXNET_ACC=$(python3 -c "import json; data=json.load(open('checkpoints/alexnet/training_history.json')); print(f\"{data['test_acc'][-1]:.2f}%\")" 2>/dev/null || echo "N/A")
                echo -e "  最终准确率: ${ALEXNET_ACC}"
            fi
            
            break
        else
            echo -e "${YELLOW}⚠ AlexNet训练进程已结束，但未找到模型文件${NC}"
            echo -e "  请检查日志: logs/retrain_alexnet.log"
            exit 1
        fi
    else
        # 显示当前进度
        CURRENT_EPOCH=$(grep -E "Epoch [0-9]+/100" logs/retrain_alexnet.log 2>/dev/null | tail -1 | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+")
        if [ ! -z "$CURRENT_EPOCH" ]; then
            echo -ne "\r  AlexNet训练进度: Epoch ${CURRENT_EPOCH}/100  "
        fi
        
        sleep 30  # 每30秒检查一次
    fi
done

echo ""

################################################################################
# 启动MobileNetV2训练
################################################################################
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}启动MobileNetV2训练（GPU 1）...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

CUDA_VISIBLE_DEVICES=1 python train_models.py \
    --model mobilenetv2 \
    --data_dir data/caltech-101 \
    --save_dir checkpoints \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --device cuda \
    2>&1 | tee logs/retrain_mobilenetv2.log

################################################################################
# 训练完成
################################################################################
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}                    所有模型训练完成！${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# 显示所有模型的最终准确率
echo -e "${YELLOW}所有模型最终准确率:${NC}"
echo ""

for model in resnet18 vgg11 alexnet mobilenetv2; do
    if [ -f "checkpoints/${model}/training_history.json" ]; then
        ACC=$(python3 -c "import json; data=json.load(open('checkpoints/${model}/training_history.json')); print(f\"{data['test_acc'][-1]:.2f}%\")" 2>/dev/null || echo "N/A")
        echo -e "  ${model}: ${ACC}"
    else
        echo -e "  ${model}: 未找到历史记录"
    fi
done

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════════${NC}"


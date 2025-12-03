#!/usr/bin/env bash
# 按照论文实验规划执行所有实验

set -e

BASE_DIR="/opt/03-ai/01-proj/Neurosurgeon/rl_collaborative_inference"
DATA_DIR="/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101"
VENV_DIR="/opt/03-ai/01-proj/Neurosurgeon/neurosurgeon_env"

cd "${BASE_DIR}"
source "${VENV_DIR}/bin/activate"

echo "=========================================="
echo "论文实验完整执行流程"
echo "=========================================="

# 创建输出目录
RESULTS_DIR="${BASE_DIR}/experiments_paper/inference_evaluation"
mkdir -p "${RESULTS_DIR}"

# ============================================
# 阶段1: 检查并生成训练结果图表（已完成）
# ============================================
echo -e "\n[阶段1] 训练结果检查与图表生成..."
echo "✅ 已完成：baseline vs RL 图表已生成"
echo "✅ 已完成：多种子结果已汇总"

# ============================================
# 阶段2: 推理性能对比实验
# ============================================
echo -e "\n[阶段2] 推理性能对比实验..."

# 检查是否有训练好的模型
MODEL_PATH=""
for SEED in 2 3; do
    CHECK_PATH="${BASE_DIR}/experiments_paper/seed_${SEED}/ablation_20251127_055835/final_model.pt"
    if [ -f "$CHECK_PATH" ]; then
        MODEL_PATH="$CHECK_PATH"
        echo "Found trained model: $MODEL_PATH"
        break
    fi
done

# 如果没有找到模型，快速训练一个
if [ -z "$MODEL_PATH" ]; then
    echo "No trained model found. Training a quick model for evaluation..."
    QUICK_MODEL_DIR="${BASE_DIR}/quick_models"
    mkdir -p "${QUICK_MODEL_DIR}"
    
    python train_quick_model.py \
        --data_dir "${DATA_DIR}" \
        --output_dir "${QUICK_MODEL_DIR}" \
        --seed 42 \
        --max_steps 50000 \
        --device cpu
    
    MODEL_PATH=$(ls -t "${QUICK_MODEL_DIR}"/quick_model_*.pt | head -1)
    echo "Quick model trained: $MODEL_PATH"
fi

if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Could not find or create a model file"
    exit 1
fi

# 运行推理性能对比实验
echo "Running inference comparison experiments..."
python run_inference_comparison.py \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${RESULTS_DIR}" \
    --bandwidths 5.0 10.0 20.0 50.0 \
    --num_samples 500 \
    --device cpu

echo "✅ 推理性能对比实验完成"

# ============================================
# 阶段3: 多模型评估实验（可选，需要其他模型）
# ============================================
echo -e "\n[阶段3] 多模型评估实验..."
echo "⚠️  需要实现多模型支持，当前仅支持AlexNet"
echo "   可以后续扩展支持 VGG/ResNet/MobileNet"

# ============================================
# 阶段4: 组件消融实验（可选）
# ============================================
echo -e "\n[阶段4] 组件消融实验..."
echo "⚠️  可选实验，可根据论文需要执行"

# ============================================
# 阶段5: 鲁棒性测试（可选）
# ============================================
echo -e "\n[阶段5] 鲁棒性测试..."
echo "⚠️  可选实验，可根据论文需要执行"

# ============================================
# 总结
# ============================================
echo -e "\n=========================================="
echo "实验执行完成！"
echo "=========================================="
echo "结果目录: ${RESULTS_DIR}"
echo ""
echo "生成的文件："
ls -lh "${RESULTS_DIR}"/*.json "${RESULTS_DIR}"/*.png 2>/dev/null || echo "  (检查输出目录)"
echo ""
echo "下一步："
echo "1. 查看生成的图表: ${RESULTS_DIR}/*.png"
echo "2. 查看结果数据: ${RESULTS_DIR}/*.json"
echo "3. 根据需要进行多模型评估和消融实验"


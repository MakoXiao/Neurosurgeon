#!/usr/bin/env bash
# 完整的论文实验执行脚本（按顺序执行所有步骤）

set -e

BASE_DIR="/opt/03-ai/01-proj/Neurosurgeon/rl_collaborative_inference"
DATA_DIR="/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101"
VENV_DIR="/opt/03-ai/01-proj/Neurosurgeon/neurosurgeon_env"

cd "${BASE_DIR}"
source "${VENV_DIR}/bin/activate"

echo "=========================================="
echo "完整论文实验执行流程"
echo "=========================================="

# 步骤1: 检查训练结果完整性
echo -e "\n[步骤1] 检查训练结果完整性..."
SEEDS=(1 2 3)
MISSING_EXPERIMENTS=()

for SEED in "${SEEDS[@]}"; do
  EXP_DIR="${BASE_DIR}/experiments_paper/seed_${SEED}/ablation_20251127_055835"
  if [ -d "$EXP_DIR" ]; then
    echo "Checking seed ${SEED}..."
    
    if [ ! -f "${EXP_DIR}/ablation_hyperparams.json" ]; then
      MISSING_EXPERIMENTS+=("${SEED}:ablation")
      echo "  - Missing: ablation_hyperparams.json"
    fi
    
    if [ ! -f "${EXP_DIR}/multi_user_scaling.json" ]; then
      MISSING_EXPERIMENTS+=("${SEED}:multi_user")
      echo "  - Missing: multi_user_scaling.json"
    fi
  else
    echo "  - Warning: Experiment directory not found for seed ${SEED}"
  fi
done

# 步骤2: 补充缺失的实验
if [ ${#MISSING_EXPERIMENTS[@]} -gt 0 ]; then
  echo -e "\n[步骤2] 补充缺失的实验..."
  for item in "${MISSING_EXPERIMENTS[@]}"; do
    SEED=$(echo $item | cut -d: -f1)
    EXP_TYPE=$(echo $item | cut -d: -f2)
    EXP_DIR="${BASE_DIR}/experiments_paper/seed_${SEED}/ablation_20251127_055835"
    
    if [ "$EXP_TYPE" == "ablation" ]; then
      echo "Running ablation for seed ${SEED}..."
      python run_missing_experiments.py \
        --output_dir "${EXP_DIR}" \
        --seed ${SEED} \
        --data_dir "${DATA_DIR}" \
        --run_ablation \
        --max_steps_rl 500000
    elif [ "$EXP_TYPE" == "multi_user" ]; then
      echo "Running multi-user for seed ${SEED}..."
      python run_missing_experiments.py \
        --output_dir "${EXP_DIR}" \
        --seed ${SEED} \
        --data_dir "${DATA_DIR}" \
        --run_multi_user \
        --max_steps_multi_user 500000
    fi
  done
else
  echo -e "\n[步骤2] 所有实验已完成，跳过补充步骤"
fi

# 步骤3: 为每个seed生成论文图
echo -e "\n[步骤3] 为每个seed生成论文图..."
for SEED in "${SEEDS[@]}"; do
  EXP_DIR="${BASE_DIR}/experiments_paper/seed_${SEED}/ablation_20251127_055835"
  if [ -d "$EXP_DIR" ]; then
    echo "Generating figures for seed ${SEED}..."
    python plot_paper_style.py --exp_dir "${EXP_DIR}"
  fi
done

# 步骤4: 汇总多种子结果
echo -e "\n[步骤4] 汇总多种子结果..."
AGGREGATE_DIR="${BASE_DIR}/experiments_paper/aggregated"
mkdir -p "${AGGREGATE_DIR}"

SEED_DIRS=()
for SEED in "${SEEDS[@]}"; do
  SEED_DIR="${BASE_DIR}/experiments_paper/seed_${SEED}"
  if [ -d "$SEED_DIR" ]; then
    SEED_DIRS+=("${SEED_DIR}")
  fi
done

if [ ${#SEED_DIRS[@]} -gt 0 ]; then
  python aggregate_multi_seed_results.py \
    --seed_dirs "${SEED_DIRS[@]}" \
    --output_dir "${AGGREGATE_DIR}"
else
  echo "Warning: No seed directories found for aggregation"
fi

# 步骤5: 推理性能评估（需要训练好的模型）
echo -e "\n[步骤5] 推理性能评估..."
# 注意：这里需要指定训练好的模型路径
# 假设模型保存在训练输出目录中
MODEL_PATHS=(
  "${BASE_DIR}/experiments_paper/seed_1/ablation_20251127_055835/final_model.pt"
  "${BASE_DIR}/experiments_paper/seed_2/ablation_20251127_055835/final_model.pt"
  "${BASE_DIR}/experiments_paper/seed_3/ablation_20251127_055835/final_model.pt"
)

EVAL_OUTPUT_DIR="${BASE_DIR}/experiments_paper/inference_evaluation"
mkdir -p "${EVAL_OUTPUT_DIR}"

# 在不同带宽下评估
BANDWIDTHS=(5.0 10.0 20.0 50.0)
for BANDWIDTH in "${BANDWIDTHS[@]}"; do
  echo "Evaluating at ${BANDWIDTH} MB/s..."
  
  # 使用第一个可用的模型
  MODEL_PATH=""
  for path in "${MODEL_PATHS[@]}"; do
    if [ -f "$path" ]; then
      MODEL_PATH="$path"
      break
    fi
  done
  
  if [ -n "$MODEL_PATH" ]; then
    OUTPUT_FILE="${EVAL_OUTPUT_DIR}/evaluation_${BANDWIDTH}mbps.json"
    python evaluate_trained_policy.py \
      --model_path "${MODEL_PATH}" \
      --data_dir "${DATA_DIR}" \
      --output_file "${OUTPUT_FILE}" \
      --network_bandwidth ${BANDWIDTH} \
      --num_samples 500
  else
    echo "Warning: No trained model found, skipping inference evaluation"
    break
  fi
done

echo -e "\n=========================================="
echo "所有实验步骤完成！"
echo "=========================================="
echo "汇总结果目录: ${AGGREGATE_DIR}"
echo "推理评估结果: ${EVAL_OUTPUT_DIR}"


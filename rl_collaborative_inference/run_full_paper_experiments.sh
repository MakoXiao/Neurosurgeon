#!/usr/bin/env bash
# 一键运行论文所需的 500k 步正式实验（多随机种子、后台执行）

set -e

########################################
# 0. 基本路径与环境设置
########################################

BASE_DIR="/opt/03-ai/01-proj/Neurosurgeon/rl_collaborative_inference"
DATA_DIR="/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101"

# neurosurgeon_env 虚拟环境路径（如有变化可在这里改）
VENV_DIR="/opt/03-ai/01-proj/Neurosurgeon/neurosurgeon_env"

cd "${BASE_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "虚拟环境目录不存在: ${VENV_DIR}"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

mkdir -p logs
mkdir -p experiments_paper

########################################
# 1. 统一实验参数
########################################

MAX_STEPS_BASELINE=500000
MAX_STEPS_RL=500000
MAX_STEPS_MULTI=500000

# 多随机种子
SEEDS=(1 2 3)

########################################
# 2. 逐种子后台运行完整实验
########################################

echo "启动 500k 步正式实验（多随机种子），进程信息记录在 logs/pids.txt"

: > logs/pids.txt

for SEED in "${SEEDS[@]}"; do
  OUT_ROOT="${BASE_DIR}/experiments_paper/seed_${SEED}"
  mkdir -p "${OUT_ROOT}"

  LOG_FILE="logs/ablation_seed_${SEED}.log"

  echo "启动 seed=${SEED} 的实验，日志：${LOG_FILE}"

  # 如有多块 GPU，可在下面一行前加 CUDA_VISIBLE_DEVICES=0/1/... 限定显卡
  nohup python run_ablation_experiments.py \
    --data_dir "${DATA_DIR}" \
    --output_root "${OUT_ROOT}" \
    --max_steps_baseline ${MAX_STEPS_BASELINE} \
    --max_steps_rl ${MAX_STEPS_RL} \
    --max_steps_multi_user ${MAX_STEPS_MULTI} \
    --seed ${SEED} \
    > "${LOG_FILE}" 2>&1 &

  PID=$!
  echo "seed=${SEED} PID=${PID}" | tee -a logs/pids.txt
done

echo "所有实验已在后台启动。"

cat <<EOF
================= 使用说明 =================
1) 检查进程是否仍在运行：
   ps aux | grep run_ablation_experiments.py | grep -v grep

2) 查看某个种子的实时日志（例如 seed=1）：
   tail -f logs/ablation_seed_1.log

3) 所有后台实验结束后，为每个 seed 生成论文图：
   for SEED in ${SEEDS[@]}; do
     EXP_DIR="${BASE_DIR}/experiments_paper/seed_\${SEED}"/ablation_*
     python plot_paper_style.py --exp_dir "\${EXP_DIR}"
   done
============================================
EOF



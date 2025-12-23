#!/bin/bash
# 执行后续实验的脚本

cd /opt/03-ai/01-proj/Neurosurgeon/rl_collaborative_inference
source ../neurosurgeon_env/bin/activate

echo "=========================================="
echo "执行后续实验"
echo "=========================================="
echo ""

# 选择要执行的任务
TASK=${1:-all}

if [ "$TASK" == "all" ] || [ "$TASK" == "hyperparameter" ]; then
    echo "1. 启动超参数敏感性实验..."
    python experiments/run_hyperparameter_background.py \
        --data_dir ../data/caltech-101 \
        --output_dir ./experiments/hyperparameter_sensitivity \
        --log_dir ./logs \
        --experiment all \
        --use_cuda
    echo "超参数实验已启动（后台运行）"
    echo ""
fi

if [ "$TASK" == "all" ] || [ "$TASK" == "comparison" ]; then
    echo "2. 运行对比实验（Local, JALAD, Proposed）..."
    python experiments/comparison_experiment.py \
        --data_dir ../data/caltech-101 \
        --output_dir ./experiments/comparison \
        --max_steps 500000 \
        --use_cuda
    echo "对比实验完成"
    echo ""
fi

if [ "$TASK" == "all" ] || [ "$TASK" == "figures" ]; then
    echo "3. 生成论文图表..."
    python experiments/generate_paper_figures.py \
        --results_dir ./experiments \
        --output_dir ./experiments/paper_figures \
        --figure all
    echo "论文图表生成完成"
    echo ""
fi

echo "=========================================="
echo "所有任务完成"
echo "=========================================="
echo ""
echo "查看状态: python run_training_background.py status"
echo "查看日志: tail -f ./logs/<job_name>.log"


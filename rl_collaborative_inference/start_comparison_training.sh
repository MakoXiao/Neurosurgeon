#!/bin/bash
# 启动正式训练脚本

cd /opt/03-ai/01-proj/Neurosurgeon/rl_collaborative_inference
source ../neurosurgeon_env/bin/activate

echo "=========================================="
echo "启动正式训练任务: comparison_experiment"
echo "=========================================="
echo ""

python run_training_background.py start \
    --script train_with_tracking.py \
    --job_name comparison_experiment \
    --data_dir ../data/caltech-101 \
    --output_dir ./experiments/comparison \
    --max_steps 500000 \
    --lr_actor 0.0001 \
    --lr_critic 0.0001 \
    --k_epochs 10 \
    --batch_size 64 \
    --network_bandwidth 10.0 \
    --seed 42 \
    --use_cuda \
    --log_dir ./logs

echo ""
echo "=========================================="
echo "训练启动命令已执行"
echo "=========================================="
echo ""
echo "查看状态: python run_training_background.py status"
echo "查看日志: tail -f ./logs/comparison_experiment.log"
echo ""


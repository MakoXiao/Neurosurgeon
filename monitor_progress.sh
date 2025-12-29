#!/bin/bash

# 实验进度监控脚本

PROJECT_ROOT="/opt/03-ai/01-proj/Neurosurgeon"
cd $PROJECT_ROOT

echo "════════════════════════════════════════════════════════════════"
echo "           Neurosurgeon 实验进度监控"
echo "════════════════════════════════════════════════════════════════"
echo ""

# 检查实验是否在运行
if pgrep -f "start_experiments.sh" > /dev/null; then
    echo "✅ 实验正在运行中..."
    echo "   进程ID: $(pgrep -f start_experiments.sh)"
else
    echo "⚠️  实验未运行或已完成"
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "GPU 使用情况"
echo "────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
awk -F, '{printf "GPU %s (%s): 使用率=%s%%, 显存=%sMB/%sMB\n", $1, $2, $3, $4, $5}'

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "训练进度"
echo "────────────────────────────────────────────────────────────────"

# 检查模型训练进度
if [ -d "checkpoints" ]; then
    echo "✅ 模型训练:"
    for model in resnet18 vgg11 mobilenetv2 alexnet; do
        if [ -f "checkpoints/$model/best_model.pth" ]; then
            echo "   ✓ $model - 已完成"
        elif [ -d "checkpoints/$model" ]; then
            echo "   ⏳ $model - 训练中..."
        else
            echo "   ○ $model - 等待中"
        fi
    done
else
    echo "⏳ 模型训练: 准备中..."
fi

echo ""

# 检查RL训练进度
if [ -d "rl_agents" ]; then
    echo "✅ RL智能体训练:"
    for model in resnet18 vgg11 mobilenetv2 alexnet; do
        if [ -f "rl_agents/rl_agent_$model/best_agent.pth" ]; then
            echo "   ✓ rl_agent_$model - 已完成"
        elif [ -d "rl_agents/rl_agent_$model" ]; then
            echo "   ⏳ rl_agent_$model - 训练中..."
        else
            echo "   ○ rl_agent_$model - 等待中"
        fi
    done
else
    echo "⏳ RL智能体训练: 准备中..."
fi

echo ""

# 检查实验结果
if [ -d "results" ] && [ "$(ls -A results 2>/dev/null)" ]; then
    echo "✅ 对比实验:"
    result_count=$(ls -1 results/*.json 2>/dev/null | wc -l)
    echo "   已完成 $result_count 个模型的对比实验"
else
    echo "⏳ 对比实验: 准备中..."
fi

echo ""

# 检查可视化结果
if [ -d "figures" ] && [ "$(ls -A figures 2>/dev/null)" ]; then
    echo "✅ 可视化图表:"
    figure_count=$(ls -1 figures/*.png 2>/dev/null | wc -l)
    echo "   已生成 $figure_count 个图表"
else
    echo "⏳ 可视化图表: 准备中..."
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "最新日志 (最后10行)"
echo "────────────────────────────────────────────────────────────────"

if [ -f "logs/full_experiment.log" ]; then
    tail -10 logs/full_experiment.log
else
    echo "日志文件尚未生成"
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "查看详细日志:"
echo "  tail -f logs/full_experiment.log    # 总体进度"
echo "  tail -f logs/train_models.log       # 模型训练"
echo "  tail -f logs/train_rl_agents.log    # RL训练"
echo "  tail -f logs/compare_methods.log    # 对比实验"
echo ""
echo "停止实验 (如果需要):"
echo "  pkill -f start_experiments.sh"
echo "════════════════════════════════════════════════════════════════"



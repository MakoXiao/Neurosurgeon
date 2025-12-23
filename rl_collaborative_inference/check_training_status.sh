#!/bin/bash
# 检查模型训练状态

echo "=== 训练进程状态 ==="
ps aux | grep "train_classification_model" | grep -v grep

echo ""
echo "=== 模型文件状态 ==="
if [ -f "./alexnet_caltech101.pth" ]; then
    ls -lh ./alexnet_caltech101.pth
    echo "模型文件已存在！"
else
    echo "模型文件尚未生成"
fi

echo ""
echo "=== 训练日志（最后20行）==="
if [ -f "/tmp/train_model_fixed.log" ]; then
    tail -20 /tmp/train_model_fixed.log
else
    echo "日志文件不存在"
fi


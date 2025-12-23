#!/usr/bin/env python3
"""
诊断和启动训练脚本
检查环境并启动正式训练
"""
import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("诊断和启动训练脚本")
    print("=" * 60)
    print()
    
    # 检查当前目录
    cwd = os.getcwd()
    print(f"当前目录: {cwd}")
    
    # 检查脚本文件
    script_dir = "/opt/03-ai/01-proj/Neurosurgeon/rl_collaborative_inference"
    os.chdir(script_dir)
    print(f"切换到: {os.getcwd()}")
    
    # 检查文件
    files_to_check = [
        "run_training_background.py",
        "train_with_tracking.py",
        "../neurosurgeon_env/bin/activate"
    ]
    
    print("\n检查必要文件:")
    for f in files_to_check:
        exists = os.path.exists(f)
        print(f"  {f}: {'✓' if exists else '✗'}")
        if not exists:
            print(f"    错误: 文件不存在!")
            return 1
    
    # 检查Python
    print("\n检查Python环境:")
    python_path = sys.executable
    print(f"  Python路径: {python_path}")
    print(f"  Python版本: {sys.version}")
    
    # 尝试导入必要的模块
    print("\n检查Python模块:")
    try:
        import torch
        print(f"  PyTorch: ✓ (版本: {torch.__version__})")
        print(f"  CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"  CUDA设备数: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"  PyTorch: ✗ ({e})")
        return 1
    
    # 启动训练
    print("\n" + "=" * 60)
    print("启动正式训练")
    print("=" * 60)
    print()
    
    cmd = [
        python_path,
        "run_training_background.py",
        "start",
        "--script", "train_with_tracking.py",
        "--job_name", "comparison_experiment",
        "--data_dir", "../data/caltech-101",
        "--output_dir", "./experiments/comparison",
        "--max_steps", "500000",
        "--lr_actor", "0.0001",
        "--lr_critic", "0.0001",
        "--k_epochs", "10",
        "--batch_size", "64",
        "--network_bandwidth", "10.0",
        "--seed", "42",
        "--use_cuda",
        "--log_dir", "./logs"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        print(f"\n命令执行完成，返回码: {result.returncode}")
        
        if result.returncode == 0:
            print("\n✓ 训练已启动!")
            print("\n查看状态:")
            print("  python run_training_background.py status")
            print("\n查看日志:")
            print("  tail -f ./logs/comparison_experiment.log")
        else:
            print("\n✗ 训练启动可能失败，请检查错误信息")
            return 1
            
    except Exception as e:
        print(f"\n✗ 执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


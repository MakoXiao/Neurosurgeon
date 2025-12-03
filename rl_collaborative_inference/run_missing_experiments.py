"""
补充运行缺失的实验部分（ablation 和多用户实验）
可以单独运行，不需要重新跑 baseline_vs_rl
"""
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.actor_critic import Actor, Critic
from src.env import CollaborativeInferenceEnv
from src.multi_user_env import MultiUserEnv
from src.ppo import PPO, PPOBuffer
from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet

# 复用 run_ablation_experiments.py 中的函数
import sys
sys.path.insert(0, os.path.dirname(__file__))
from run_ablation_experiments import (
    set_seed, build_single_env, build_agent, run_rl_training,
    experiment_ablation, experiment_multi_user_scaling
)


def main():
    parser = argparse.ArgumentParser(description="补充运行缺失的实验部分")
    parser.add_argument("--data_dir", type=str, default="../data/caltech-101")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="已有的实验结果目录（包含 baseline_vs_rl.json）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--network_bandwidth", type=float, default=10.0)
    parser.add_argument("--pruning_type", type=str, default="structured")
    parser.add_argument("--target_accuracy", type=float, default=0.95)
    parser.add_argument("--max_latency", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    
    # PPO 超参
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--k_epochs", type=int, default=10)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_freq", type=int, default=10)
    
    # 实验步数
    parser.add_argument("--max_steps_rl", type=int, default=500000)
    parser.add_argument("--max_steps_multi_user", type=int, default=500000)
    
    # 消融范围
    parser.add_argument("--lr_list", type=float, nargs="+",
                       default=[1e-5, 1e-4, 3e-4, 1e-3, 1e-2])
    parser.add_argument("--update_freq_list", type=int, nargs="+",
                       default=[5, 10, 20, 40, 80])
    parser.add_argument("--buffer_size_list", type=int, nargs="+",
                       default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--num_users_list", type=int, nargs="+",
                       default=[3, 4, 5, 6, 7, 8, 9, 10])
    
    # 选择要运行的实验
    parser.add_argument("--run_ablation", action="store_true",
                       help="运行超参数消融实验")
    parser.add_argument("--run_multi_user", action="store_true",
                       help="运行多用户扩展性实验")
    parser.add_argument("--run_all", action="store_true",
                       help="运行所有缺失的实验")
    
    args = parser.parse_args()
    # 统一使用CPU以避免设备不一致问题，如需GPU可后续统一修改
    args.device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查哪些实验已经存在
    ablation_file = os.path.join(output_dir, "ablation_hyperparams.json")
    multi_user_file = os.path.join(output_dir, "multi_user_scaling.json")
    
    run_ablation = args.run_ablation or args.run_all
    run_multi_user = args.run_multi_user or args.run_all
    
    if os.path.exists(ablation_file) and run_ablation:
        print(f"Warning: {ablation_file} already exists. Skipping ablation experiment.")
        run_ablation = False
    
    if os.path.exists(multi_user_file) and run_multi_user:
        print(f"Warning: {multi_user_file} already exists. Skipping multi-user experiment.")
        run_multi_user = False
    
    if not run_ablation and not run_multi_user:
        print("No experiments to run. All required files already exist or no experiments selected.")
        return
    
    # 运行缺失的实验
    if run_ablation:
        print("Running ablation experiments...")
        experiment_ablation(args, output_dir)
    
    if run_multi_user:
        print("Running multi-user scaling experiments...")
        experiment_multi_user_scaling(args, output_dir)
    
    print(f"Missing experiments completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()


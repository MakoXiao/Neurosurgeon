"""
快速训练一个模型用于推理评估
使用较少的步数快速获得一个可用的模型
"""
import argparse
import os
import torch
import json
from datetime import datetime

from src.actor_critic import Actor, Critic
from src.env import CollaborativeInferenceEnv
from src.ppo import PPO
from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet
from run_ablation_experiments import set_seed, build_agent, run_rl_training


def main():
    parser = argparse.ArgumentParser(description="快速训练模型用于推理评估")
    parser.add_argument("--data_dir", type=str, default="../data/caltech-101")
    parser.add_argument("--output_dir", type=str, default="./quick_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=50000,
                       help="训练步数（快速训练使用较少步数）")
    parser.add_argument("--network_bandwidth", type=float, default=10.0)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置其他默认参数
    args.pruning_type = "structured"
    args.target_accuracy = 0.95
    args.max_latency = 1.0
    args.alpha = 0.6
    args.beta = 0.4
    args.lr_actor = 3e-4
    args.lr_critic = 3e-4
    args.gamma = 0.99
    args.eps_clip = 0.2
    args.k_epochs = 10
    args.entropy_coef = 0.01
    args.batch_size = 64
    args.update_freq = 10
    
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Quick training model for inference evaluation")
    print(f"Steps: {args.max_steps}, Device: {args.device}")
    
    # 加载数据集
    dataloader, dataset = get_caltech101_dataloader(
        args.data_dir, batch_size=1, split="train", num_workers=0
    )
    model = AlexNet(input_channels=3, num_classes=101)
    model.eval()
    
    # 创建环境
    env = CollaborativeInferenceEnv(
        model=model,
        dataset=dataset,
        edge_device=args.device,
        cloud_device=args.device,
        network_bandwidth=args.network_bandwidth,
        pruning_type=args.pruning_type,
        target_accuracy=args.target_accuracy,
        max_latency=args.max_latency,
        alpha=args.alpha,
        beta=args.beta,
    )
    
    # 创建agent并训练
    agent = build_agent(env, args)
    rl_result = run_rl_training(env, agent, args, args.max_steps)
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"quick_model_{timestamp}.pt")
    if "agent" in rl_result:
        rl_result["agent"].save(model_path)
    else:
        agent.save(model_path)
    
    print(f"\nQuick training completed!")
    print(f"Model saved to: {model_path}")
    print(f"You can use this model for inference evaluation:")
    print(f"  python run_inference_comparison.py --model_path {model_path} --output_dir ./inference_results")


if __name__ == "__main__":
    main()


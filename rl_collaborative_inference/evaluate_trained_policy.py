"""
使用训练好的RL策略进行推理性能评估，与Local/JALAD/Neurosurgeon对比
"""
import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

from src.actor_critic import Actor
from src.env import CollaborativeInferenceEnv
from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet


def load_trained_actor(model_path: str, state_dim: int, num_partition_points: int, device: str) -> Actor:
    """加载训练好的actor网络"""
    actor = Actor(state_dim, num_partition_points).to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'actor' in checkpoint:
            actor.load_state_dict(checkpoint['actor'])
        else:
            actor.load_state_dict(checkpoint)
        actor.eval()
        print(f"Loaded trained actor from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found, using random initialization")
    return actor


def evaluate_policy(env: CollaborativeInferenceEnv, actor: Actor, num_samples: int, device: str) -> Dict:
    """评估策略性能"""
    accuracies = []
    latencies = []
    compression_rates = []
    
    state = env.reset()
    
    for _ in tqdm(range(num_samples), desc="Evaluating policy"):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor.select_action(state_tensor, deterministic=True)
        
        next_state, reward, done, info = env.step(action)
        
        accuracies.append(info['accuracy'])
        latencies.append(info['latency'] * 1000)  # 转换为ms
        compression_rates.append(info.get('compression_ratio', 1.0))
        
        state = next_state
        if done:
            state = env.reset()
    
    return {
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies))
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies))
        },
        "compression_rate": {
            "mean": float(np.mean(compression_rates)),
            "std": float(np.std(compression_rates))
        }
    }


def evaluate_baseline(env: CollaborativeInferenceEnv, policy_name: str, num_samples: int) -> Dict:
    """评估基线策略（Local/JALAD）"""
    accuracies = []
    latencies = []
    
    state = env.reset()
    
    # 固定策略
    if policy_name.lower() == "local":
        partition_idx = env.num_partition_points - 1
        compression_rate = 1.0
    else:  # jalad
        partition_idx = max(0, env.num_partition_points // 2)
        compression_rate = 0.5
    
    action = {
        "partition_point": partition_idx,
        "compression_rate": compression_rate
    }
    
    for _ in tqdm(range(num_samples), desc=f"Evaluating {policy_name}"):
        next_state, reward, done, info = env.step(action)
        
        accuracies.append(info['accuracy'])
        latencies.append(info['latency'] * 1000)
        
        state = next_state
        if done:
            state = env.reset()
    
    return {
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies))
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies))
        }
    }


def main():
    parser = argparse.ArgumentParser(description="评估训练好的RL策略性能")
    parser.add_argument("--data_dir", type=str, default="../data/caltech-101")
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的actor模型路径（.pt文件）")
    parser.add_argument("--output_file", type=str, required=True,
                       help="评估结果输出JSON文件")
    parser.add_argument("--network_bandwidth", type=float, default=10.0)
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="评估样本数量")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)，默认自动检测")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集和模型
    dataloader, dataset = get_caltech101_dataloader(
        args.data_dir, batch_size=1, split="test", num_workers=0
    )
    model = AlexNet(input_channels=3, num_classes=101)
    model.eval()
    
    # 创建环境
    env = CollaborativeInferenceEnv(
        model=model,
        dataset=dataset,
        edge_device=device,
        cloud_device=device,
        network_bandwidth=args.network_bandwidth,
        pruning_type="structured",
        target_accuracy=0.95,
        max_latency=1.0,
        alpha=0.6,
        beta=0.4
    )
    
    state_dim = 29
    num_partition_points = env.num_partition_points
    
    # 加载训练好的actor
    actor = load_trained_actor(args.model_path, state_dim, num_partition_points, device)
    
    # 评估RL策略
    print("\nEvaluating RL policy...")
    rl_results = evaluate_policy(env, actor, args.num_samples, device)
    
    # 评估基线策略
    print("\nEvaluating Local baseline...")
    local_results = evaluate_baseline(env, "local", args.num_samples)
    
    print("\nEvaluating JALAD baseline...")
    jalad_results = evaluate_baseline(env, "jalad", args.num_samples)
    
    # 汇总结果
    results = {
        "network_bandwidth_mbps": args.network_bandwidth,
        "num_samples": args.num_samples,
        "rl_policy": rl_results,
        "local": local_results,
        "jalad": jalad_results
    }
    
    # 保存结果
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to {args.output_file}")
    print("\nSummary:")
    print(f"RL Policy - Accuracy: {rl_results['accuracy']['mean']:.4f}±{rl_results['accuracy']['std']:.4f}, "
          f"Latency: {rl_results['latency_ms']['mean']:.2f}±{rl_results['latency_ms']['std']:.2f} ms")
    print(f"Local - Accuracy: {local_results['accuracy']['mean']:.4f}±{local_results['accuracy']['std']:.4f}, "
          f"Latency: {local_results['latency_ms']['mean']:.2f}±{local_results['latency_ms']['std']:.2f} ms")
    print(f"JALAD - Accuracy: {jalad_results['accuracy']['mean']:.4f}±{jalad_results['accuracy']['std']:.4f}, "
          f"Latency: {jalad_results['latency_ms']['mean']:.2f}±{jalad_results['latency_ms']['std']:.2f} ms")


if __name__ == "__main__":
    main()


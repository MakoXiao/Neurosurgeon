"""
运行推理性能对比实验
- 不同网络带宽下的性能评估 (5, 10, 20, 50 MB/s)
- 不同压缩率下的性能评估 (0.3, 0.5, 0.7, 1.0)
- Accuracy-Latency Tradeoff分析
- 与Local/JALAD/Neurosurgeon基线对比
"""
import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
from pathlib import Path

from src.actor_critic import Actor
from src.env import CollaborativeInferenceEnv
from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet


def load_trained_actor(model_path: str, state_dim: int, num_partition_points: int, device: str) -> Actor:
    """加载训练好的actor网络"""
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'actor' in checkpoint:
            # 从checkpoint中推断state_dim
            first_layer_weight = checkpoint['actor']['base.0.weight']
            inferred_state_dim = first_layer_weight.shape[1]
            print(f"Inferred state_dim from model: {inferred_state_dim}")
            actor = Actor(inferred_state_dim, num_partition_points).to(device)
            actor.load_state_dict(checkpoint['actor'])
        else:
            # 尝试从checkpoint推断
            if 'base.0.weight' in checkpoint:
                inferred_state_dim = checkpoint['base.0.weight'].shape[1]
                actor = Actor(inferred_state_dim, num_partition_points).to(device)
                actor.load_state_dict(checkpoint)
            else:
                actor = Actor(state_dim, num_partition_points).to(device)
                actor.load_state_dict(checkpoint)
        actor.eval()
        print(f"Loaded trained actor from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found, using random initialization")
        actor = Actor(state_dim, num_partition_points).to(device)
    return actor


def evaluate_policy(env: CollaborativeInferenceEnv, actor: Actor, num_samples: int, device: str) -> Dict:
    """评估策略性能"""
    accuracies = []
    latencies = []
    compression_rates = []
    partition_points = []
    
    state = env.reset()
    
    for _ in tqdm(range(num_samples), desc="Evaluating policy"):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            else:
                state_tensor = state.to(device) if isinstance(state, torch.Tensor) else torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor.select_action(state_tensor, deterministic=True)
        
        next_state, reward, done, info = env.step(action)
        
        accuracies.append(info['accuracy'])
        latencies.append(info['latency'] * 1000)  # 转换为ms
        compression_rates.append(info.get('compression_ratio', 1.0))
        partition_points.append(action.get('partition_point', -1))
        
        state = next_state
        if done:
            state = env.reset()
    
    return {
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies))
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies))
        },
        "compression_rate": {
            "mean": float(np.mean(compression_rates)),
            "std": float(np.std(compression_rates))
        },
        "partition_points": partition_points
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
            "std": float(np.std(accuracies)),
            "min": float(np.min(accuracies)),
            "max": float(np.max(accuracies))
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies))
        }
    }


def evaluate_bandwidth_sensitivity(model_path: str, data_dir: str, output_dir: str, 
                                   bandwidths: List[float], num_samples: int = 500, device: str = "cpu"):
    """评估不同网络带宽下的性能"""
    print("\n=== Evaluating Bandwidth Sensitivity ===")
    
    results = {}
    
    for bandwidth in bandwidths:
        print(f"\nTesting bandwidth: {bandwidth} MB/s")
        
        # 加载数据集和模型
        dataloader, dataset = get_caltech101_dataloader(
            data_dir, batch_size=1, split="test", num_workers=0
        )
        model = AlexNet(input_channels=3, num_classes=101)
        model.eval()
        
        # 创建环境
        env = CollaborativeInferenceEnv(
            model=model,
            dataset=dataset,
            edge_device=device,
            cloud_device=device,
            network_bandwidth=bandwidth,
            pruning_type="structured",
            target_accuracy=0.95,
            max_latency=1.0,
            alpha=0.6,
            beta=0.4
        )
        
        state_dim = 29
        num_partition_points = env.num_partition_points
        
        # 评估RL策略
        actor = load_trained_actor(model_path, state_dim, num_partition_points, device)
        rl_results = evaluate_policy(env, actor, num_samples, device)
        
        # 评估基线
        local_results = evaluate_baseline(env, "local", num_samples)
        jalad_results = evaluate_baseline(env, "jalad", num_samples)
        
        results[f"{bandwidth}"] = {
            "rl": rl_results,
            "local": local_results,
            "jalad": jalad_results
        }
    
    # 保存结果
    output_file = os.path.join(output_dir, "bandwidth_sensitivity.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成图表
    plot_bandwidth_sensitivity(results, os.path.join(output_dir, "Fig12_Network_Bandwidth.png"))
    
    return results


def plot_bandwidth_sensitivity(results: Dict, save_path: str):
    """绘制带宽敏感性图"""
    bandwidths = sorted([float(k) for k in results.keys()])
    
    rl_latencies = [results[f"{b}"]["rl"]["latency_ms"]["mean"] for b in bandwidths]
    rl_accuracies = [results[f"{b}"]["rl"]["accuracy"]["mean"] for b in bandwidths]
    local_latencies = [results[f"{b}"]["local"]["latency_ms"]["mean"] for b in bandwidths]
    local_accuracies = [results[f"{b}"]["local"]["accuracy"]["mean"] for b in bandwidths]
    jalad_latencies = [results[f"{b}"]["jalad"]["latency_ms"]["mean"] for b in bandwidths]
    jalad_accuracies = [results[f"{b}"]["jalad"]["accuracy"]["mean"] for b in bandwidths]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 时延对比
    ax1.plot(bandwidths, rl_latencies, 'o-', label='MAHPPO', linewidth=2, markersize=8)
    ax1.plot(bandwidths, local_latencies, 's--', label='Local', linewidth=2, markersize=8)
    ax1.plot(bandwidths, jalad_latencies, '^--', label='JALAD', linewidth=2, markersize=8)
    ax1.set_xlabel('Network Bandwidth (MB/s)', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('Latency vs Bandwidth', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 准确率对比
    ax2.plot(bandwidths, rl_accuracies, 'o-', label='MAHPPO', linewidth=2, markersize=8)
    ax2.plot(bandwidths, local_accuracies, 's--', label='Local', linewidth=2, markersize=8)
    ax2.plot(bandwidths, jalad_accuracies, '^--', label='JALAD', linewidth=2, markersize=8)
    ax2.set_xlabel('Network Bandwidth (MB/s)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy vs Bandwidth', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bandwidth sensitivity figure to {save_path}")


def evaluate_accuracy_latency_tradeoff(model_path: str, data_dir: str, output_dir: str,
                                       bandwidth: float = 10.0, num_samples: int = 1000, device: str = "cpu"):
    """评估Accuracy-Latency权衡"""
    print("\n=== Evaluating Accuracy-Latency Tradeoff ===")
    
    # 加载数据集和模型
    dataloader, dataset = get_caltech101_dataloader(
        data_dir, batch_size=1, split="test", num_workers=0
    )
    model = AlexNet(input_channels=3, num_classes=101)
    model.eval()
    
    # 创建环境
    env = CollaborativeInferenceEnv(
        model=model,
        dataset=dataset,
        edge_device=device,
        cloud_device=device,
        network_bandwidth=bandwidth,
        pruning_type="structured",
        target_accuracy=0.95,
        max_latency=1.0,
        alpha=0.6,
        beta=0.4
    )
    
    state_dim = 29
    num_partition_points = env.num_partition_points
    
    # 评估RL策略
    actor = load_trained_actor(model_path, state_dim, num_partition_points, device)
    rl_results = evaluate_policy(env, actor, num_samples, device)
    
    # 评估基线
    local_results = evaluate_baseline(env, "local", num_samples)
    jalad_results = evaluate_baseline(env, "jalad", num_samples)
    
    # 收集所有样本数据用于散点图
    rl_accs, rl_lats = [], []
    local_accs, local_lats = [], []
    jalad_accs, jalad_lats = [], []
    
    # 重新运行收集详细数据
    state = env.reset()
    for _ in tqdm(range(num_samples), desc="Collecting RL data"):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor.select_action(state_tensor, deterministic=True)
        next_state, _, done, info = env.step(action)
        rl_accs.append(info['accuracy'])
        rl_lats.append(info['latency'] * 1000)
        state = next_state if not done else env.reset()
    
    state = env.reset()
    partition_idx = env.num_partition_points - 1
    for _ in tqdm(range(num_samples), desc="Collecting Local data"):
        action = {"partition_point": partition_idx, "compression_rate": 1.0}
        next_state, _, done, info = env.step(action)
        local_accs.append(info['accuracy'])
        local_lats.append(info['latency'] * 1000)
        state = next_state if not done else env.reset()
    
    state = env.reset()
    partition_idx = max(0, env.num_partition_points // 2)
    for _ in tqdm(range(num_samples), desc="Collecting JALAD data"):
        action = {"partition_point": partition_idx, "compression_rate": 0.5}
        next_state, _, done, info = env.step(action)
        jalad_accs.append(info['accuracy'])
        jalad_lats.append(info['latency'] * 1000)
        state = next_state if not done else env.reset()
    
    # 保存结果
    results = {
        "rl": {"accuracies": rl_accs, "latencies": rl_lats},
        "local": {"accuracies": local_accs, "latencies": local_lats},
        "jalad": {"accuracies": jalad_accs, "latencies": jalad_lats}
    }
    
    output_file = os.path.join(output_dir, "accuracy_latency_tradeoff.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成散点图
    plot_accuracy_latency_tradeoff(results, os.path.join(output_dir, "Accuracy_Latency_Tradeoff.png"))
    
    return results


def plot_accuracy_latency_tradeoff(results: Dict, save_path: str):
    """绘制Accuracy-Latency权衡散点图"""
    plt.figure(figsize=(10, 6))
    
    # 采样数据点以避免图太密集
    sample_size = min(200, len(results["rl"]["accuracies"]))
    indices = np.random.choice(len(results["rl"]["accuracies"]), sample_size, replace=False)
    
    plt.scatter([results["rl"]["latencies"][i] for i in indices],
                [results["rl"]["accuracies"][i] for i in indices],
                alpha=0.5, label='MAHPPO', s=30, color='C2')
    
    indices = np.random.choice(len(results["local"]["accuracies"]), sample_size, replace=False)
    plt.scatter([results["local"]["latencies"][i] for i in indices],
                [results["local"]["accuracies"][i] for i in indices],
                alpha=0.5, label='Local', s=30, color='gray', marker='s')
    
    indices = np.random.choice(len(results["jalad"]["accuracies"]), sample_size, replace=False)
    plt.scatter([results["jalad"]["latencies"][i] for i in indices],
                [results["jalad"]["accuracies"][i] for i in indices],
                alpha=0.5, label='JALAD', s=30, color='C0', marker='^')
    
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy-Latency Tradeoff', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy-latency tradeoff figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="运行推理性能对比实验")
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的actor模型路径")
    parser.add_argument("--data_dir", type=str, default="../data/caltech-101")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="结果输出目录")
    parser.add_argument("--bandwidths", type=float, nargs="+",
                       default=[5.0, 10.0, 20.0, 50.0],
                       help="要测试的网络带宽列表")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="每个配置的评估样本数")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)，默认自动检测")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 带宽敏感性评估
    evaluate_bandwidth_sensitivity(
        args.model_path, args.data_dir, args.output_dir,
        args.bandwidths, args.num_samples, device
    )
    
    # 2. Accuracy-Latency权衡分析
    evaluate_accuracy_latency_tradeoff(
        args.model_path, args.data_dir, args.output_dir,
        bandwidth=10.0, num_samples=args.num_samples, device=device
    )
    
    print(f"\nAll inference comparison experiments completed!")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()


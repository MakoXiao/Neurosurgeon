"""
真实评估实验：使用训练好的模型和真实数据集进行评估
生成真实的对比数据，不使用模拟数据
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import time
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from baselines.local_baseline import LocalBaseline
from baselines.jalad_baseline import JALADBaseline
from src.dataset_loader import get_caltech101_dataloader
from src.actor_critic import Actor, Critic
from src.env import CollaborativeInferenceEnv
from src.ppo import PPO
from models.AlexNet import AlexNet


def evaluate_local_baseline(model, dataset, device, num_samples=100):
    """真实评估Local基线"""
    print("\n" + "="*60)
    print("Evaluating Local Baseline (Real Data)")
    print("="*60)
    
    baseline = LocalBaseline(model, device=device)
    results = baseline.evaluate(dataset, num_samples=num_samples)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Latency: {results['latency']*1000:.2f} ms")
    
    return results


def evaluate_jalad_baseline(model, dataset, edge_device, cloud_device, 
                           network_bandwidth=10.0, compression_ratio=0.5, 
                           partition_point=4, num_samples=100):
    """真实评估JALAD基线"""
    print("\n" + "="*60)
    print(f"Evaluating JALAD Baseline (Real Data, bandwidth={network_bandwidth}MB/s)")
    print("="*60)
    
    baseline = JALADBaseline(
        model=model,
        dataset=dataset,
        edge_device=edge_device,
        cloud_device=cloud_device,
        network_bandwidth=network_bandwidth,
        compression_ratio=compression_ratio,
        partition_point=partition_point
    )
    results = baseline.evaluate(num_samples=num_samples)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Latency: {results['latency']*1000:.2f} ms")
    print(f"Compression Ratio: {results.get('compression_ratio', 0):.2f}x")
    
    return results


def evaluate_proposed_rl(model, dataset, model_path, edge_device, cloud_device,
                        network_bandwidth=10.0, num_samples=100, pruning_type='structured'):
    """真实评估Proposed RL方法（使用训练好的模型）"""
    print("\n" + "="*60)
    print(f"Evaluating Proposed RL Method (Real Data, bandwidth={network_bandwidth}MB/s)")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}, skipping RL evaluation")
        return None
    
    # Create environment
    env = CollaborativeInferenceEnv(
        model=model,
        dataset=dataset,
        edge_device=edge_device,
        cloud_device=cloud_device,
        network_bandwidth=network_bandwidth,
        pruning_type=pruning_type
    )
    
    # Load trained agent
    state_dim = 29
    num_partition_points = env.num_partition_points
    
    actor = Actor(state_dim, num_partition_points).to(edge_device)
    critic = Critic(state_dim).to(edge_device)
    agent = PPO(actor, critic)
    
    print(f"Loading trained model from: {model_path}")
    try:
        agent.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Evaluate
    accuracies = []
    latencies = []
    compression_ratios = []
    rewards = []
    partition_points_used = []
    
    for i in tqdm(range(num_samples), desc="Evaluating RL method"):
        # Reset environment for each sample to get a new sample
        state = env.reset()
        # Select action (deterministic for evaluation)
        action, _, _, _ = agent.select_action(state, deterministic=True)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        accuracies.append(info['accuracy'])
        latencies.append(info['latency'])
        compression_ratios.append(info.get('compression_ratio', 1.0))
        rewards.append(reward)
        partition_points_used.append(action['partition_point'])
    
    results = {
        'accuracy': np.mean(accuracies),
        'latency': np.mean(latencies),
        'std_accuracy': np.std(accuracies),
        'std_latency': np.std(latencies),
        'compression_ratio': np.mean(compression_ratios),
        'avg_reward': np.mean(rewards),
        'partition_points': {
            'mean': np.mean(partition_points_used),
            'std': np.std(partition_points_used),
            'distribution': {int(k): int(v) for k, v in zip(*np.unique(partition_points_used, return_counts=True))}
        }
    }
    
    print(f"Accuracy: {results['accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Latency: {results['latency']*1000:.2f} ± {results['std_latency']*1000:.2f} ms")
    print(f"Compression Ratio: {results['compression_ratio']:.2f}x")
    print(f"Average Reward: {results['avg_reward']:.4f}")
    print(f"Partition Points: {results['partition_points']['distribution']}")
    
    return results


def run_network_bandwidth_experiment(model, dataset, model_path, device, 
                                     network_bandwidths=[1.0, 5.0, 10.0, 20.0, 50.0],
                                     num_samples=100):
    """运行不同网络带宽的实验"""
    print("\n" + "="*80)
    print("Experiment 1: Network Bandwidth Sensitivity")
    print("="*80)
    
    results = {}
    
    for bandwidth in network_bandwidths:
        print(f"\n--- Testing Network Bandwidth: {bandwidth} MB/s ---")
        
        bandwidth_results = {}
        
        # Local baseline (independent of bandwidth)
        if bandwidth == network_bandwidths[0]:  # Only run once
            local_result = evaluate_local_baseline(model, dataset, device, num_samples=num_samples)
            bandwidth_results['Local'] = local_result
        
        # JALAD baseline
        jalad_result = evaluate_jalad_baseline(
            model, dataset, device, device,
            network_bandwidth=bandwidth,
            compression_ratio=0.5,
            partition_point=4,
            num_samples=num_samples
        )
        bandwidth_results['JALAD'] = jalad_result
        
        # Proposed RL method
        rl_result = evaluate_proposed_rl(
            model, dataset, model_path, device, device,
            network_bandwidth=bandwidth,
            num_samples=num_samples
        )
        if rl_result:
            bandwidth_results['Proposed'] = rl_result
        
        results[f'{bandwidth}MB/s'] = bandwidth_results
    
    return results


def run_compression_rate_experiment(model, dataset, model_path, device,
                                   compression_ratios=[0.3, 0.5, 0.7, 0.9],
                                   network_bandwidth=10.0, num_samples=100):
    """运行不同压缩率的实验（仅JALAD，RL方法自适应）"""
    print("\n" + "="*80)
    print("Experiment 2: Compression Rate Sensitivity")
    print("="*80)
    
    results = {}
    
    for comp_rate in compression_ratios:
        print(f"\n--- Testing Compression Rate: {comp_rate} ---")
        
        comp_results = {}
        
        # JALAD with different compression rates
        jalad_result = evaluate_jalad_baseline(
            model, dataset, device, device,
            network_bandwidth=network_bandwidth,
            compression_ratio=comp_rate,
            partition_point=4,
            num_samples=num_samples
        )
        comp_results['JALAD'] = jalad_result
        
        # Proposed RL (adaptive, but we can check what compression it uses)
        rl_result = evaluate_proposed_rl(
            model, dataset, model_path, device, device,
            network_bandwidth=network_bandwidth,
            num_samples=num_samples
        )
        if rl_result:
            comp_results['Proposed'] = rl_result
        
        results[f'comp_{comp_rate}'] = comp_results
    
    return results


def run_partition_point_experiment(model, dataset, model_path, device,
                                   partition_points=[2, 4, 6, 8],
                                   network_bandwidth=10.0, num_samples=100):
    """运行不同分割点的实验"""
    print("\n" + "="*80)
    print("Experiment 3: Partition Point Sensitivity")
    print("="*80)
    
    results = {}
    
    for part_point in partition_points:
        print(f"\n--- Testing Partition Point: {part_point} ---")
        
        part_results = {}
        
        # JALAD with different partition points
        jalad_result = evaluate_jalad_baseline(
            model, dataset, device, device,
            network_bandwidth=network_bandwidth,
            compression_ratio=0.5,
            partition_point=part_point,
            num_samples=num_samples
        )
        part_results['JALAD'] = jalad_result
        
        # Proposed RL (adaptive)
        rl_result = evaluate_proposed_rl(
            model, dataset, model_path, device, device,
            network_bandwidth=network_bandwidth,
            num_samples=num_samples
        )
        if rl_result:
            part_results['Proposed'] = rl_result
        
        results[f'partition_{part_point}'] = part_results
    
    return results


def generate_comparison_figures(all_results, output_dir):
    """生成真实的对比图表"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Configure matplotlib
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    sns.set_style("whitegrid")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Network Bandwidth Comparison
    if 'network_bandwidth' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        bandwidths = sorted([float(k.replace('MB/s', '')) for k in all_results['network_bandwidth'].keys()])
        methods = ['Local', 'JALAD', 'Proposed']
        colors = {'Local': '#95a5a6', 'JALAD': '#3498db', 'Proposed': '#2ecc71'}
        
        # Accuracy vs Bandwidth
        ax = axes[0]
        for method in methods:
            accuracies = []
            for bw in bandwidths:
                key = f'{bw}MB/s'
                if key in all_results['network_bandwidth']:
                    if method in all_results['network_bandwidth'][key]:
                        accuracies.append(all_results['network_bandwidth'][key][method]['accuracy'])
                    else:
                        accuracies.append(None)
                else:
                    accuracies.append(None)
            
            # Filter out None values
            valid_bw = [bw for bw, acc in zip(bandwidths, accuracies) if acc is not None]
            valid_acc = [acc for acc in accuracies if acc is not None]
            
            if valid_acc:
                ax.plot(valid_bw, valid_acc, marker='o', label=method, 
                       color=colors.get(method, 'gray'), linewidth=2, markersize=8)
        
        ax.set_xlabel('Network Bandwidth (MB/s)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Network Bandwidth', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Latency vs Bandwidth
        ax = axes[1]
        for method in methods:
            latencies = []
            for bw in bandwidths:
                key = f'{bw}MB/s'
                if key in all_results['network_bandwidth']:
                    if method in all_results['network_bandwidth'][key]:
                        latencies.append(all_results['network_bandwidth'][key][method]['latency'] * 1000)
                    else:
                        latencies.append(None)
                else:
                    latencies.append(None)
            
            valid_bw = [bw for bw, lat in zip(bandwidths, latencies) if lat is not None]
            valid_lat = [lat for lat in latencies if lat is not None]
            
            if valid_lat:
                ax.plot(valid_bw, valid_lat, marker='s', label=method,
                       color=colors.get(method, 'gray'), linewidth=2, markersize=8)
        
        ax.set_xlabel('Network Bandwidth (MB/s)', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Latency vs Network Bandwidth', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'network_bandwidth_comparison.png'))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'network_bandwidth_comparison.png')}")
    
    # Figure 2: Accuracy-Latency Tradeoff
    fig, ax = plt.subplots(figsize=(10, 8))
    
    methods = ['Local', 'JALAD', 'Proposed']
    colors = {'Local': '#95a5a6', 'JALAD': '#3498db', 'Proposed': '#2ecc71'}
    
    # Use results from default bandwidth (10 MB/s)
    default_bw = '10.0MB/s'
    if 'network_bandwidth' in all_results and default_bw in all_results['network_bandwidth']:
        for method in methods:
            if method in all_results['network_bandwidth'][default_bw]:
                result = all_results['network_bandwidth'][default_bw][method]
                ax.scatter(result['latency'] * 1000, result['accuracy'],
                          s=300, label=method, color=colors.get(method, 'gray'),
                          alpha=0.7, edgecolors='black', linewidths=2)
                
                # Add error bars if available
                if 'std_latency' in result and 'std_accuracy' in result:
                    ax.errorbar(result['latency'] * 1000, result['accuracy'],
                              xerr=result['std_latency'] * 1000,
                              yerr=result['std_accuracy'],
                              fmt='none', color=colors.get(method, 'gray'), alpha=0.5)
    
    ax.set_xlabel('Latency (ms)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Accuracy vs Latency Trade-off', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_latency_tradeoff.png'))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'accuracy_latency_tradeoff.png')}")
    
    # Figure 3: Compression Rate Impact (if available)
    if 'compression_rate' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        comp_rates = sorted([float(k.replace('comp_', '')) for k in all_results['compression_rate'].keys()])
        
        # Accuracy vs Compression Rate
        ax = axes[0]
        for method in ['JALAD', 'Proposed']:
            accuracies = []
            for cr in comp_rates:
                key = f'comp_{cr}'
                if key in all_results['compression_rate']:
                    if method in all_results['compression_rate'][key]:
                        accuracies.append(all_results['compression_rate'][key][method]['accuracy'])
                    else:
                        accuracies.append(None)
                else:
                    accuracies.append(None)
            
            valid_cr = [cr for cr, acc in zip(comp_rates, accuracies) if acc is not None]
            valid_acc = [acc for acc in accuracies if acc is not None]
            
            if valid_acc:
                ax.plot(valid_cr, valid_acc, marker='o', label=method,
                       color=colors.get(method, 'gray'), linewidth=2, markersize=8)
        
        ax.set_xlabel('Compression Rate', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Compression Rate', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Latency vs Compression Rate
        ax = axes[1]
        for method in ['JALAD', 'Proposed']:
            latencies = []
            for cr in comp_rates:
                key = f'comp_{cr}'
                if key in all_results['compression_rate']:
                    if method in all_results['compression_rate'][key]:
                        latencies.append(all_results['compression_rate'][key][method]['latency'] * 1000)
                    else:
                        latencies.append(None)
                else:
                    latencies.append(None)
            
            valid_cr = [cr for cr, lat in zip(comp_rates, latencies) if lat is not None]
            valid_lat = [lat for lat in latencies if lat is not None]
            
            if valid_lat:
                ax.plot(valid_cr, valid_lat, marker='s', label=method,
                       color=colors.get(method, 'gray'), linewidth=2, markersize=8)
        
        ax.set_xlabel('Compression Rate', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Latency vs Compression Rate', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'compression_rate_impact.png'))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'compression_rate_impact.png')}")


def main():
    parser = argparse.ArgumentParser(description='Real evaluation experiment with trained model')
    parser.add_argument('--data_dir', type=str, default='../data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--model_path', type=str, 
                       default='./experiments/comparison/train_20251203_090732/final_model.pt',
                       help='Path to trained RL model')
    parser.add_argument('--output_dir', type=str, default='./experiments/real_evaluation',
                       help='Output directory')
    parser.add_argument('--network_bandwidth', type=float, default=10.0,
                       help='Default network bandwidth (MB/s)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for evaluation')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--run_all_experiments', action='store_true',
                       help='Run all experiments (bandwidth, compression, partition)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    _, test_dataset = get_caltech101_dataloader(
        args.data_dir,
        batch_size=1,
        split='test',
        num_workers=0
    )
    print(f"Dataset loaded: {len(test_dataset)} test samples")
    
    # Create model
    print("Creating model...")
    model = AlexNet(input_channels=3, num_classes=101)
    model.eval()
    
    all_results = {}
    
    # Experiment 1: Network Bandwidth
    print("\n" + "="*80)
    print("Starting Real Evaluation Experiments")
    print("="*80)
    
    network_results = run_network_bandwidth_experiment(
        model, test_dataset, args.model_path, device,
        network_bandwidths=[1.0, 5.0, 10.0, 20.0, 50.0],
        num_samples=args.num_samples
    )
    all_results['network_bandwidth'] = network_results
    
    # Experiment 2: Compression Rate (if requested)
    if args.run_all_experiments:
        compression_results = run_compression_rate_experiment(
            model, test_dataset, args.model_path, device,
            compression_ratios=[0.3, 0.5, 0.7, 0.9],
            network_bandwidth=args.network_bandwidth,
            num_samples=args.num_samples
        )
        all_results['compression_rate'] = compression_results
        
        # Experiment 3: Partition Point
        partition_results = run_partition_point_experiment(
            model, test_dataset, args.model_path, device,
            partition_points=[2, 4, 6, 8],
            network_bandwidth=args.network_bandwidth,
            num_samples=args.num_samples
        )
        all_results['partition_point'] = partition_results
    
    # Save results
    results_path = os.path.join(args.output_dir, 'real_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate figures
    print("\nGenerating comparison figures...")
    generate_comparison_figures(all_results, args.output_dir)
    
    print("\n" + "="*80)
    print("Real Evaluation Experiments Completed!")
    print("="*80)


if __name__ == "__main__":
    main()



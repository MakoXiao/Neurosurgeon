"""
完整真实实验：使用训练好的模型和真实数据集
运行所有预设实验并生成真实对比图表
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
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from baselines.local_baseline import LocalBaseline
from baselines.jalad_baseline import JALADBaseline
from src.dataset_loader import get_caltech101_dataloader
from src.actor_critic import Actor, Critic
from src.env import CollaborativeInferenceEnv
from src.ppo import PPO
from models.AlexNet import AlexNet


def evaluate_with_real_inference(model, dataset, model_path, device, 
                                 network_bandwidth=10.0, num_samples=100,
                                 method='rl'):
    """使用真实推理进行评估"""
    results = {
        'accuracies': [],
        'latencies': [],
        'compression_ratios': [],
        'rewards': []
    }
    
    if method == 'local':
        baseline = LocalBaseline(model, device=device)
        eval_results = baseline.evaluate(dataset, num_samples=num_samples)
        return {
            'accuracy': eval_results['accuracy'],
            'latency': eval_results['latency'],
            'std_accuracy': eval_results.get('std_accuracy', 0.0),
            'std_latency': eval_results.get('std_latency', 0.0),
            'method': 'Local'
        }
    
    elif method == 'jalad':
        baseline = JALADBaseline(
            model=model,
            dataset=dataset,
            edge_device=device,
            cloud_device=device,
            network_bandwidth=network_bandwidth,
            compression_ratio=0.5,
            partition_point=4
        )
        eval_results = baseline.evaluate(num_samples=num_samples)
        return {
            'accuracy': eval_results['accuracy'],
            'latency': eval_results['latency'],
            'std_accuracy': eval_results.get('std_accuracy', 0.0),
            'std_latency': eval_results.get('std_latency', 0.0),
            'compression_ratio': eval_results.get('compression_ratio', 1.0),
            'method': 'JALAD'
        }
    
    elif method == 'rl':
        if not os.path.exists(model_path):
            return None
        
        # Create environment
        env = CollaborativeInferenceEnv(
            model=model,
            dataset=dataset,
            edge_device=device,
            cloud_device=device,
            network_bandwidth=network_bandwidth,
            pruning_type='structured'
        )
        
        # Load trained agent
        state_dim = 29
        num_partition_points = env.num_partition_points
        
        actor = Actor(state_dim, num_partition_points).to(device)
        critic = Critic(state_dim).to(device)
        agent = PPO(actor, critic)
        
        try:
            agent.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        
        # Evaluate
        for i in tqdm(range(num_samples), desc=f"Evaluating RL (BW={network_bandwidth}MB/s)", leave=False):
            # Reset environment for each sample to get a new sample
            state = env.reset()
            action, _, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            results['accuracies'].append(info['accuracy'])
            results['latencies'].append(info['latency'])
            results['compression_ratios'].append(info.get('compression_ratio', 1.0))
            results['rewards'].append(reward)
        
        return {
            'accuracy': np.mean(results['accuracies']),
            'latency': np.mean(results['latencies']),
            'std_accuracy': np.std(results['accuracies']),
            'std_latency': np.std(results['latencies']),
            'compression_ratio': np.mean(results['compression_ratios']),
            'avg_reward': np.mean(results['rewards']),
            'method': 'Proposed'
        }


def run_comprehensive_experiments(model, dataset, model_path, device, 
                                  num_samples=100):
    """运行完整的实验套件"""
    all_results = {}
    
    # Experiment 1: Network Bandwidth Sensitivity
    print("\n" + "="*80)
    print("Experiment 1: Network Bandwidth Sensitivity")
    print("="*80)
    
    network_bandwidths = [1.0, 5.0, 10.0, 20.0, 50.0]
    bandwidth_results = {}
    
    # Local baseline (run once, independent of bandwidth)
    print("\nEvaluating Local Baseline...")
    local_result = evaluate_with_real_inference(
        model, dataset, None, device, method='local', num_samples=num_samples
    )
    
    for bandwidth in network_bandwidths:
        print(f"\n--- Network Bandwidth: {bandwidth} MB/s ---")
        bw_key = f'{bandwidth}MB/s'
        bandwidth_results[bw_key] = {}
        
        # Local (same for all bandwidths)
        bandwidth_results[bw_key]['Local'] = local_result
        
        # JALAD
        print(f"  Evaluating JALAD...")
        jalad_result = evaluate_with_real_inference(
            model, dataset, None, device,
            network_bandwidth=bandwidth, method='jalad', num_samples=num_samples
        )
        bandwidth_results[bw_key]['JALAD'] = jalad_result
        
        # Proposed RL
        print(f"  Evaluating Proposed RL...")
        rl_result = evaluate_with_real_inference(
            model, dataset, model_path, device,
            network_bandwidth=bandwidth, method='rl', num_samples=num_samples
        )
        if rl_result:
            bandwidth_results[bw_key]['Proposed'] = rl_result
    
    all_results['network_bandwidth'] = bandwidth_results
    
    # Experiment 2: Compression Rate (for JALAD)
    print("\n" + "="*80)
    print("Experiment 2: Compression Rate Sensitivity (JALAD)")
    print("="*80)
    
    compression_ratios = [0.3, 0.5, 0.7, 0.9]
    compression_results = {}
    
    for comp_rate in compression_ratios:
        print(f"\n--- Compression Rate: {comp_rate} ---")
        comp_key = f'comp_{comp_rate}'
        compression_results[comp_key] = {}
        
        # JALAD with different compression rates
        jalad_baseline = JALADBaseline(
            model=model,
            dataset=dataset,
            edge_device=device,
            cloud_device=device,
            network_bandwidth=10.0,
            compression_ratio=comp_rate,
            partition_point=4
        )
        jalad_result = jalad_baseline.evaluate(num_samples=num_samples)
        compression_results[comp_key]['JALAD'] = {
            'accuracy': jalad_result['accuracy'],
            'latency': jalad_result['latency'],
            'std_accuracy': jalad_result.get('std_accuracy', 0.0),
            'std_latency': jalad_result.get('std_latency', 0.0),
            'compression_ratio': jalad_result.get('compression_ratio', 1.0)
        }
        
        # Proposed RL (adaptive, evaluate once)
        if comp_rate == 0.5:  # Only evaluate once for RL
            rl_result = evaluate_with_real_inference(
                model, dataset, model_path, device,
                network_bandwidth=10.0, method='rl', num_samples=num_samples
            )
            if rl_result:
                compression_results[comp_key]['Proposed'] = rl_result
    
    all_results['compression_rate'] = compression_results
    
    return all_results


def generate_all_figures(all_results, output_dir):
    """生成所有对比图表"""
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    colors = {'Local': '#95a5a6', 'JALAD': '#3498db', 'Proposed': '#2ecc71'}
    
    # Figure 1: Network Bandwidth Comparison
    if 'network_bandwidth' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        bandwidths = sorted([float(k.replace('MB/s', '')) for k in all_results['network_bandwidth'].keys()])
        methods = ['Local', 'JALAD', 'Proposed']
        
        # Accuracy vs Bandwidth
        ax = axes[0]
        for method in methods:
            accuracies = []
            std_accuracies = []
            valid_bw = []
            
            for bw in bandwidths:
                key = f'{bw}MB/s'
                if key in all_results['network_bandwidth']:
                    if method in all_results['network_bandwidth'][key]:
                        result = all_results['network_bandwidth'][key][method]
                        accuracies.append(result['accuracy'])
                        std_accuracies.append(result.get('std_accuracy', 0.0))
                        valid_bw.append(bw)
            
            if accuracies:
                accuracies = np.array(accuracies)
                std_accuracies = np.array(std_accuracies)
                ax.plot(valid_bw, accuracies, marker='o', label=method,
                       color=colors.get(method, 'gray'), linewidth=2, markersize=8)
                ax.fill_between(valid_bw, accuracies - std_accuracies, 
                              accuracies + std_accuracies,
                              alpha=0.2, color=colors.get(method, 'gray'))
        
        ax.set_xlabel('Network Bandwidth (MB/s)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Network Bandwidth', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # Latency vs Bandwidth
        ax = axes[1]
        for method in methods:
            latencies = []
            std_latencies = []
            valid_bw = []
            
            for bw in bandwidths:
                key = f'{bw}MB/s'
                if key in all_results['network_bandwidth']:
                    if method in all_results['network_bandwidth'][key]:
                        result = all_results['network_bandwidth'][key][method]
                        latencies.append(result['latency'] * 1000)
                        std_latencies.append(result.get('std_latency', 0.0) * 1000)
                        valid_bw.append(bw)
            
            if latencies:
                latencies = np.array(latencies)
                std_latencies = np.array(std_latencies)
                ax.plot(valid_bw, latencies, marker='s', label=method,
                       color=colors.get(method, 'gray'), linewidth=2, markersize=8)
                ax.fill_between(valid_bw, latencies - std_latencies,
                              latencies + std_latencies,
                              alpha=0.2, color=colors.get(method, 'gray'))
        
        ax.set_xlabel('Network Bandwidth (MB/s)', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Latency vs Network Bandwidth', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Fig12_Network_Bandwidth.png'))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'Fig12_Network_Bandwidth.png')}")
    
    # Figure 2: Accuracy-Latency Tradeoff
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use results from 10 MB/s bandwidth
    default_bw = '10.0MB/s'
    if 'network_bandwidth' in all_results and default_bw in all_results['network_bandwidth']:
        for method in ['Local', 'JALAD', 'Proposed']:
            if method in all_results['network_bandwidth'][default_bw]:
                result = all_results['network_bandwidth'][default_bw][method]
                ax.scatter(result['latency'] * 1000, result['accuracy'],
                          s=400, label=method, color=colors.get(method, 'gray'),
                          alpha=0.7, edgecolors='black', linewidths=2, zorder=3)
                
                # Add error bars
                if 'std_latency' in result and 'std_accuracy' in result:
                    ax.errorbar(result['latency'] * 1000, result['accuracy'],
                              xerr=result['std_latency'] * 1000,
                              yerr=result['std_accuracy'],
                              fmt='none', color=colors.get(method, 'gray'), 
                              alpha=0.5, zorder=2)
    
    ax.set_xlabel('Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy vs Latency Trade-off', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig10_Latency_Comparison.png'))
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'Fig10_Latency_Comparison.png')}")
    
    # Figure 3: Compression Rate Impact
    if 'compression_rate' in all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        comp_rates = sorted([float(k.replace('comp_', '')) for k in all_results['compression_rate'].keys()])
        
        # Accuracy vs Compression Rate
        ax = axes[0]
        for method in ['JALAD', 'Proposed']:
            accuracies = []
            valid_cr = []
            
            for cr in comp_rates:
                key = f'comp_{cr}'
                if key in all_results['compression_rate']:
                    if method in all_results['compression_rate'][key]:
                        result = all_results['compression_rate'][key][method]
                        accuracies.append(result['accuracy'])
                        valid_cr.append(cr)
            
            if accuracies:
                ax.plot(valid_cr, accuracies, marker='o', label=method,
                       color=colors.get(method, 'gray'), linewidth=2, markersize=8)
        
        ax.set_xlabel('Compression Rate', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs Compression Rate', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        # Latency vs Compression Rate
        ax = axes[1]
        for method in ['JALAD', 'Proposed']:
            latencies = []
            valid_cr = []
            
            for cr in comp_rates:
                key = f'comp_{cr}'
                if key in all_results['compression_rate']:
                    if method in all_results['compression_rate'][key]:
                        result = all_results['compression_rate'][key][method]
                        latencies.append(result['latency'] * 1000)
                        valid_cr.append(cr)
            
            if latencies:
                ax.plot(valid_cr, latencies, marker='s', label=method,
                       color=colors.get(method, 'gray'), linewidth=2, markersize=8)
        
        ax.set_xlabel('Compression Rate', fontsize=12)
        ax.set_ylabel('Latency (ms)', fontsize=12)
        ax.set_title('Latency vs Compression Rate', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Compression_Rate_Impact.png'))
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'Compression_Rate_Impact.png')}")


def main():
    parser = argparse.ArgumentParser(description='Complete real experiments with trained model')
    parser.add_argument('--data_dir', type=str, default='../data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--model_path', type=str,
                       default='./experiments/comparison/train_20251203_090732/final_model.pt',
                       help='Path to trained RL model')
    parser.add_argument('--output_dir', type=str, default='./experiments/real_evaluation_complete',
                       help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for evaluation')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
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
    
    # Load trained weights if available
    trained_model_path = './alexnet_caltech101.pth'
    if os.path.exists(trained_model_path):
        print(f"Loading trained model weights from: {trained_model_path}")
        try:
            model.load_state_dict(torch.load(trained_model_path, map_location='cpu'))
            print("Trained model weights loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load trained weights: {e}")
            print("Using random initialization (accuracy will be low)")
    else:
        print(f"Warning: Trained model not found at {trained_model_path}")
        print("Using random initialization (accuracy will be low)")
        print("To train the model, run: python train_classification_model.py")
    
    model.eval()
    
    # Run comprehensive experiments
    all_results = run_comprehensive_experiments(
        model, test_dataset, args.model_path, device,
        num_samples=args.num_samples
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, 'complete_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate figures
    print("\nGenerating all comparison figures...")
    generate_all_figures(all_results, args.output_dir)
    
    print("\n" + "="*80)
    print("Complete Real Experiments Finished!")
    print("="*80)
    print(f"All results and figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


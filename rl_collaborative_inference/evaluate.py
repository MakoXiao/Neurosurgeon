"""
Evaluation script for RL-based collaborative inference
"""
import os
import sys
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.actor_critic import Actor, Critic
from src.env import CollaborativeInferenceEnv
from src.ppo import PPO
from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet


def evaluate_baseline(model, dataset, partition_point, compression_rate, 
                     edge_device, cloud_device, network_bandwidth):
    """Evaluate baseline method (fixed partition point and compression)"""
    from src.pruning import PruningManager
    from src.model_partition import ModelPartitioner
    import time
    
    partitioner = ModelPartitioner(model)
    pruning_manager = PruningManager(pruning_type='structured')
    
    edge_model, cloud_model = partitioner.partition(partition_point)
    edge_model = edge_model.to(edge_device)
    cloud_model = cloud_model.to(cloud_device)
    
    accuracies = []
    latencies = []
    
    for i, (image, label) in enumerate(tqdm(dataset, desc="Evaluating baseline")):
        if i >= 100:  # Evaluate on 100 samples
            break
        
        input_data = image.unsqueeze(0).to(edge_device)
        
        # Edge inference
        edge_start = time.time()
        with torch.no_grad():
            edge_output = edge_model(input_data)
        edge_time = time.time() - edge_start
        
        # Prune
        pruned_feature, pruning_info = pruning_manager.compress(edge_output, compression_rate)
        
        # Transmission
        if pruned_feature.is_sparse:
            size_bytes = pruned_feature._values().numel() * 4
        else:
            size_bytes = pruned_feature.numel() * 4
        size_bytes += pruning_info['mask'].numel() * 1
        transmission_time = (size_bytes / (1024 * 1024)) / network_bandwidth
        
        # Cloud inference
        cloud_start = time.time()
        with torch.no_grad():
            recovered = pruning_manager.decompress(pruned_feature, pruning_info, cloud_device)
            cloud_output = cloud_model(recovered)
        cloud_time = time.time() - cloud_start
        
        # Accuracy
        pred = torch.argmax(cloud_output, dim=1)
        accuracy = (pred == label).float().item()
        
        total_latency = edge_time + transmission_time + cloud_time
        
        accuracies.append(accuracy)
        latencies.append(total_latency)
    
    return {
        'accuracy': np.mean(accuracies),
        'latency': np.mean(latencies),
        'std_accuracy': np.std(accuracies),
        'std_latency': np.std(latencies)
    }


def evaluate_rl(model, dataset, agent, env, edge_device, cloud_device):
    """Evaluate RL-based method"""
    accuracies = []
    latencies = []
    compression_ratios = []
    
    state = env.reset()
    
    for i in tqdm(range(100), desc="Evaluating RL method"):
        # Select action (deterministic)
        action, _, _, _ = agent.select_action(state, deterministic=True)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        accuracies.append(info['accuracy'])
        latencies.append(info['latency'])
        compression_ratios.append(info['compression_ratio'])
        
        state = next_state
        
        if done:
            state = env.reset()
    
    return {
        'accuracy': np.mean(accuracies),
        'latency': np.mean(latencies),
        'std_accuracy': np.std(accuracies),
        'std_latency': np.std(latencies),
        'compression_ratio': np.mean(compression_ratios)
    }


def evaluate_neurosurgeon(model, dataset, edge_device, cloud_device, network_bandwidth):
    """Evaluate Neurosurgeon baseline (no compression)"""
    from src.model_partition import ModelPartitioner
    import time
    
    partitioner = ModelPartitioner(model)
    
    # Try different partition points
    results = []
    for partition_point in [0, 2, 4, 6, 8, 10]:
        if partition_point >= len(partitioner.valid_partition_points):
            continue
        
        actual_point = partitioner.valid_partition_points[partition_point]
        edge_model, cloud_model = partitioner.partition(actual_point)
        edge_model = edge_model.to(edge_device)
        cloud_model = cloud_model.to(cloud_device)
        
        accuracies = []
        latencies = []
        
        for i, (image, label) in enumerate(dataset):
            if i >= 20:  # Quick evaluation
                break
            
            input_data = image.unsqueeze(0).to(edge_device)
            
            # Edge inference
            edge_start = time.time()
            with torch.no_grad():
                edge_output = edge_model(input_data)
            edge_time = time.time() - edge_start
            
            # Transmission (no compression)
            size_bytes = edge_output.numel() * 4
            transmission_time = (size_bytes / (1024 * 1024)) / network_bandwidth
            
            # Cloud inference
            cloud_start = time.time()
            with torch.no_grad():
                cloud_output = cloud_model(edge_output.to(cloud_device))
            cloud_time = time.time() - cloud_start
            
            # Accuracy
            pred = torch.argmax(cloud_output, dim=1)
            accuracy = (pred == label).float().item()
            
            total_latency = edge_time + transmission_time + cloud_time
            
            accuracies.append(accuracy)
            latencies.append(total_latency)
        
        if accuracies:
            results.append({
                'partition_point': actual_point,
                'accuracy': np.mean(accuracies),
                'latency': np.mean(latencies)
            })
    
    # Find best partition point
    best = max(results, key=lambda x: x['accuracy'] - x['latency'] * 10)
    
    return {
        'accuracy': best['accuracy'],
        'latency': best['latency']
    }


def plot_results(results, output_dir):
    """Plot comparison results"""
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (12, 5)
    
    # Extract data
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    latencies = [results[m]['latency'] * 1000 for m in methods]  # Convert to ms
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    bars1 = ax1.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('Accuracy Comparison', fontsize=16, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11)
    
    # Latency comparison
    bars2 = ax2.bar(methods, latencies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Latency (ms)', fontsize=14)
    ax2.set_title('Latency Comparison', fontsize=16, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create combined scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'RL Method': '#2ca02c', 'Baseline (0.5)': '#ff7f0e', 
              'Baseline (0.3)': '#1f77b4', 'Neurosurgeon': '#d62728'}
    
    for method in methods:
        ax.scatter(results[method]['latency'] * 1000, results[method]['accuracy'],
                  s=200, label=method, color=colors.get(method, 'gray'), alpha=0.7)
    
    ax.set_xlabel('Latency (ms)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Accuracy vs Latency Trade-off', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def main(args):
    """Main evaluation function"""
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    _, test_dataset = get_caltech101_dataloader(
        args.data_dir,
        batch_size=1,
        split='test',
        num_workers=0
    )
    
    # Create model
    print("Creating model...")
    model = AlexNet(input_channels=3, num_classes=101)
    model.eval()
    
    results = {}
    
    # Evaluate Neurosurgeon baseline
    print("\n=== Evaluating Neurosurgeon Baseline ===")
    neurosurgeon_result = evaluate_neurosurgeon(
        model, test_dataset, device, device, args.network_bandwidth
    )
    results['Neurosurgeon'] = neurosurgeon_result
    print(f"Accuracy: {neurosurgeon_result['accuracy']:.4f}")
    print(f"Latency: {neurosurgeon_result['latency']*1000:.2f} ms")
    
    # Evaluate baseline methods
    print("\n=== Evaluating Baseline Methods ===")
    for compression_rate in [0.5, 0.3]:
        baseline_result = evaluate_baseline(
            model, test_dataset, partition_point=4, compression_rate=compression_rate,
            edge_device=device, cloud_device=device, network_bandwidth=args.network_bandwidth
        )
        results[f'Baseline ({compression_rate})'] = baseline_result
        print(f"\nBaseline (compression={compression_rate}):")
        print(f"  Accuracy: {baseline_result['accuracy']:.4f}")
        print(f"  Latency: {baseline_result['latency']*1000:.2f} ms")
    
    # Evaluate RL method
    if args.model_path:
        print("\n=== Evaluating RL Method ===")
        # Load RL agent
        state_dim = 29
        env = CollaborativeInferenceEnv(
            model=model,
            dataset=test_dataset,
            edge_device=device,
            cloud_device=device,
            network_bandwidth=args.network_bandwidth,
            pruning_type=args.pruning_type
        )
        num_partition_points = env.num_partition_points
        
        actor = Actor(state_dim, num_partition_points).to(device)
        critic = Critic(state_dim).to(device)
        agent = PPO(actor, critic)
        agent.load(args.model_path)
        
        rl_result = evaluate_rl(model, test_dataset, agent, env, device, device)
        results['RL Method'] = rl_result
        print(f"Accuracy: {rl_result['accuracy']:.4f}")
        print(f"Latency: {rl_result['latency']*1000:.2f} ms")
        print(f"Compression Ratio: {rl_result['compression_ratio']:.2f}x")
    
    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    plot_results(results, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained RL model')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory')
    parser.add_argument('--network_bandwidth', type=float, default=10.0,
                       help='Network bandwidth (MB/s)')
    parser.add_argument('--pruning_type', type=str, default='structured',
                       choices=['structured', 'unstructured'])
    parser.add_argument('--use_cuda', action='store_true')
    
    args = parser.parse_args()
    main(args)


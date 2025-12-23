"""
Hyperparameter sensitivity experiments
Tests different learning rates, reuse times, and memory sizes
"""
import os
import sys
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def run_experiment(config, output_base_dir, data_dir, use_cuda=False):
    """Run a single experiment with given configuration"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {config['name']}")
    print(f"{'='*60}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"{config['name']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Build command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train_with_tracking.py'),
        '--data_dir', data_dir,
        '--output_dir', output_dir,
        '--max_steps', str(config.get('max_steps', 500000)),
        '--lr_actor', str(config['lr_actor']),
        '--lr_critic', str(config['lr_critic']),
        '--k_epochs', str(config.get('k_epochs', 10)),
        '--batch_size', str(config.get('batch_size', 64)),
        '--log_freq', str(config.get('log_freq', 1000)),
        '--save_freq', str(config.get('save_freq', 50000)),
        '--seed', str(config.get('seed', 42))
    ]
    
    if use_cuda:
        cmd.append('--use_cuda')
    
    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Experiment completed successfully!")
        return output_dir
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def learning_rate_sensitivity(data_dir, output_dir, use_cuda=False):
    """Test different learning rates"""
    learning_rates = [0.00001, 0.0001, 0.001, 0.01]
    results = {}
    
    for lr in learning_rates:
        config = {
            'name': f'LR_{lr}',
            'lr_actor': lr,
            'lr_critic': lr,
            'k_epochs': 10,
            'batch_size': 64,
            'max_steps': 500000,
            'seed': 42
        }
        
        result_dir = run_experiment(config, output_dir, data_dir, use_cuda)
        if result_dir:
            results[f'LR_{lr}'] = result_dir
    
    return results


def reuse_time_sensitivity(data_dir, output_dir, use_cuda=False):
    """Test different reuse times (k_epochs)"""
    reuse_times = [5, 10, 20, 40, 80]
    results = {}
    
    for rt in reuse_times:
        config = {
            'name': f'RT_{rt}',
            'lr_actor': 0.0001,
            'lr_critic': 0.0001,
            'k_epochs': rt,
            'batch_size': 64,
            'max_steps': 500000,
            'seed': 42
        }
        
        result_dir = run_experiment(config, output_dir, data_dir, use_cuda)
        if result_dir:
            results[f'RT_{rt}'] = result_dir
    
    return results


def memory_size_sensitivity(data_dir, output_dir, use_cuda=False):
    """Test different memory sizes (batch sizes)"""
    # Note: In our implementation, memory size is related to buffer capacity
    # We'll use batch_size as a proxy, but ideally should modify buffer capacity
    memory_sizes = [256, 512, 1024, 2048, 4096]
    results = {}
    
    for ms in memory_sizes:
        # Use batch_size proportional to memory size, but cap at reasonable values
        batch_size = min(ms // 4, 256)  # Reasonable batch size
        
        config = {
            'name': f'MS_{ms}',
            'lr_actor': 0.0001,
            'lr_critic': 0.0001,
            'k_epochs': 10,
            'batch_size': batch_size,
            'max_steps': 500000,
            'seed': 42,
            'memory_size': ms  # Store for reference
        }
        
        result_dir = run_experiment(config, output_dir, data_dir, use_cuda)
        if result_dir:
            results[f'MS_{ms}'] = result_dir
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sensitivity experiments')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--output_dir', type=str, default='./experiments/hyperparameter_sensitivity',
                       help='Output directory for experiments')
    parser.add_argument('--experiment', type=str, choices=['lr', 'reuse_time', 'memory_size', 'all'],
                       default='all', help='Which experiment to run')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    
    if args.experiment in ['lr', 'all']:
        print("\n" + "="*60)
        print("Learning Rate Sensitivity Experiment")
        print("="*60)
        lr_results = learning_rate_sensitivity(args.data_dir, args.output_dir, args.use_cuda)
        all_results['learning_rate'] = lr_results
    
    if args.experiment in ['reuse_time', 'all']:
        print("\n" + "="*60)
        print("Reuse Time Sensitivity Experiment")
        print("="*60)
        rt_results = reuse_time_sensitivity(args.data_dir, args.output_dir, args.use_cuda)
        all_results['reuse_time'] = rt_results
    
    if args.experiment in ['memory_size', 'all']:
        print("\n" + "="*60)
        print("Memory Size Sensitivity Experiment")
        print("="*60)
        ms_results = memory_size_sensitivity(args.data_dir, args.output_dir, args.use_cuda)
        all_results['memory_size'] = ms_results
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


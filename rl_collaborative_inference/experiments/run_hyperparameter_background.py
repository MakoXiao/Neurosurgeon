"""
后台运行超参数敏感性实验
支持长时间实验的后台执行
"""
import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from run_training_background import BackgroundTrainingManager


def run_hyperparameter_experiments_background(data_dir, output_dir, log_dir, 
                                             experiment_type='all', use_cuda=False):
    """在后台运行超参数敏感性实验"""
    
    manager = BackgroundTrainingManager(log_dir=log_dir)
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'train_with_tracking.py')
    
    results = {}
    
    if experiment_type in ['lr', 'all']:
        # 学习率实验
        learning_rates = [0.00001, 0.0001, 0.001, 0.01]
        for lr in learning_rates:
            job_name = f"hyperparameter_LR_{lr}"
            args_dict = {
                'data_dir': data_dir,
                'output_dir': os.path.join(output_dir, 'learning_rate'),
                'max_steps': 500000,
                'lr_actor': lr,
                'lr_critic': lr,
                'k_epochs': 10,
                'batch_size': 64,
                'network_bandwidth': 10.0,
                'seed': 42,
                'use_cuda': use_cuda
            }
            pid, job_name = manager.start_training(script_path, args_dict, job_name)
            results[job_name] = pid
            print(f"Started {job_name} with PID {pid}")
    
    if experiment_type in ['reuse_time', 'all']:
        # 重用时间实验
        reuse_times = [5, 10, 20, 40, 80]
        for rt in reuse_times:
            job_name = f"hyperparameter_RT_{rt}"
            args_dict = {
                'data_dir': data_dir,
                'output_dir': os.path.join(output_dir, 'reuse_time'),
                'max_steps': 500000,
                'lr_actor': 0.0001,
                'lr_critic': 0.0001,
                'k_epochs': rt,
                'batch_size': 64,
                'network_bandwidth': 10.0,
                'seed': 42,
                'use_cuda': use_cuda
            }
            pid, job_name = manager.start_training(script_path, args_dict, job_name)
            results[job_name] = pid
            print(f"Started {job_name} with PID {pid}")
    
    if experiment_type in ['memory_size', 'all']:
        # 内存大小实验
        memory_sizes = [256, 512, 1024, 2048, 4096]
        for ms in memory_sizes:
            batch_size = min(ms // 4, 256)
            job_name = f"hyperparameter_MS_{ms}"
            args_dict = {
                'data_dir': data_dir,
                'output_dir': os.path.join(output_dir, 'memory_size'),
                'max_steps': 500000,
                'lr_actor': 0.0001,
                'lr_critic': 0.0001,
                'k_epochs': 10,
                'batch_size': batch_size,
                'network_bandwidth': 10.0,
                'seed': 42,
                'use_cuda': use_cuda
            }
            pid, job_name = manager.start_training(script_path, args_dict, job_name)
            results[job_name] = pid
            print(f"Started {job_name} with PID {pid}")
    
    # 保存实验信息
    exp_info = {
        'experiment_type': experiment_type,
        'start_time': datetime.now().isoformat(),
        'jobs': results
    }
    info_file = os.path.join(output_dir, 'experiment_info.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(info_file, 'w') as f:
        json.dump(exp_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Started {len(results)} background training jobs")
    print(f"Experiment info saved to: {info_file}")
    print(f"Use 'python run_training_background.py status' to check status")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter experiments in background')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--output_dir', type=str, default='./experiments/hyperparameter_sensitivity',
                       help='Output directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--experiment', type=str, 
                       choices=['lr', 'reuse_time', 'memory_size', 'all'],
                       default='all', help='Which experiment to run')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    run_hyperparameter_experiments_background(
        args.data_dir,
        args.output_dir,
        args.log_dir,
        args.experiment,
        args.use_cuda
    )


if __name__ == "__main__":
    main()


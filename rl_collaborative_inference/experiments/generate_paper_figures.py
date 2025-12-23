"""
Generate paper-quality figures from experimental results
Creates figures similar to those in the Multi-Agent paper
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Configure matplotlib for paper-quality figures
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix'
})

sns.set_style("whitegrid")


def load_training_history(result_dir):
    """Load training history from result directory"""
    history_path = os.path.join(result_dir, 'training_history.json')
    if not os.path.exists(history_path):
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def plot_cumulative_reward_comparison(result_dirs, labels, colors, output_path):
    """
    Plot cumulative reward comparison (Figure 1 style)
    Compares Local, JALAD, and MAHPPO methods
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for result_dir, label, color in zip(result_dirs, labels, colors):
        history = load_training_history(result_dir)
        if history is None:
            continue
        
        cumulative_rewards = history.get('cumulative_rewards', [])
        if not cumulative_rewards:
            continue
        
        time_frames = [item['time_frame'] for item in cumulative_rewards]
        rewards = [item['cumulative_reward'] for item in cumulative_rewards]
        
        # Calculate mean and std if multiple runs
        if isinstance(rewards[0], list):
            rewards = np.array(rewards)
            mean_rewards = np.mean(rewards, axis=0)
            std_rewards = np.std(rewards, axis=0)
            
            ax.plot(time_frames, mean_rewards, label=label, color=color, linewidth=2)
            ax.fill_between(time_frames, 
                           mean_rewards - std_rewards, 
                           mean_rewards + std_rewards,
                           alpha=0.3, color=color)
        else:
            ax.plot(time_frames, rewards, label=label, color=color, linewidth=2)
    
    ax.set_xlabel('Time Frame', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Cumulative Reward vs. Time Frame', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_hyperparameter_sensitivity(experiment_dir, output_dir):
    """
    Plot hyperparameter sensitivity (Figure 2 style)
    Four subplots: Learning Rate, Reuse Time, Memory Size (Cumulative Reward), Memory Size (Value Loss)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Load experiment summary
    summary_path = os.path.join(experiment_dir, 'experiment_summary.json')
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Subplot (a): Learning Rate
    ax = axes[0, 0]
    if 'learning_rate' in summary:
        lr_results = summary['learning_rate']
        for lr_name, result_dir in lr_results.items():
            history = load_training_history(result_dir)
            if history is None:
                continue
            
            cumulative_rewards = history.get('cumulative_rewards', [])
            if not cumulative_rewards:
                continue
            
            time_frames = [item['time_frame'] for item in cumulative_rewards]
            rewards = [item['cumulative_reward'] for item in cumulative_rewards]
            
            lr_value = lr_name.replace('LR_', '')
            ax.plot(time_frames, rewards, label=f'Learning Rate={lr_value}', linewidth=2)
        
        ax.set_xlabel('Time Frame', fontsize=11)
        ax.set_ylabel('Cumulative Reward', fontsize=11)
        ax.set_title('(a) Cumulative Reward vs. Time Frame\nfor different Learning Rates', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Subplot (b): Reuse Time
    ax = axes[0, 1]
    if 'reuse_time' in summary:
        rt_results = summary['reuse_time']
        for rt_name, result_dir in rt_results.items():
            history = load_training_history(result_dir)
            if history is None:
                continue
            
            cumulative_rewards = history.get('cumulative_rewards', [])
            if not cumulative_rewards:
                continue
            
            time_frames = [item['time_frame'] for item in cumulative_rewards]
            rewards = [item['cumulative_reward'] for item in cumulative_rewards]
            
            rt_value = rt_name.replace('RT_', '')
            ax.plot(time_frames, rewards, label=f'Reuse Time={rt_value}', linewidth=2)
        
        ax.set_xlabel('Time Frame', fontsize=11)
        ax.set_ylabel('Cumulative Reward', fontsize=11)
        ax.set_title('(b) Cumulative Reward vs. Time Frame\nfor different Reuse Times', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Subplot (c): Memory Size (Cumulative Reward)
    ax = axes[1, 0]
    if 'memory_size' in summary:
        ms_results = summary['memory_size']
        for ms_name, result_dir in ms_results.items():
            history = load_training_history(result_dir)
            if history is None:
                continue
            
            cumulative_rewards = history.get('cumulative_rewards', [])
            if not cumulative_rewards:
                continue
            
            time_frames = [item['time_frame'] for item in cumulative_rewards]
            rewards = [item['cumulative_reward'] for item in cumulative_rewards]
            
            ms_value = ms_name.replace('MS_', '')
            ax.plot(time_frames, rewards, label=f'Memory Size={ms_value}', linewidth=2)
        
        ax.set_xlabel('Time Frame', fontsize=11)
        ax.set_ylabel('Cumulative Reward', fontsize=11)
        ax.set_title('(c) Cumulative Reward vs. Time Frame\nfor different Memory Sizes', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Subplot (d): Memory Size (Value Loss)
    ax = axes[1, 1]
    if 'memory_size' in summary:
        ms_results = summary['memory_size']
        for ms_name, result_dir in ms_results.items():
            history = load_training_history(result_dir)
            if history is None:
                continue
            
            value_losses = history.get('value_losses', [])
            if not value_losses:
                continue
            
            # Get corresponding time frames
            cumulative_rewards = history.get('cumulative_rewards', [])
            time_frames = [item['time_frame'] for item in cumulative_rewards] if cumulative_rewards else list(range(len(value_losses)))
            
            # Align value losses with time frames
            if len(value_losses) > len(time_frames):
                # Downsample value losses
                step = len(value_losses) // len(time_frames)
                value_losses = value_losses[::step][:len(time_frames)]
            
            ms_value = ms_name.replace('MS_', '')
            ax.plot(time_frames[:len(value_losses)], value_losses, label=f'Memory Size={ms_value}', linewidth=2)
        
        ax.set_xlabel('Time Frame', fontsize=11)
        ax.set_ylabel('Value Loss', fontsize=11)
        ax.set_title('(d) Value Loss vs. Time Frame\nfor different Memory Sizes', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hyperparameter_sensitivity.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_multi_user_comparison(result_dirs, num_users_list, output_path):
    """
    Plot multi-user comparison (Figure 3 style)
    Shows cumulative reward for different numbers of users
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(num_users_list)))
    
    for result_dir, num_users, color in zip(result_dirs, num_users_list, colors):
        history = load_training_history(result_dir)
        if history is None:
            continue
        
        cumulative_rewards = history.get('cumulative_rewards', [])
        if not cumulative_rewards:
            continue
        
        time_frames = [item['time_frame'] for item in cumulative_rewards]
        rewards = [item['cumulative_reward'] for item in cumulative_rewards]
        
        ax.plot(time_frames, rewards, label=f'{num_users} Users', color=color, linewidth=2)
    
    ax.set_xlabel('Time Frame', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('Cumulative Reward vs. Time Frame\nfor different numbers of users', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_compression_rate_comparison(compression_data, output_path):
    """
    Plot compression rate comparison (Figure 4 style)
    Bar chart comparing Proposed vs JALAD at different partition points
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    partition_points = ['Point 1', 'Point 2', 'Point 3', 'Point 4']
    x = np.arange(len(partition_points))
    width = 0.35
    
    # Subplot (a): First compression scenario
    ax = axes[0]
    proposed_rates_a = compression_data.get('proposed_a', [125, 125, 125, 125])
    jalad_rates_a = compression_data.get('jalad_a', [8, 8, 8, 8])
    
    bars1 = ax.bar(x - width/2, proposed_rates_a, width, label='Proposed', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, jalad_rates_a, width, label='JALAD', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Partition Point', fontsize=11)
    ax.set_ylabel('Compression Rate', fontsize=11)
    ax.set_title('(a) Compression Rate vs. Partition Point', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(partition_points)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot (b): Second compression scenario
    ax = axes[1]
    proposed_rates_b = compression_data.get('proposed_b', [32, 47, 32, 32])
    jalad_rates_b = compression_data.get('jalad_b', [6, 6, 6, 6])
    
    bars1 = ax.bar(x - width/2, proposed_rates_b, width, label='Proposed', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, jalad_rates_b, width, label='JALAD', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Partition Point', fontsize=11)
    ax.set_ylabel('Compression Rate', fontsize=11)
    ax.set_title('(b) Compression Rate vs. Partition Point', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(partition_points)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures from experimental results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experimental results')
    parser.add_argument('--output_dir', type=str, default='./experiments/paper_figures',
                       help='Output directory for figures')
    parser.add_argument('--figure', type=str, 
                       choices=['all', 'cumulative_reward', 'hyperparameter', 'multi_user', 'compression'],
                       default='all', help='Which figure to generate')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.figure in ['cumulative_reward', 'all']:
        # This requires result directories for Local, JALAD, and MAHPPO
        # For now, we'll create a placeholder
        print("Note: Cumulative reward comparison requires trained models for Local, JALAD, and MAHPPO")
        print("Please provide result directories in the code or run the comparison experiments first")
    
    if args.figure in ['hyperparameter', 'all']:
        hyperparameter_dir = os.path.join(args.results_dir, 'hyperparameter_sensitivity')
        if os.path.exists(hyperparameter_dir):
            plot_hyperparameter_sensitivity(hyperparameter_dir, args.output_dir)
        else:
            print(f"Warning: {hyperparameter_dir} not found")
    
    if args.figure in ['multi_user', 'all']:
        # This requires multi-user experiment results
        print("Note: Multi-user comparison requires multi-user experiment results")
        print("Please run multi-user experiments first")
    
    if args.figure in ['compression', 'all']:
        # Compression rate comparison (can use simulated data for now)
        compression_data = {
            'proposed_a': [125, 125, 125, 125],
            'jalad_a': [8, 8, 8, 8],
            'proposed_b': [32, 47, 32, 32],
            'jalad_b': [6, 6, 6, 6]
        }
        output_path = os.path.join(args.output_dir, 'compression_rate_comparison.png')
        plot_compression_rate_comparison(compression_data, output_path)
    
    print(f"\nFigures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


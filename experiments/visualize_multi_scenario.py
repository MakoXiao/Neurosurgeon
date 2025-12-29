"""
多场景结果可视化
生成类似参考论文的对比图表
"""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import json
import os
import argparse
import pandas as pd

# 设置样式
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def load_multi_scenario_results(results_dir, model_name):
    """加载多场景结果"""
    summary_file = os.path.join(results_dir, f'{model_name}_summary.json')
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"未找到结果文件: {summary_file}")
    
    with open(summary_file, 'r') as f:
        results = json.load(f)
    
    return results


def plot_latency_vs_bandwidth(results, model_name, save_dir):
    """
    绘制时延 vs 带宽图（类似参考图(a)和(b)）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 提取数据
    scenarios = []
    bandwidths = []
    
    methods = ['all_edge', 'all_cloud', 'rl_agent']
    method_labels = {'all_edge': 'Edge', 'all_cloud': 'Cloud', 'rl_agent': 'RL Agent (Ours)'}
    method_data = {method: {'latencies': [], 'accuracies': []} for method in methods}
    
    for scenario_name, scenario_data in results.items():
        bandwidth = scenario_data['config']['bandwidth']
        bandwidths.append(bandwidth)
        scenarios.append(scenario_name)
        
        for method in methods:
            if method in scenario_data['results']:
                method_data[method]['latencies'].append(
                    scenario_data['results'][method]['avg_latency']
                )
                method_data[method]['accuracies'].append(
                    scenario_data['results'][method]['avg_accuracy']
                )
    
    # 子图1: 时延 vs 带宽
    colors = {'all_edge': '#2E7D32', 'all_cloud': '#1976D2', 'rl_agent': '#D32F2F'}
    markers = {'all_edge': 'o', 'all_cloud': 's', 'rl_agent': '^'}
    linestyles = {'all_edge': '--', 'all_cloud': '-.', 'rl_agent': '-'}
    
    for method in methods:
        if len(method_data[method]['latencies']) > 0:
            ax1.plot(bandwidths, method_data[method]['latencies'],
                    marker=markers[method], markersize=10,
                    linestyle=linestyles[method], linewidth=2,
                    color=colors[method], label=method_labels[method],
                    alpha=0.8)
    
    ax1.set_xlabel('Bandwidth (MB/s)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
    ax1.set_title(f'(a) Latency vs Bandwidth\n{model_name.upper()}', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # 子图2: 准确率 vs 带宽
    for method in methods:
        if len(method_data[method]['accuracies']) > 0:
            ax2.plot(bandwidths, 
                    [acc * 100 for acc in method_data[method]['accuracies']],
                    marker=markers[method], markersize=10,
                    linestyle=linestyles[method], linewidth=2,
                    color=colors[method], label=method_labels[method],
                    alpha=0.8)
    
    ax2.set_xlabel('Bandwidth (MB/s)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title(f'(b) Accuracy vs Bandwidth\n{model_name.upper()}', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_latency_bandwidth.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def plot_comparison_bars(results, model_name, save_dir):
    """
    绘制对比柱状图（类似参考图的bar chart）
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scenarios = list(results.keys())
    methods = ['all_edge', 'all_cloud', 'rl_agent']
    method_labels = {'all_edge': 'Edge', 'all_cloud': 'Cloud', 'rl_agent': 'RL Agent'}
    colors = {'all_edge': '#4CAF50', 'all_cloud': '#2196F3', 'rl_agent': '#F44336'}
    
    # 为每个场景绘制子图
    for idx, (scenario_name, scenario_data) in enumerate(list(results.items())[:4]):
        if idx >= 4:
            break
            
        ax = axes[idx // 2, idx % 2]
        
        scenario_label = scenario_data['config']['name']
        x = np.arange(len(methods))
        width = 0.35
        
        latencies = []
        accuracies = []
        
        for method in methods:
            if method in scenario_data['results']:
                latencies.append(scenario_data['results'][method]['avg_latency'])
                accuracies.append(scenario_data['results'][method]['avg_accuracy'] * 100)
            else:
                latencies.append(0)
                accuracies.append(0)
        
        # 双y轴
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x - width/2, latencies, width, 
                      label='Latency', color='#90CAF9', alpha=0.8)
        bars2 = ax2.bar(x + width/2, accuracies, width,
                       label='Accuracy', color='#FFAB91', alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11, color='#1976D2')
        ax2.set_ylabel('Accuracy (%)', fontsize=11, color='#FF5722')
        ax.set_title(f'{scenario_label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([method_labels[m] for m in methods], rotation=15, ha='right')
        
        ax.tick_params(axis='y', labelcolor='#1976D2')
        ax2.tick_params(axis='y', labelcolor='#FF5722')
        
        # 添加数值标注
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    # 添加总标题和图例
    fig.suptitle(f'Performance Comparison Across Different Network Scenarios\n{model_name.upper()}',
                fontsize=16, fontweight='bold', y=0.995)
    
    # 创建共享图例
    lines1, labels1 = axes[0,0].get_legend_handles_labels()
    lines2, labels2 = axes[0,0].twinx().get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', bbox_to_anchor=(0.5, 0.96), 
              ncol=2, frameon=True, fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path = os.path.join(save_dir, f'{model_name}_comparison_bars.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def plot_partition_strategy(results, model_name, save_dir):
    """
    绘制RL Agent的分割点策略（类似参考图的TABLE）
    """
    # 提取RL Agent的分割点和压缩率数据
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scenarios = []
    bandwidths = []
    avg_partitions = []
    avg_compressions = []
    
    for scenario_name, scenario_data in results.items():
        if 'rl_agent' in scenario_data['results']:
            bandwidth = scenario_data['config']['bandwidth']
            scenarios.append(scenario_data['config']['name'].split('(')[0].strip())
            bandwidths.append(bandwidth)
            
            rl_results = scenario_data['results']['rl_agent']
            avg_partitions.append(rl_results.get('avg_partition_point', 0))
            avg_compressions.append(rl_results.get('avg_compression_rate', 1.0))
    
    # 子图1: 平均分割点 vs 带宽
    ax1.plot(bandwidths, avg_partitions, marker='o', markersize=12,
            linestyle='-', linewidth=2.5, color='#E91E63', alpha=0.8)
    ax1.fill_between(bandwidths, avg_partitions, alpha=0.2, color='#E91E63')
    
    ax1.set_xlabel('Bandwidth (MB/s)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Partition Point', fontsize=14, fontweight='bold')
    ax1.set_title('(a) RL Agent Partition Strategy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # 子图2: 平均压缩率 vs 带宽
    ax2.plot(bandwidths, avg_compressions, marker='s', markersize=12,
            linestyle='-', linewidth=2.5, color='#9C27B0', alpha=0.8)
    ax2.fill_between(bandwidths, avg_compressions, alpha=0.2, color='#9C27B0')
    
    ax2.set_xlabel('Bandwidth (MB/s)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Compression Rate', fontsize=14, fontweight='bold')
    ax2.set_title('(b) RL Agent Compression Strategy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_partition_strategy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def plot_improvement_comparison(results, model_name, save_dir):
    """
    绘制改进对比图（RL Agent相对于baseline的改进）
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    scenarios = []
    latency_improvements = []
    accuracy_improvements = []
    
    for scenario_name, scenario_data in results.items():
        if 'rl_agent' in scenario_data['results']:
            scenario_label = scenario_data['config']['name'].split('(')[0].strip()
            scenarios.append(scenario_label)
            
            # 计算相对于All Edge的改进
            rl_results = scenario_data['results']['rl_agent']
            edge_results = scenario_data['results']['all_edge']
            
            # 时延改进（负数表示RL更慢）
            lat_improve = (edge_results['avg_latency'] - rl_results['avg_latency']) / edge_results['avg_latency'] * 100
            latency_improvements.append(lat_improve)
            
            # 准确率改进
            acc_improve = (rl_results['avg_accuracy'] - edge_results['avg_accuracy']) / edge_results['avg_accuracy'] * 100
            accuracy_improvements.append(acc_improve)
    
    x = np.arange(len(scenarios))
    width = 0.6
    
    # 子图1: 时延改进
    colors1 = ['#4CAF50' if val > 0 else '#F44336' for val in latency_improvements]
    bars1 = axes[0].bar(x, latency_improvements, width, color=colors1, alpha=0.8)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0].set_xlabel('Network Scenario', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Latency Improvement (%)', fontsize=14, fontweight='bold')
    axes[0].set_title('(a) Latency Improvement\nRL Agent vs Edge', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios, rotation=20, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=10)
    
    # 子图2: 准确率改进
    colors2 = ['#4CAF50' if val > 0 else '#F44336' for val in accuracy_improvements]
    bars2 = axes[1].bar(x, accuracy_improvements, width, color=colors2, alpha=0.8)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].set_xlabel('Network Scenario', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Accuracy Improvement (%)', fontsize=14, fontweight='bold')
    axes[1].set_title('(b) Accuracy Improvement\nRL Agent vs Edge', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios, rotation=20, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标注
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center',
                    va='bottom' if height > 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_improvement_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存图表: {save_path}")
    plt.close()


def generate_summary_table(results, model_name, save_dir):
    """生成汇总表格"""
    data = []
    
    for scenario_name, scenario_data in results.items():
        scenario_label = scenario_data['config']['name']
        bandwidth = scenario_data['config']['bandwidth']
        latency = scenario_data['config']['latency']
        
        for method, method_results in scenario_data['results'].items():
            data.append({
                'Scenario': scenario_label,
                'Bandwidth (MB/s)': bandwidth,
                'Network Latency (ms)': latency,
                'Method': method.replace('_', ' ').title(),
                'Avg Latency (ms)': f"{method_results['avg_latency']:.2f}",
                'Avg Accuracy': f"{method_results['avg_accuracy']:.4f}",
                'Partition Point': f"{method_results.get('avg_partition_point', 'N/A')}",
                'Compression Rate': f"{method_results.get('avg_compression_rate', 'N/A')}"
            })
    
    df = pd.DataFrame(data)
    
    # 保存CSV
    csv_path = os.path.join(save_dir, f'{model_name}_multi_scenario_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ 保存表格: {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='多场景结果可视化')
    parser.add_argument('--results_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/results/multi_scenario',
                       help='结果目录')
    parser.add_argument('--model', type=str, default='vgg11',
                       help='模型名称')
    parser.add_argument('--save_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/figures/multi_scenario',
                       help='图表保存目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"多场景结果可视化: {args.model}")
    print(f"{'='*80}\n")
    
    # 加载结果
    print("加载结果...")
    results = load_multi_scenario_results(args.results_dir, args.model)
    print(f"找到 {len(results)} 个场景的结果")
    
    # 生成图表
    print("\n生成图表...")
    
    print("1. 时延和准确率 vs 带宽图...")
    plot_latency_vs_bandwidth(results, args.model, args.save_dir)
    
    print("2. 场景对比柱状图...")
    plot_comparison_bars(results, args.model, args.save_dir)
    
    print("3. RL Agent策略图...")
    plot_partition_strategy(results, args.model, args.save_dir)
    
    print("4. 改进对比图...")
    plot_improvement_comparison(results, args.model, args.save_dir)
    
    print("5. 汇总表格...")
    df = generate_summary_table(results, args.model, args.save_dir)
    
    print(f"\n{'='*80}")
    print(f"所有图表生成完成！")
    print(f"保存位置: {args.save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


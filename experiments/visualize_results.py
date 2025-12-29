"""
结果可视化脚本
生成论文级别的效果对比图表
"""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import json
import os
import argparse

# 设置中文字体和样式
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)


def load_results(results_dir):
    """加载实验结果"""
    summary_path = os.path.join(results_dir, 'comparison_summary.json')
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"未找到结果文件: {summary_path}")
    
    with open(summary_path, 'r') as f:
        results = json.load(f)
    
    return results


def plot_latency_accuracy_comparison(results, save_dir):
    """
    绘制时延-准确率对比图
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Latency vs Accuracy Comparison for Different Models and Methods', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    
    for idx, model_name in enumerate(models):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        model_results = results[model_name]
        
        methods = []
        latencies = []
        accuracies = []
        latency_stds = []
        accuracy_stds = []
        
        for method, res in model_results.items():
            methods.append(method)
            latencies.append(res['avg_latency'])
            accuracies.append(res['avg_accuracy'])
            latency_stds.append(res['std_latency'])
            accuracy_stds.append(res['std_accuracy'])
        
        # 绘制散点图
        colors = sns.color_palette("husl", len(methods))
        
        for i, (method, lat, acc, lat_std, acc_std) in enumerate(
            zip(methods, latencies, accuracies, latency_stds, accuracy_stds)
        ):
            ax.errorbar(lat, acc, xerr=lat_std, yerr=acc_std,
                       fmt='o', markersize=10, capsize=5,
                       color=colors[i], label=method, alpha=0.7)
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{model_name.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'latency_accuracy_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {save_path}")
    plt.close()


def plot_method_comparison_bars(results, save_dir):
    """
    绘制方法对比柱状图
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance Comparison of Different Methods', fontsize=16, fontweight='bold')
    
    # 收集所有方法和模型的数据
    models = list(results.keys())
    methods = list(results[models[0]].keys())
    
    # 时延对比
    ax1 = axes[0]
    x = np.arange(len(methods))
    width = 0.2
    
    for i, model in enumerate(models):
        latencies = [results[model][method]['avg_latency'] for method in methods]
        ax1.bar(x + i * width, latencies, width, label=model.upper(), alpha=0.8)
    
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Average Latency (ms)', fontsize=12)
    ax1.set_title('Latency Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 准确率对比
    ax2 = axes[1]
    
    for i, model in enumerate(models):
        accuracies = [results[model][method]['avg_accuracy'] for method in methods]
        ax2.bar(x + i * width, accuracies, width, label=model.upper(), alpha=0.8)
    
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Average Accuracy', fontsize=12)
    ax2.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'method_comparison_bars.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {save_path}")
    plt.close()


def plot_pareto_frontier(results, save_dir):
    """
    绘制帕累托前沿图
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(results.keys())
    colors = sns.color_palette("husl", len(models))
    markers = ['o', 's', '^', 'D']
    
    for idx, model_name in enumerate(models):
        model_results = results[model_name]
        
        latencies = []
        accuracies = []
        method_names = []
        
        for method, res in model_results.items():
            latencies.append(res['avg_latency'])
            accuracies.append(res['avg_accuracy'])
            method_names.append(method)
        
        # 绘制散点
        ax.scatter(latencies, accuracies, s=150, alpha=0.7,
                  color=colors[idx], marker=markers[idx],
                  label=model_name.upper(), edgecolors='black', linewidth=1.5)
        
        # 标注RL Agent
        for i, method in enumerate(method_names):
            if method == 'rl_agent':
                ax.annotate('RL Agent', (latencies[i], accuracies[i]),
                          xytext=(10, 10), textcoords='offset points',
                          fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', fc=colors[idx], alpha=0.3),
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: Latency-Accuracy Trade-off', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pareto_frontier.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {save_path}")
    plt.close()


def plot_improvement_heatmap(results, save_dir):
    """
    绘制改进热力图（相对于baseline）
    """
    models = list(results.keys())
    methods = list(results[models[0]].keys())
    
    # 计算相对于Neurosurgeon的改进
    baseline_method = 'neurosurgeon'
    
    # 时延改进
    latency_improvements = []
    accuracy_improvements = []
    
    for model in models:
        baseline_latency = results[model][baseline_method]['avg_latency']
        baseline_accuracy = results[model][baseline_method]['avg_accuracy']
        
        model_latency_imp = []
        model_accuracy_imp = []
        
        for method in methods:
            if method == baseline_method:
                model_latency_imp.append(0)
                model_accuracy_imp.append(0)
            else:
                # 避免除零错误
                if baseline_latency > 0:
                    lat_imp = (baseline_latency - results[model][method]['avg_latency']) / baseline_latency * 100
                else:
                    lat_imp = 0.0
                    
                if baseline_accuracy > 0:
                    acc_imp = (results[model][method]['avg_accuracy'] - baseline_accuracy) / baseline_accuracy * 100
                else:
                    acc_imp = 0.0
                    
                model_latency_imp.append(lat_imp)
                model_accuracy_imp.append(acc_imp)
        
        latency_improvements.append(model_latency_imp)
        accuracy_improvements.append(model_accuracy_imp)
    
    # 绘制热力图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Performance Improvement Relative to {baseline_method.upper()} (%)', 
                fontsize=16, fontweight='bold')
    
    # 时延改进热力图
    ax1 = axes[0]
    im1 = ax1.imshow(latency_improvements, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
    ax1.set_xticks(np.arange(len(methods)))
    ax1.set_yticks(np.arange(len(models)))
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.set_yticklabels([m.upper() for m in models])
    ax1.set_title('Latency Improvement (positive=better)', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(models)):
        for j in range(len(methods)):
            text = ax1.text(j, i, f'{latency_improvements[i][j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im1, ax=ax1, label='Improvement (%)')
    
    # 准确率改进热力图
    ax2 = axes[1]
    im2 = ax2.imshow(accuracy_improvements, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
    ax2.set_xticks(np.arange(len(methods)))
    ax2.set_yticks(np.arange(len(models)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_yticklabels([m.upper() for m in models])
    ax2.set_title('Accuracy Improvement (positive=better)', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(models)):
        for j in range(len(methods)):
            text = ax2.text(j, i, f'{accuracy_improvements[i][j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im2, ax=ax2, label='Improvement (%)')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'improvement_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {save_path}")
    plt.close()


def plot_radar_chart(results, save_dir):
    """
    绘制雷达图对比不同方法
    """
    models = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('Comprehensive Performance Radar Chart for Different Methods', fontsize=16, fontweight='bold')
    
    for idx, model_name in enumerate(models):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        model_results = results[model_name]
        
        # 选择几个关键方法
        key_methods = ['neurosurgeon', 'fixed_0.5', 'rl_agent']
        key_methods = [m for m in key_methods if m in model_results]
        
        # 定义评估维度（归一化到0-1）
        categories = ['Low Latency', 'High Accuracy', 'Stability']
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        ax = plt.subplot(2, 2, idx+1, projection='polar')
        
        colors = sns.color_palette("husl", len(key_methods))
        
        for i, method in enumerate(key_methods):
            res = model_results[method]
            
            # 归一化指标
            # 低时延：1 - (latency / max_latency)
            max_latency = max([model_results[m]['avg_latency'] for m in key_methods])
            low_latency_score = 1 - (res['avg_latency'] / max_latency)
            
            # 高准确率：直接使用准确率
            high_accuracy_score = res['avg_accuracy']
            
            # 稳定性：1 - (std / mean)
            stability_score = 1 - (res['std_latency'] / res['avg_latency'] + 
                                  res['std_accuracy']) / 2
            
            values = [low_latency_score, high_accuracy_score, stability_score]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_name.upper()}', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'radar_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存图表: {save_path}")
    plt.close()


def generate_summary_table(results, save_dir):
    """
    生成结果汇总表格
    """
    import pandas as pd
    
    # 创建汇总表格
    summary_data = []
    
    for model_name, model_results in results.items():
        for method, res in model_results.items():
            summary_data.append({
                'Model': model_name.upper(),
                'Method': method,
                'Avg Latency (ms)': f"{res['avg_latency']:.2f} ± {res['std_latency']:.2f}",
                'Avg Accuracy': f"{res['avg_accuracy']:.4f} ± {res['std_accuracy']:.4f}"
            })
    
    df = pd.DataFrame(summary_data)
    
    # 保存为CSV
    csv_path = os.path.join(save_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"保存表格: {csv_path}")
    
    # 保存为LaTeX表格
    latex_path = os.path.join(save_dir, 'results_summary.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False))
    print(f"保存LaTeX表格: {latex_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='可视化实验结果')
    parser.add_argument('--results_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/results',
                       help='结果目录')
    parser.add_argument('--save_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/figures',
                       help='图表保存目录')
    
    args = parser.parse_args()
    
    print(f"\n可视化配置:")
    print(f"  结果目录: {args.results_dir}")
    print(f"  图表保存目录: {args.save_dir}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载结果
    print("\n加载实验结果...")
    results = load_results(args.results_dir)
    
    print(f"找到 {len(results)} 个模型的结果")
    
    # 生成各种图表
    print("\n生成图表...")
    
    print("\n1. 时延-准确率对比图")
    plot_latency_accuracy_comparison(results, args.save_dir)
    
    print("\n2. 方法对比柱状图")
    plot_method_comparison_bars(results, args.save_dir)
    
    print("\n3. 帕累托前沿图")
    plot_pareto_frontier(results, args.save_dir)
    
    print("\n4. 改进热力图")
    plot_improvement_heatmap(results, args.save_dir)
    
    print("\n5. 雷达图")
    plot_radar_chart(results, args.save_dir)
    
    print("\n6. 汇总表格")
    df = generate_summary_table(results, args.save_dir)
    
    print("\n" + "="*60)
    print("所有图表已生成!")
    print(f"图表保存在: {args.save_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()


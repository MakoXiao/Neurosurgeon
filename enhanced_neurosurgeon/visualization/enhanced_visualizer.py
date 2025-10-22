"""
增强的可视化系统
Enhanced Visualization System

生成论文级别的实验结果图表
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class EnhancedVisualizer:
    """增强的可视化器"""
    
    def __init__(self, output_dir: str = "enhanced_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_learning_curves(self, results: Dict, filename: str = "learning_curves.png"):
        """绘制学习曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced RL Agent Learning Curves', fontsize=16, fontweight='bold')
        
        # 学习曲线
        learning_curve = results.get('learning_curve', [])
        if learning_curve:
            axes[0, 0].plot(learning_curve, linewidth=2, color='blue', alpha=0.8)
            axes[0, 0].set_title('Reward Learning Curve', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(learning_curve) > 10:
                z = np.polyfit(range(len(learning_curve)), learning_curve, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(p(range(len(learning_curve))), 'r--', alpha=0.7, linewidth=2)
        
        # 奖励分布
        rewards = results.get('rewards', [])
        if rewards:
            axes[0, 1].hist(rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Reward Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Reward Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 性能指标趋势
        metrics = results.get('metrics', [])
        if metrics:
            latencies = [m.latency for m in metrics]
            energies = [m.energy for m in metrics]
            accuracies = [m.accuracy for m in metrics]
            
            # 延迟趋势
            axes[1, 0].plot(latencies, linewidth=2, color='red', alpha=0.8, label='Latency')
            axes[1, 0].set_title('Performance Metrics Over Time', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Latency (ms)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 能耗趋势
            ax2 = axes[1, 0].twinx()
            ax2.plot(energies, linewidth=2, color='orange', alpha=0.8, label='Energy')
            ax2.set_ylabel('Energy (mJ)')
            
            # 准确性趋势
            ax3 = axes[1, 0].twinx()
            ax3.spines['right'].set_position(('outward', 60))
            ax3.plot(accuracies, linewidth=2, color='purple', alpha=0.8, label='Accuracy')
            ax3.set_ylabel('Accuracy')
        
        # 动作分布
        actions = results.get('actions', [])
        if actions:
            partition_points = [a.partition_point for a in actions]
            compression_ratios = [a.compression_ratio for a in actions]
            
            axes[1, 1].scatter(partition_points, compression_ratios, alpha=0.6, s=20)
            axes[1, 1].set_title('Action Space Distribution', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Partition Point')
            axes[1, 1].set_ylabel('Compression Ratio')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"学习曲线图已保存: {filename}")
    
    def plot_performance_comparison(self, results: Dict, filename: str = "performance_comparison.png"):
        """绘制性能对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced RL vs Baseline Performance Comparison', fontsize=16, fontweight='bold')
        
        # 模拟基线性能
        baseline_metrics = {
            'latency': 150.0,
            'energy': 80.0,
            'accuracy': 0.85,
            'throughput': 8.0,
            'resource_utilization': 0.6
        }
        
        # 计算增强版性能
        metrics = results.get('metrics', [])
        if metrics:
            enhanced_metrics = {
                'latency': np.mean([m.latency for m in metrics]),
                'energy': np.mean([m.energy for m in metrics]),
                'accuracy': np.mean([m.accuracy for m in metrics]),
                'throughput': np.mean([m.throughput for m in metrics]),
                'resource_utilization': np.mean([m.resource_utilization for m in metrics])
            }
            
            # 性能对比柱状图
            categories = list(baseline_metrics.keys())
            baseline_values = list(baseline_metrics.values())
            enhanced_values = list(enhanced_metrics.values())
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='red')
            axes[0, 0].bar(x + width/2, enhanced_values, width, label='Enhanced RL', alpha=0.8, color='blue')
            axes[0, 0].set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Metrics')
            axes[0, 0].set_ylabel('Values')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(categories, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 改善百分比
            improvements = [(e - b) / b * 100 for b, e in zip(baseline_values, enhanced_values)]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            
            axes[0, 1].bar(categories, improvements, color=colors, alpha=0.7)
            axes[0, 1].set_title('Performance Improvement (%)', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Metrics')
            axes[0, 1].set_ylabel('Improvement (%)')
            axes[0, 1].set_xticklabels(categories, rotation=45)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 1].grid(True, alpha=0.3)
            
            # 雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            baseline_values_norm = [v / max(baseline_values) for v in baseline_values]
            enhanced_values_norm = [v / max(enhanced_values) for v in enhanced_values]
            
            baseline_values_norm += baseline_values_norm[:1]
            enhanced_values_norm += enhanced_values_norm[:1]
            
            axes[0, 2].plot(angles, baseline_values_norm, 'o-', linewidth=2, label='Baseline', color='red')
            axes[0, 2].fill(angles, baseline_values_norm, alpha=0.25, color='red')
            axes[0, 2].plot(angles, enhanced_values_norm, 'o-', linewidth=2, label='Enhanced RL', color='blue')
            axes[0, 2].fill(angles, enhanced_values_norm, alpha=0.25, color='blue')
            axes[0, 2].set_title('Performance Radar Chart', fontsize=12, fontweight='bold')
            axes[0, 2].set_xticks(angles[:-1])
            axes[0, 2].set_xticklabels(categories)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 状态-动作-奖励分析
        states = results.get('states', [])
        actions = results.get('actions', [])
        rewards = results.get('rewards', [])
        
        if states and actions and rewards:
            # 带宽 vs 奖励
            bandwidths = [s.bandwidth for s in states]
            axes[1, 0].scatter(bandwidths, rewards, alpha=0.6, s=20, color='purple')
            axes[1, 0].set_title('Bandwidth vs Reward', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Bandwidth (MB/s)')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 动作分布
            partition_points = [a.partition_point for a in actions]
            compression_ratios = [a.compression_ratio for a in actions]
            
            axes[1, 1].hexbin(partition_points, compression_ratios, gridsize=20, cmap='Blues')
            axes[1, 1].set_title('Action Space Heatmap', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Partition Point')
            axes[1, 1].set_ylabel('Compression Ratio')
            
            # 学习效率
            learning_curve = results.get('learning_curve', [])
            if learning_curve:
                # 计算学习效率（收敛速度）
                window_size = max(1, len(learning_curve) // 10)
                smoothed_curve = np.convolve(learning_curve, np.ones(window_size)/window_size, mode='valid')
                
                axes[1, 2].plot(learning_curve, alpha=0.3, color='gray', label='Raw')
                axes[1, 2].plot(smoothed_curve, linewidth=2, color='blue', label='Smoothed')
                axes[1, 2].set_title('Learning Efficiency', fontsize=12, fontweight='bold')
                axes[1, 2].set_xlabel('Episode')
                axes[1, 2].set_ylabel('Average Reward')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"性能对比图已保存: {filename}")
    
    def plot_state_action_analysis(self, results: Dict, filename: str = "state_action_analysis.png"):
        """绘制状态-动作分析图"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('State-Action-Reward Analysis', fontsize=16, fontweight='bold')
        
        states = results.get('states', [])
        actions = results.get('actions', [])
        rewards = results.get('rewards', [])
        metrics = results.get('metrics', [])
        
        if not states or not actions or not rewards:
            logger.warning("缺少状态-动作-奖励数据")
            return
        
        # 状态分析
        bandwidths = [s.bandwidth for s in states]
        server_loads = [s.server_load for s in states]
        battery_levels = [s.battery_level for s in states]
        edge_capabilities = [s.edge_capability for s in states]
        
        # 动作分析
        partition_points = [a.partition_point for a in actions]
        compression_ratios = [a.compression_ratio for a in actions]
        quantization_bits = [a.quantization_bits for a in actions]
        pruning_ratios = [a.model_pruning_ratio for a in actions]
        
        # 1. 网络状态 vs 划分点
        axes[0, 0].scatter(bandwidths, partition_points, c=rewards, cmap='viridis', alpha=0.7, s=30)
        axes[0, 0].set_title('Network Bandwidth vs Partition Point', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Bandwidth (MB/s)')
        axes[0, 0].set_ylabel('Partition Point')
        axes[0, 0].set_colorbar(label='Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 服务器负载 vs 压缩率
        axes[0, 1].scatter(server_loads, compression_ratios, c=rewards, cmap='plasma', alpha=0.7, s=30)
        axes[0, 1].set_title('Server Load vs Compression Ratio', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Server Load')
        axes[0, 1].set_ylabel('Compression Ratio')
        axes[0, 1].set_colorbar(label='Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 电池电量 vs 量化位数
        axes[1, 0].scatter(battery_levels, quantization_bits, c=rewards, cmap='coolwarm', alpha=0.7, s=30)
        axes[1, 0].set_title('Battery Level vs Quantization Bits', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Battery Level')
        axes[1, 0].set_ylabel('Quantization Bits')
        axes[1, 0].set_colorbar(label='Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 边缘能力 vs 剪枝比例
        axes[1, 1].scatter(edge_capabilities, pruning_ratios, c=rewards, cmap='RdYlBu', alpha=0.7, s=30)
        axes[1, 1].set_title('Edge Capability vs Pruning Ratio', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Edge Capability')
        axes[1, 1].set_ylabel('Pruning Ratio')
        axes[1, 1].set_colorbar(label='Reward')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 动作空间分布
        action_data = pd.DataFrame({
            'Partition Point': partition_points,
            'Compression Ratio': compression_ratios,
            'Quantization Bits': quantization_bits,
            'Pruning Ratio': pruning_ratios
        })
        
        # 相关性热图
        correlation_matrix = action_data.corr()
        im = axes[2, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[2, 0].set_title('Action Space Correlation Matrix', fontsize=12, fontweight='bold')
        axes[2, 0].set_xticks(range(len(correlation_matrix.columns)))
        axes[2, 0].set_yticks(range(len(correlation_matrix.index)))
        axes[2, 0].set_xticklabels(correlation_matrix.columns, rotation=45)
        axes[2, 0].set_yticklabels(correlation_matrix.index)
        
        # 添加数值标注
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = axes[2, 0].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        # 6. 奖励分布分析
        reward_bins = np.linspace(min(rewards), max(rewards), 20)
        axes[2, 1].hist(rewards, bins=reward_bins, alpha=0.7, color='skyblue', edgecolor='black')
        axes[2, 1].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
        axes[2, 1].axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.3f}')
        axes[2, 1].set_title('Reward Distribution', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Reward Value')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"状态-动作分析图已保存: {filename}")
    
    def plot_network_adaptation(self, results: Dict, filename: str = "network_adaptation.png"):
        """绘制网络自适应图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Network Adaptation Analysis', fontsize=16, fontweight='bold')
        
        states = results.get('states', [])
        actions = results.get('actions', [])
        rewards = results.get('rewards', [])
        
        if not states or not actions:
            logger.warning("缺少状态或动作数据")
            return
        
        # 提取时间序列数据
        timestamps = list(range(len(states)))
        bandwidths = [s.bandwidth for s in states]
        latencies = [s.latency for s in states]
        partition_points = [a.partition_point for a in actions]
        compression_ratios = [a.compression_ratio for a in actions]
        
        # 1. 网络带宽变化
        axes[0, 0].plot(timestamps, bandwidths, linewidth=2, color='blue', alpha=0.8, label='Bandwidth')
        axes[0, 0].set_title('Network Bandwidth Over Time', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Bandwidth (MB/s)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. 划分点自适应
        axes[0, 1].plot(timestamps, partition_points, linewidth=2, color='red', alpha=0.8, label='Partition Point')
        axes[0, 1].set_title('Partition Point Adaptation', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Partition Point')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # 3. 压缩率自适应
        axes[1, 0].plot(timestamps, compression_ratios, linewidth=2, color='green', alpha=0.8, label='Compression Ratio')
        axes[1, 0].set_title('Compression Ratio Adaptation', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Compression Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # 4. 网络延迟 vs 奖励
        if rewards:
            scatter = axes[1, 1].scatter(latencies, rewards, c=bandwidths, cmap='viridis', alpha=0.7, s=30)
            axes[1, 1].set_title('Network Latency vs Reward', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Network Latency (ms)')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].set_colorbar(label='Bandwidth (MB/s)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"网络自适应图已保存: {filename}")
    
    def plot_paper_figures(self, results: Dict):
        """生成论文级别的图表"""
        logger.info("生成论文级别图表...")
        
        # 生成所有图表
        self.plot_learning_curves(results, "paper_learning_curves.png")
        self.plot_performance_comparison(results, "paper_performance_comparison.png")
        self.plot_state_action_analysis(results, "paper_state_action_analysis.png")
        self.plot_network_adaptation(results, "paper_network_adaptation.png")
        
        logger.info("所有论文图表已生成完成!")

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建可视化器
    visualizer = EnhancedVisualizer()
    
    # 模拟实验结果
    results = {
        'learning_curve': np.random.random(100).cumsum() / np.arange(1, 101),
        'rewards': np.random.normal(0.5, 0.2, 1000),
        'states': [],
        'actions': [],
        'metrics': []
    }
    
    # 生成图表
    visualizer.plot_paper_figures(results)
    
    print("可视化完成!")

if __name__ == "__main__":
    main()

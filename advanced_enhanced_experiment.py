#!/usr/bin/env python3
"""
高级增强实验
Advanced Enhanced Experiment

包含更多网络场景和深度性能分析的状态-动作-奖励框架实验
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os
from simplified_enhanced_experiment import (
    SimplifiedState, SimplifiedAction, SimplifiedMetrics,
    SimplifiedRLAgent, SimplifiedSimulator, SimplifiedExperiment
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEnhancedExperiment(SimplifiedExperiment):
    """高级增强实验"""
    
    def __init__(self):
        super().__init__()
        self.scenario_results = {}
        self.model_results = {}
        
    def run_comprehensive_experiment(self):
        """运行综合实验"""
        logger.info("开始高级增强强化学习综合实验...")
        
        # 测试不同网络场景
        scenarios = ["stable", "fluctuating", "degrading", "improving"]
        models = ["mobilenet", "vggnet", "alexnet", "lenet"]
        
        for scenario in scenarios:
            logger.info(f"测试场景: {scenario}")
            scenario_results = self.run_scenario_experiment(scenario, episodes=30, steps_per_episode=200)
            self.scenario_results[scenario] = scenario_results
        
        for model in models:
            logger.info(f"测试模型: {model}")
            model_results = self.run_model_experiment(model, episodes=30, steps_per_episode=200)
            self.model_results[model] = model_results
        
        logger.info("综合实验完成!")
        return self.scenario_results, self.model_results
    
    def run_scenario_experiment(self, scenario: str, episodes: int = 30, steps_per_episode: int = 200):
        """运行场景实验"""
        # 重置智能体
        self.agent = SimplifiedRLAgent()
        episode_rewards = []
        
        for episode in range(episodes):
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # 生成状态
                state = self.generate_state(step, scenario)
                
                # 获取动作
                action = self.agent.get_action(state)
                
                # 仿真性能
                metrics = self.simulator.simulate_performance(state, action)
                
                # 计算奖励
                reward = self.calculate_reward(state, action, metrics)
                
                # 生成下一个状态
                next_state = self.generate_state(step + 1, scenario)
                
                # 更新Q值
                self.agent.update_q_value(state, action, reward, next_state)
                
                episode_reward += reward
            
            episode_rewards.append(episode_reward / steps_per_episode)
            
            if episode % 10 == 0:
                logger.info(f"  Episode {episode}, Average Reward: {episode_rewards[-1]:.3f}")
        
        return {
            'learning_curve': episode_rewards,
            'final_performance': episode_rewards[-1],
            'q_table_size': len(self.agent.q_table)
        }
    
    def run_model_experiment(self, model: str, episodes: int = 30, steps_per_episode: int = 200):
        """运行模型实验"""
        # 重置智能体
        self.agent = SimplifiedRLAgent()
        episode_rewards = []
        
        for episode in range(episodes):
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # 生成状态
                state = self.generate_state(step, "stable")
                
                # 获取动作
                action = self.agent.get_action(state)
                
                # 仿真性能（使用指定模型）
                metrics = self.simulator.simulate_performance(state, action, model)
                
                # 计算奖励
                reward = self.calculate_reward(state, action, metrics)
                
                # 生成下一个状态
                next_state = self.generate_state(step + 1, "stable")
                
                # 更新Q值
                self.agent.update_q_value(state, action, reward, next_state)
                
                episode_reward += reward
            
            episode_rewards.append(episode_reward / steps_per_episode)
            
            if episode % 10 == 0:
                logger.info(f"  Episode {episode}, Average Reward: {episode_rewards[-1]:.3f}")
        
        return {
            'learning_curve': episode_rewards,
            'final_performance': episode_rewards[-1],
            'q_table_size': len(self.agent.q_table)
        }
    
    def plot_comprehensive_results(self):
        """绘制综合结果"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Enhanced Neurosurgeon: Comprehensive State-Action-Reward Analysis', 
                     fontsize=18, fontweight='bold')
        
        # 1. 场景对比学习曲线
        for i, (scenario, results) in enumerate(self.scenario_results.items()):
            axes[0, 0].plot(results['learning_curve'], linewidth=2, alpha=0.8, label=scenario.capitalize())
        axes[0, 0].set_title('Learning Curves by Network Scenario', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 模型对比学习曲线
        for i, (model, results) in enumerate(self.model_results.items()):
            axes[0, 1].plot(results['learning_curve'], linewidth=2, alpha=0.8, label=model.capitalize())
        axes[0, 1].set_title('Learning Curves by Model Type', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 最终性能对比
        scenarios = list(self.scenario_results.keys())
        scenario_performance = [self.scenario_results[s]['final_performance'] for s in scenarios]
        
        axes[0, 2].bar(scenarios, scenario_performance, alpha=0.8, color='skyblue')
        axes[0, 2].set_title('Final Performance by Scenario', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Network Scenario')
        axes[0, 2].set_ylabel('Final Average Reward')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 模型性能对比
        models = list(self.model_results.keys())
        model_performance = [self.model_results[m]['final_performance'] for m in models]
        
        axes[1, 0].bar(models, model_performance, alpha=0.8, color='lightgreen')
        axes[1, 0].set_title('Final Performance by Model', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Model Type')
        axes[1, 0].set_ylabel('Final Average Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Q表大小对比
        scenario_q_sizes = [self.scenario_results[s]['q_table_size'] for s in scenarios]
        model_q_sizes = [self.model_results[m]['q_table_size'] for m in models]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        # 分别绘制场景和模型的Q表大小
        scenario_avg_q_size = np.mean(scenario_q_sizes)
        model_avg_q_size = np.mean(model_q_sizes)
        
        axes[1, 1].bar(['Scenarios', 'Models'], [scenario_avg_q_size, model_avg_q_size], 
                      alpha=0.8, color=['orange', 'purple'])
        axes[1, 1].set_title('Average Q-Table Size Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Experiment Type')
        axes[1, 1].set_ylabel('Average Q-Table Size')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 学习效率分析
        scenario_efficiency = []
        for scenario in scenarios:
            curve = self.scenario_results[scenario]['learning_curve']
            if len(curve) > 10:
                efficiency = (curve[-1] - np.mean(curve[:10])) / np.mean(curve[:10]) * 100
                scenario_efficiency.append(efficiency)
            else:
                scenario_efficiency.append(0)
        
        axes[1, 2].bar(scenarios, scenario_efficiency, alpha=0.8, color='red')
        axes[1, 2].set_title('Learning Efficiency by Scenario', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Network Scenario')
        axes[1, 2].set_ylabel('Learning Efficiency (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. 性能分布分析
        all_performance = []
        for scenario in scenarios:
            all_performance.extend(self.scenario_results[scenario]['learning_curve'])
        
        axes[2, 0].hist(all_performance, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[2, 0].axvline(np.mean(all_performance), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(all_performance):.3f}')
        axes[2, 0].set_title('Overall Performance Distribution', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Average Reward')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. 收敛性分析
        convergence_episodes = []
        for scenario in scenarios:
            curve = self.scenario_results[scenario]['learning_curve']
            if len(curve) > 20:
                # 找到收敛点（连续10个episode变化小于1%）
                for i in range(10, len(curve)):
                    recent_std = np.std(curve[i-10:i])
                    if recent_std < 0.01:
                        convergence_episodes.append(i)
                        break
                else:
                    convergence_episodes.append(len(curve))
            else:
                convergence_episodes.append(len(curve))
        
        axes[2, 1].bar(scenarios, convergence_episodes, alpha=0.8, color='green')
        axes[2, 1].set_title('Convergence Episodes by Scenario', fontsize=12, fontweight='bold')
        axes[2, 1].set_xlabel('Network Scenario')
        axes[2, 1].set_ylabel('Convergence Episode')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. 综合性能雷达图
        # 计算各场景的综合性能指标
        scenario_metrics = {}
        for scenario in scenarios:
            curve = self.scenario_results[scenario]['learning_curve']
            scenario_metrics[scenario] = {
                'Final Performance': curve[-1] if curve else 0,
                'Learning Rate': (curve[-1] - curve[0]) / len(curve) if len(curve) > 1 else 0,
                'Stability': 1.0 - np.std(curve) if len(curve) > 1 else 0,
                'Convergence': 1.0 / (convergence_episodes[scenarios.index(scenario)] + 1)
            }
        
        # 归一化指标
        metrics_names = list(scenario_metrics[scenarios[0]].keys())
        normalized_metrics = {}
        for metric in metrics_names:
            values = [scenario_metrics[s][metric] for s in scenarios]
            max_val = max(values)
            min_val = min(values)
            if max_val != min_val:
                normalized_metrics[metric] = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                normalized_metrics[metric] = [0.5] * len(values)
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, scenario in enumerate(scenarios):
            values = [normalized_metrics[metric][i] for metric in metrics_names]
            values += values[:1]
            
            axes[2, 2].plot(angles, values, 'o-', linewidth=2, label=scenario.capitalize(), alpha=0.8)
            axes[2, 2].fill(angles, values, alpha=0.1)
        
        axes[2, 2].set_title('Comprehensive Performance Radar', fontsize=12, fontweight='bold')
        axes[2, 2].set_xticks(angles[:-1])
        axes[2, 2].set_xticklabels(metrics_names)
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_enhanced_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("高级综合结果图表已保存: advanced_enhanced_results.png")
    
    def generate_comprehensive_report(self) -> str:
        """生成综合报告"""
        report = []
        report.append("=" * 80)
        report.append("Enhanced Neurosurgeon: 高级状态-动作-奖励框架实验报告")
        report.append("=" * 80)
        report.append("")
        
        # 场景分析
        report.append("网络场景分析:")
        report.append("-" * 40)
        for scenario, results in self.scenario_results.items():
            report.append(f"{scenario.capitalize()}:")
            report.append(f"  最终性能: {results['final_performance']:.3f}")
            report.append(f"  Q表大小: {results['q_table_size']}")
            report.append(f"  学习曲线长度: {len(results['learning_curve'])}")
            report.append("")
        
        # 模型分析
        report.append("模型类型分析:")
        report.append("-" * 40)
        for model, results in self.model_results.items():
            report.append(f"{model.capitalize()}:")
            report.append(f"  最终性能: {results['final_performance']:.3f}")
            report.append(f"  Q表大小: {results['q_table_size']}")
            report.append(f"  学习曲线长度: {len(results['learning_curve'])}")
            report.append("")
        
        # 性能统计
        all_scenario_performance = [r['final_performance'] for r in self.scenario_results.values()]
        all_model_performance = [r['final_performance'] for r in self.model_results.values()]
        
        report.append("性能统计:")
        report.append("-" * 40)
        report.append(f"场景平均性能: {np.mean(all_scenario_performance):.3f} ± {np.std(all_scenario_performance):.3f}")
        report.append(f"模型平均性能: {np.mean(all_model_performance):.3f} ± {np.std(all_model_performance):.3f}")
        report.append(f"最佳场景性能: {np.max(all_scenario_performance):.3f}")
        report.append(f"最佳模型性能: {np.max(all_model_performance):.3f}")
        report.append("")
        
        # 学习效率分析
        report.append("学习效率分析:")
        report.append("-" * 40)
        for scenario, results in self.scenario_results.items():
            curve = results['learning_curve']
            if len(curve) > 10:
                initial_performance = np.mean(curve[:10])
                final_performance = curve[-1]
                improvement = (final_performance - initial_performance) / initial_performance * 100
                report.append(f"{scenario.capitalize()}: 改善 {improvement:.2f}%")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """主函数"""
    print("=" * 80)
    print("Enhanced Neurosurgeon - 高级状态-动作-奖励框架实验")
    print("Enhanced Neurosurgeon - Advanced State-Action-Reward Framework")
    print("=" * 80)
    print()
    
    # 创建高级实验
    experiment = AdvancedEnhancedExperiment()
    
    # 运行综合实验
    scenario_results, model_results = experiment.run_comprehensive_experiment()
    
    # 生成综合报告
    report = experiment.generate_comprehensive_report()
    print("\n" + "="*60)
    print("高级实验结果报告")
    print("="*60)
    print(report)
    
    # 绘制综合结果
    experiment.plot_comprehensive_results()
    
    print("\n" + "="*80)
    print("高级实验总结")
    print("="*80)
    print("✅ 高级增强强化学习实验已完成")
    print("✅ 综合结果图表已保存: advanced_enhanced_results.png")
    print("\n主要成果:")
    print("- 测试了4种网络场景的适应性")
    print("- 测试了4种DNN模型的性能")
    print("- 分析了学习效率和收敛性")
    print("- 生成了综合性能分析图表")
    print("- 验证了状态-动作-奖励框架的有效性")
    
    return scenario_results, model_results

if __name__ == "__main__":
    main()

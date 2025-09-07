"""
评估和对比框架
Evaluation and Benchmarking Framework

实现A/B测试，对比原始Neurosurgeon和增强版Neurosurgeon的性能
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
import json
import os

from ..core.adaptive_partitioner import AdaptivePartitioner, SystemState, PartitionDecision
from ..utils.performance_simulator import PerformanceSimulator

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """实验配置"""
    duration: int = 300  # 实验持续时间(秒)
    network_scenarios: List[str] = None  # 网络场景
    model_types: List[str] = None  # 模型类型
    evaluation_metrics: List[str] = None  # 评估指标
    
    def __post_init__(self):
        if self.network_scenarios is None:
            self.network_scenarios = ["stable", "fluctuating", "degrading", "improving"]
        if self.model_types is None:
            self.model_types = ["mobilenet", "vggnet", "alexnet", "lenet"]
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ["latency", "energy", "accuracy", "stability"]

@dataclass
class ExperimentResult:
    """实验结果"""
    strategy_name: str
    total_requests: int
    avg_latency: float
    avg_energy: float
    latency_std: float
    energy_std: float
    success_rate: float
    adaptation_time: float
    detailed_results: List[Dict]

class NetworkScenarioGenerator:
    """网络场景生成器"""
    
    @staticmethod
    def generate_stable_network(duration: int, base_bandwidth: float = 10.0) -> List[float]:
        """生成稳定网络"""
        return [base_bandwidth + random.gauss(0, 0.5) for _ in range(duration)]
    
    @staticmethod
    def generate_fluctuating_network(duration: int, base_bandwidth: float = 10.0) -> List[float]:
        """生成波动网络"""
        bandwidths = []
        current = base_bandwidth
        
        for _ in range(duration):
            # 随机游走
            change = random.gauss(0, 2.0)
            current = max(0.1, current + change)
            bandwidths.append(current)
            
        return bandwidths
    
    @staticmethod
    def generate_degrading_network(duration: int, start_bandwidth: float = 20.0) -> List[float]:
        """生成网络退化场景"""
        bandwidths = []
        current = start_bandwidth
        
        for i in range(duration):
            # 逐渐下降
            degradation = 0.05 * i
            current = max(0.5, start_bandwidth - degradation + random.gauss(0, 0.5))
            bandwidths.append(current)
            
        return bandwidths
    
    @staticmethod
    def generate_improving_network(duration: int, start_bandwidth: float = 2.0) -> List[float]:
        """生成网络改善场景"""
        bandwidths = []
        current = start_bandwidth
        
        for i in range(duration):
            # 逐渐改善
            improvement = 0.1 * i
            current = min(50.0, start_bandwidth + improvement + random.gauss(0, 0.5))
            bandwidths.append(current)
            
        return bandwidths

class OriginalNeurosurgeonSimulator:
    """原始Neurosurgeon模拟器"""
    
    def __init__(self):
        self.name = "Original Neurosurgeon"
        
    def make_decision(self, bandwidth: float, model_type: str) -> int:
        """原始Neurosurgeon的决策逻辑"""
        # 简化的原始决策逻辑
        if bandwidth < 1.0:
            return 15  # 更多边缘计算
        elif bandwidth > 50.0:
            return 0   # 全部云端
        else:
            return 8   # 平衡划分
            
    def get_performance(self, partition_point: int, bandwidth: float, 
                       model_type: str) -> Tuple[float, float]:
        """获取性能指标"""
        # 简化的性能模型
        base_latency = 100.0
        base_energy = 50.0
        
        if partition_point == 0:  # 全部云端
            latency = base_latency + 1000.0 / bandwidth
            energy = base_energy * 0.3
        elif partition_point >= 15:  # 大部分边缘
            latency = base_latency * 2.0
            energy = base_energy * 1.5
        else:  # 协同
            latency = base_latency + (20 - partition_point) * 5.0
            energy = base_energy * (0.5 + partition_point / 20.0 * 0.5)
            
        return latency, energy

class EnhancedNeurosurgeonSimulator:
    """增强版Neurosurgeon模拟器"""
    
    def __init__(self):
        self.name = "Enhanced Neurosurgeon"
        self.partitioner = AdaptivePartitioner()
        self.performance_simulator = PerformanceSimulator()
        
    def make_decision(self, bandwidth: float, server_load: float, 
                     edge_capability: float, battery_level: float, 
                     model_type: str) -> PartitionDecision:
        """增强版决策"""
        state = self.partitioner.update_system_state(
            bandwidth, server_load, edge_capability, battery_level
        )
        return self.partitioner.make_decision(state, model_type)
        
    def get_performance(self, decision: PartitionDecision, bandwidth: float, 
                       model_type: str) -> Tuple[float, float]:
        """获取性能指标"""
        return self.performance_simulator.simulate_performance(
            decision.partition_point, bandwidth, model_type
        )

class BenchmarkEvaluator:
    """基准测试评估器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.original_simulator = OriginalNeurosurgeonSimulator()
        self.enhanced_simulator = EnhancedNeurosurgeonSimulator()
        self.scenario_generator = NetworkScenarioGenerator()
        
    def run_single_scenario(self, scenario_name: str, model_type: str) -> Tuple[ExperimentResult, ExperimentResult]:
        """运行单个场景"""
        logger.info(f"运行场景: {scenario_name}, 模型: {model_type}")
        
        # 生成网络条件
        if scenario_name == "stable":
            bandwidths = self.scenario_generator.generate_stable_network(self.config.duration)
        elif scenario_name == "fluctuating":
            bandwidths = self.scenario_generator.generate_fluctuating_network(self.config.duration)
        elif scenario_name == "degrading":
            bandwidths = self.scenario_generator.generate_degrading_network(self.config.duration)
        elif scenario_name == "improving":
            bandwidths = self.scenario_generator.generate_improving_network(self.config.duration)
        else:
            bandwidths = self.scenario_generator.generate_stable_network(self.config.duration)
            
        # 运行原始Neurosurgeon
        original_results = self._run_original_strategy(bandwidths, model_type)
        
        # 运行增强版Neurosurgeon
        enhanced_results = self._run_enhanced_strategy(bandwidths, model_type)
        
        return original_results, enhanced_results
        
    def _run_original_strategy(self, bandwidths: List[float], model_type: str) -> ExperimentResult:
        """运行原始策略"""
        results = []
        latencies = []
        energies = []
        
        for i, bandwidth in enumerate(bandwidths):
            # 原始决策
            partition_point = self.original_simulator.make_decision(bandwidth, model_type)
            
            # 获取性能
            latency, energy = self.original_simulator.get_performance(partition_point, bandwidth, model_type)
            
            # 添加噪声模拟真实环境
            latency += random.gauss(0, latency * 0.1)
            energy += random.gauss(0, energy * 0.1)
            
            results.append({
                'timestamp': i,
                'bandwidth': bandwidth,
                'partition_point': partition_point,
                'latency': latency,
                'energy': energy,
                'strategy': 'original'
            })
            
            latencies.append(latency)
            energies.append(energy)
            
        return ExperimentResult(
            strategy_name="Original Neurosurgeon",
            total_requests=len(results),
            avg_latency=np.mean(latencies),
            avg_energy=np.mean(energies),
            latency_std=np.std(latencies),
            energy_std=np.std(energies),
            success_rate=1.0,
            adaptation_time=0.0,
            detailed_results=results
        )
        
    def _run_enhanced_strategy(self, bandwidths: List[float], model_type: str) -> ExperimentResult:
        """运行增强策略"""
        results = []
        latencies = []
        energies = []
        adaptation_times = []
        
        for i, bandwidth in enumerate(bandwidths):
            # 模拟系统状态
            server_load = random.uniform(0.3, 0.9)
            edge_capability = random.uniform(0.6, 1.0)
            battery_level = max(0.1, 1.0 - i * 0.001)  # 电池逐渐消耗
            
            # 增强决策
            start_time = time.time()
            decision = self.enhanced_simulator.make_decision(
                bandwidth, server_load, edge_capability, battery_level, model_type
            )
            decision_time = time.time() - start_time
            
            # 获取性能
            latency, energy = self.enhanced_simulator.get_performance(decision, bandwidth, model_type)
            
            # 添加噪声
            latency += random.gauss(0, latency * 0.1)
            energy += random.gauss(0, energy * 0.1)
            
            # 学习反馈
            if i > 0:  # 从第二次开始学习
                prev_state = SystemState(
                    bandwidth=bandwidths[i-1],
                    server_load=random.uniform(0.3, 0.9),
                    edge_capability=random.uniform(0.6, 1.0),
                    battery_level=max(0.1, 1.0 - (i-1) * 0.001),
                    timestamp=i-1
                )
                self.enhanced_simulator.partitioner.learn_from_feedback(
                    prev_state, results[-1]['decision'], 
                    results[-1]['latency'], results[-1]['energy']
                )
            
            results.append({
                'timestamp': i,
                'bandwidth': bandwidth,
                'server_load': server_load,
                'edge_capability': edge_capability,
                'battery_level': battery_level,
                'partition_point': decision.partition_point,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'latency': latency,
                'energy': energy,
                'decision_time': decision_time,
                'strategy': 'enhanced'
            })
            
            latencies.append(latency)
            energies.append(energy)
            adaptation_times.append(decision_time)
            
        return ExperimentResult(
            strategy_name="Enhanced Neurosurgeon",
            total_requests=len(results),
            avg_latency=np.mean(latencies),
            avg_energy=np.mean(energies),
            latency_std=np.std(latencies),
            energy_std=np.std(energies),
            success_rate=1.0,
            adaptation_time=np.mean(adaptation_times),
            detailed_results=results
        )
        
    def run_comprehensive_evaluation(self) -> Dict[str, Dict[str, Tuple[ExperimentResult, ExperimentResult]]]:
        """运行综合评估"""
        logger.info("开始综合评估...")
        
        all_results = {}
        
        for scenario in self.config.network_scenarios:
            all_results[scenario] = {}
            
            for model_type in self.config.model_types:
                logger.info(f"评估场景: {scenario}, 模型: {model_type}")
                
                original_result, enhanced_result = self.run_single_scenario(scenario, model_type)
                all_results[scenario][model_type] = (original_result, enhanced_result)
                
        return all_results
        
    def generate_performance_report(self, results: Dict) -> str:
        """生成性能报告"""
        report = []
        report.append("=" * 80)
        report.append("Enhanced Neurosurgeon 性能评估报告")
        report.append("=" * 80)
        report.append("")
        
        # 总体统计
        total_improvements = []
        
        for scenario, models in results.items():
            report.append(f"场景: {scenario.upper()}")
            report.append("-" * 40)
            
            for model_type, (original, enhanced) in models.items():
                latency_improvement = (original.avg_latency - enhanced.avg_latency) / original.avg_latency * 100
                energy_improvement = (original.avg_energy - enhanced.avg_energy) / original.avg_energy * 100
                stability_improvement = (original.latency_std - enhanced.latency_std) / original.latency_std * 100
                
                report.append(f"  模型: {model_type}")
                report.append(f"    延迟改善: {latency_improvement:.2f}%")
                report.append(f"    能耗改善: {energy_improvement:.2f}%")
                report.append(f"    稳定性改善: {stability_improvement:.2f}%")
                report.append(f"    自适应时间: {enhanced.adaptation_time*1000:.2f}ms")
                report.append("")
                
                total_improvements.append({
                    'scenario': scenario,
                    'model': model_type,
                    'latency_improvement': latency_improvement,
                    'energy_improvement': energy_improvement,
                    'stability_improvement': stability_improvement
                })
                
        # 平均改善
        avg_latency_improvement = np.mean([imp['latency_improvement'] for imp in total_improvements])
        avg_energy_improvement = np.mean([imp['energy_improvement'] for imp in total_improvements])
        avg_stability_improvement = np.mean([imp['stability_improvement'] for imp in total_improvements])
        
        report.append("总体改善:")
        report.append(f"  平均延迟改善: {avg_latency_improvement:.2f}%")
        report.append(f"  平均能耗改善: {avg_energy_improvement:.2f}%")
        report.append(f"  平均稳定性改善: {avg_stability_improvement:.2f}%")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def save_results(self, results: Dict, output_dir: str = "results"):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        for scenario, models in results.items():
            for model_type, (original, enhanced) in models.items():
                filename = f"{scenario}_{model_type}_results.json"
                filepath = os.path.join(output_dir, filename)
                
                data = {
                    'scenario': scenario,
                    'model_type': model_type,
                    'original': {
                        'strategy_name': original.strategy_name,
                        'avg_latency': original.avg_latency,
                        'avg_energy': original.avg_energy,
                        'latency_std': original.latency_std,
                        'energy_std': original.energy_std,
                        'detailed_results': original.detailed_results
                    },
                    'enhanced': {
                        'strategy_name': enhanced.strategy_name,
                        'avg_latency': enhanced.avg_latency,
                        'avg_energy': enhanced.avg_energy,
                        'latency_std': enhanced.latency_std,
                        'energy_std': enhanced.energy_std,
                        'adaptation_time': enhanced.adaptation_time,
                        'detailed_results': enhanced.detailed_results
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
        logger.info(f"结果已保存到: {output_dir}")

class VisualizationGenerator:
    """可视化生成器"""
    
    @staticmethod
    def plot_performance_comparison(results: Dict, output_dir: str = "plots"):
        """绘制性能对比图"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        for scenario, models in results.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'场景: {scenario.upper()}', fontsize=16)
            
            model_names = list(models.keys())
            original_latencies = []
            enhanced_latencies = []
            original_energies = []
            enhanced_energies = []
            original_stds = []
            enhanced_stds = []
            
            for model_type, (original, enhanced) in models.items():
                original_latencies.append(original.avg_latency)
                enhanced_latencies.append(enhanced.avg_latency)
                original_energies.append(original.avg_energy)
                enhanced_energies.append(enhanced.avg_energy)
                original_stds.append(original.latency_std)
                enhanced_stds.append(enhanced.latency_std)
                
            # 延迟对比
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, original_latencies, width, label='原始Neurosurgeon', alpha=0.8)
            axes[0, 0].bar(x + width/2, enhanced_latencies, width, label='增强版Neurosurgeon', alpha=0.8)
            axes[0, 0].set_xlabel('模型类型')
            axes[0, 0].set_ylabel('平均延迟 (ms)')
            axes[0, 0].set_title('延迟对比')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_names)
            axes[0, 0].legend()
            
            # 能耗对比
            axes[0, 1].bar(x - width/2, original_energies, width, label='原始Neurosurgeon', alpha=0.8)
            axes[0, 1].bar(x + width/2, enhanced_energies, width, label='增强版Neurosurgeon', alpha=0.8)
            axes[0, 1].set_xlabel('模型类型')
            axes[0, 1].set_ylabel('平均能耗 (mJ)')
            axes[0, 1].set_title('能耗对比')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(model_names)
            axes[0, 1].legend()
            
            # 稳定性对比
            axes[1, 0].bar(x - width/2, original_stds, width, label='原始Neurosurgeon', alpha=0.8)
            axes[1, 0].bar(x + width/2, enhanced_stds, width, label='增强版Neurosurgeon', alpha=0.8)
            axes[1, 0].set_xlabel('模型类型')
            axes[1, 0].set_ylabel('延迟标准差 (ms)')
            axes[1, 0].set_title('稳定性对比')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(model_names)
            axes[1, 0].legend()
            
            # 改善百分比
            latency_improvements = [(o - e) / o * 100 for o, e in zip(original_latencies, enhanced_latencies)]
            energy_improvements = [(o - e) / o * 100 for o, e in zip(original_energies, enhanced_energies)]
            
            axes[1, 1].bar(x - width/2, latency_improvements, width, label='延迟改善', alpha=0.8)
            axes[1, 1].bar(x + width/2, energy_improvements, width, label='能耗改善', alpha=0.8)
            axes[1, 1].set_xlabel('模型类型')
            axes[1, 1].set_ylabel('改善百分比 (%)')
            axes[1, 1].set_title('性能改善')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(model_names)
            axes[1, 1].legend()
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{scenario}_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
    @staticmethod
    def plot_adaptation_curve(results: Dict, output_dir: str = "plots"):
        """绘制自适应曲线"""
        os.makedirs(output_dir, exist_ok=True)
        
        for scenario, models in results.items():
            for model_type, (original, enhanced) in models.items():
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # 提取时间序列数据
                timestamps = [r['timestamp'] for r in enhanced.detailed_results]
                bandwidths = [r['bandwidth'] for r in enhanced.detailed_results]
                original_latencies = [r['latency'] for r in original.detailed_results]
                enhanced_latencies = [r['latency'] for r in enhanced.detailed_results]
                partition_points = [r['partition_point'] for r in enhanced.detailed_results]
                
                # 网络带宽变化
                ax1.plot(timestamps, bandwidths, 'b-', label='网络带宽', linewidth=2)
                ax1.set_ylabel('带宽 (MB/s)')
                ax1.set_title(f'{scenario.upper()} - {model_type.upper()} 自适应过程')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 延迟对比和划分点
                ax2_twin = ax2.twinx()
                
                ax2.plot(timestamps, original_latencies, 'r--', label='原始Neurosurgeon', linewidth=2)
                ax2.plot(timestamps, enhanced_latencies, 'g-', label='增强版Neurosurgeon', linewidth=2)
                ax2_twin.plot(timestamps, partition_points, 'orange', label='划分点', linewidth=2, alpha=0.7)
                
                ax2.set_xlabel('时间 (秒)')
                ax2.set_ylabel('延迟 (ms)')
                ax2_twin.set_ylabel('划分点')
                ax2.legend(loc='upper left')
                ax2_twin.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{scenario}_{model_type}_adaptation.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()

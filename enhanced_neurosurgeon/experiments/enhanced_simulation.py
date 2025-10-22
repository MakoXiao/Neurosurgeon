"""
增强的仿真实验系统
Enhanced Simulation System

基于状态-动作-奖励框架的深度优化实验
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
from pathlib import Path

# 导入增强的RL智能体
from ..core.enhanced_rl_agent import (
    EnhancedSystemState, EnhancedAction, PerformanceMetrics,
    EnhancedRLAgent, EnhancedRewardFunction
)

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """仿真配置"""
    duration: int = 1000  # 仿真时长 (步数)
    models: List[str] = None  # 模型类型
    network_scenarios: List[str] = None  # 网络场景
    learning_episodes: int = 100  # 学习轮数
    
    def __post_init__(self):
        if self.models is None:
            self.models = ["mobilenet", "vggnet", "alexnet", "lenet"]
        if self.network_scenarios is None:
            self.network_scenarios = ["stable", "fluctuating", "degrading", "improving"]

class CloudEdgeSimulator:
    """云边协同仿真器"""
    
    def __init__(self):
        # 云服务器配置
        self.cloud_config = {
            'compute_power': 10.0,  # 计算能力倍数
            'memory_bandwidth': 100.0,  # 内存带宽 (GB/s)
            'energy_efficiency': 0.8,  # 能耗效率
            'cost_per_compute': 0.01  # 每单位计算成本
        }
        
        # 边缘设备配置
        self.edge_config = {
            'compute_power': 1.0,  # 计算能力倍数
            'memory_bandwidth': 10.0,  # 内存带宽 (GB/s)
            'energy_efficiency': 1.2,  # 能耗效率
            'battery_capacity': 100.0  # 电池容量 (mAh)
        }
        
        # 模型性能配置
        self.model_profiles = {
            'mobilenet': {
                'layers': 20,
                'base_latency': 80.0,
                'base_energy': 40.0,
                'base_accuracy': 0.95,
                'layer_complexity': [1.0, 1.2, 1.1, 1.3, 1.0, 1.2, 1.1, 1.3, 1.0, 1.2, 
                                   1.1, 1.3, 1.0, 1.2, 1.1, 1.3, 1.0, 1.2, 1.1, 1.3],
                'data_sizes': [0.1, 0.2, 0.15, 0.25, 0.1, 0.2, 0.15, 0.25, 0.1, 0.2,
                             0.15, 0.25, 0.1, 0.2, 0.15, 0.25, 0.1, 0.2, 0.15, 0.25]
            },
            'vggnet': {
                'layers': 20,
                'base_latency': 200.0,
                'base_energy': 100.0,
                'base_accuracy': 0.98,
                'layer_complexity': [2.0, 2.5, 2.2, 2.8, 2.0, 2.5, 2.2, 2.8, 2.0, 2.5,
                                   2.2, 2.8, 2.0, 2.5, 2.2, 2.8, 2.0, 2.5, 2.2, 2.8],
                'data_sizes': [0.3, 0.4, 0.35, 0.45, 0.3, 0.4, 0.35, 0.45, 0.3, 0.4,
                             0.35, 0.45, 0.3, 0.4, 0.35, 0.45, 0.3, 0.4, 0.35, 0.45]
            },
            'alexnet': {
                'layers': 20,
                'base_latency': 120.0,
                'base_energy': 60.0,
                'base_accuracy': 0.96,
                'layer_complexity': [1.5, 1.8, 1.6, 1.9, 1.5, 1.8, 1.6, 1.9, 1.5, 1.8,
                                   1.6, 1.9, 1.5, 1.8, 1.6, 1.9, 1.5, 1.8, 1.6, 1.9],
                'data_sizes': [0.2, 0.3, 0.25, 0.35, 0.2, 0.3, 0.25, 0.35, 0.2, 0.3,
                             0.25, 0.35, 0.2, 0.3, 0.25, 0.35, 0.2, 0.3, 0.25, 0.35]
            },
            'lenet': {
                'layers': 20,
                'base_latency': 50.0,
                'base_energy': 25.0,
                'base_accuracy': 0.92,
                'layer_complexity': [0.8, 1.0, 0.9, 1.1, 0.8, 1.0, 0.9, 1.1, 0.8, 1.0,
                                   0.9, 1.1, 0.8, 1.0, 0.9, 1.1, 0.8, 1.0, 0.9, 1.1],
                'data_sizes': [0.05, 0.1, 0.08, 0.12, 0.05, 0.1, 0.08, 0.12, 0.05, 0.1,
                             0.08, 0.12, 0.05, 0.1, 0.08, 0.12, 0.05, 0.1, 0.08, 0.12]
            }
        }
    
    def simulate_performance(self, state: EnhancedSystemState, action: EnhancedAction, 
                           model_type: str) -> PerformanceMetrics:
        """仿真性能"""
        profile = self.model_profiles.get(model_type, self.model_profiles['mobilenet'])
        
        # 计算边缘端性能
        edge_latency, edge_energy = self._calculate_edge_performance(
            action, profile, state, self.edge_config
        )
        
        # 计算传输性能
        transmission_latency, transmission_energy = self._calculate_transmission_performance(
            action, profile, state
        )
        
        # 计算云端性能
        cloud_latency, cloud_energy = self._calculate_cloud_performance(
            action, profile, state, self.cloud_config
        )
        
        # 计算准确性影响
        accuracy = self._calculate_accuracy_impact(action, profile)
        
        # 计算吞吐量
        throughput = self._calculate_throughput(action, state)
        
        # 计算资源利用率
        resource_utilization = self._calculate_resource_utilization(action, state)
        
        # 计算成本
        cost = self._calculate_cost(action, state)
        
        # 总性能
        total_latency = edge_latency + transmission_latency + cloud_latency
        total_energy = edge_energy + transmission_energy + cloud_energy
        
        return PerformanceMetrics(
            latency=total_latency,
            energy=total_energy,
            accuracy=accuracy,
            throughput=throughput,
            resource_utilization=resource_utilization,
            cost=cost
        )
    
    def _calculate_edge_performance(self, action: EnhancedAction, profile: Dict, 
                                  state: EnhancedSystemState, config: Dict) -> Tuple[float, float]:
        """计算边缘端性能"""
        if action.partition_point == 0:
            return 0.0, 0.0
        
        # 计算压缩和量化影响
        compression_factor = action.compression_ratio
        quantization_factor = 32.0 / action.quantization_bits
        
        # 计算剪枝影响
        pruning_factor = 1.0 - action.model_pruning_ratio
        
        # 边缘端层数
        edge_layers = min(action.partition_point, len(profile["layer_complexity"]))
        complexity_sum = sum(profile["layer_complexity"][:edge_layers])
        
        # 基础性能
        base_latency = profile["base_latency"]
        base_energy = profile["base_energy"]
        
        # 应用压缩和量化
        adjusted_complexity = complexity_sum * compression_factor * quantization_factor * pruning_factor
        
        # 计算延迟和能耗
        latency = (base_latency * adjusted_complexity / len(profile["layer_complexity"]) 
                  / config["compute_power"] / state.edge_capability)
        
        energy = (base_energy * adjusted_complexity / len(profile["layer_complexity"]) 
                 * config["energy_efficiency"] / state.edge_capability)
        
        # 批处理影响
        batch_factor = 1.0 / np.sqrt(action.batch_size)
        latency *= batch_factor
        energy *= batch_factor
        
        return latency, energy
    
    def _calculate_transmission_performance(self, action: EnhancedAction, profile: Dict, 
                                          state: EnhancedSystemState) -> Tuple[float, float]:
        """计算传输性能"""
        if action.partition_point == 0:
            # 传输原始输入
            data_size = 0.1  # MB
        elif action.partition_point >= len(profile["data_sizes"]):
            # 传输最终输出
            data_size = 0.01  # MB
        else:
            # 传输中间结果
            data_size = profile["data_sizes"][action.partition_point - 1]
        
        # 应用压缩
        compressed_data_size = data_size * action.compression_ratio
        
        # 传输延迟 (ms)
        transmission_latency = (compressed_data_size * 8) / state.bandwidth
        
        # 传输能耗 (mJ)
        transmission_energy = compressed_data_size * 0.1 * (1.0 + state.packet_loss)
        
        return transmission_latency, transmission_energy
    
    def _calculate_cloud_performance(self, action: EnhancedAction, profile: Dict, 
                                   state: EnhancedSystemState, config: Dict) -> Tuple[float, float]:
        """计算云端性能"""
        if action.partition_point >= len(profile["layer_complexity"]):
            return 0.0, 0.0
        
        # 云端层数
        cloud_layers = len(profile["layer_complexity"]) - action.partition_point
        complexity_sum = sum(profile["layer_complexity"][action.partition_point:])
        
        # 基础性能
        base_latency = profile["base_latency"]
        base_energy = profile["base_energy"]
        
        # 计算延迟和能耗
        latency = (base_latency * complexity_sum / len(profile["layer_complexity"]) 
                  / config["compute_power"] / (1.0 - state.server_load))
        
        energy = (base_energy * complexity_sum / len(profile["layer_complexity"]) 
                 * config["energy_efficiency"])
        
        # 并行处理影响
        parallel_factor = 1.0 / action.parallel_degree
        latency *= parallel_factor
        energy *= parallel_factor
        
        return latency, energy
    
    def _calculate_accuracy_impact(self, action: EnhancedAction, profile: Dict) -> float:
        """计算准确性影响"""
        base_accuracy = profile["base_accuracy"]
        
        # 压缩对准确性的影响
        compression_penalty = (1.0 - action.compression_ratio) * 0.1
        
        # 量化对准确性的影响
        quantization_penalty = (32.0 - action.quantization_bits) / 32.0 * 0.05
        
        # 剪枝对准确性的影响
        pruning_penalty = action.model_pruning_ratio * 0.15
        
        accuracy = base_accuracy - compression_penalty - quantization_penalty - pruning_penalty
        return max(0.5, min(1.0, accuracy))
    
    def _calculate_throughput(self, action: EnhancedAction, state: EnhancedSystemState) -> float:
        """计算吞吐量"""
        # 基于批处理大小和并行度
        base_throughput = 10.0  # requests/s
        
        # 批处理影响
        batch_factor = np.sqrt(action.batch_size)
        
        # 并行度影响
        parallel_factor = action.parallel_degree
        
        # 网络影响
        network_factor = min(1.0, state.bandwidth / 10.0)
        
        throughput = base_throughput * batch_factor * parallel_factor * network_factor
        return throughput
    
    def _calculate_resource_utilization(self, action: EnhancedAction, state: EnhancedSystemState) -> float:
        """计算资源利用率"""
        # 基于划分点和设备能力
        edge_utilization = min(1.0, action.partition_point / 20.0 * state.edge_capability)
        cloud_utilization = min(1.0, (20 - action.partition_point) / 20.0 * (1.0 - state.server_load))
        
        return (edge_utilization + cloud_utilization) / 2.0
    
    def _calculate_cost(self, action: EnhancedAction, state: EnhancedSystemState) -> float:
        """计算成本"""
        # 云端计算成本
        cloud_cost = (20 - action.partition_point) / 20.0 * 0.1
        
        # 传输成本
        transmission_cost = action.compression_ratio * 0.05
        
        # 能耗成本
        energy_cost = (1.0 - state.battery_level) * 0.02
        
        return cloud_cost + transmission_cost + energy_cost

class NetworkScenarioGenerator:
    """网络场景生成器"""
    
    @staticmethod
    def generate_enhanced_state(scenario: str, step: int, base_bandwidth: float = 10.0) -> EnhancedSystemState:
        """生成增强的系统状态"""
        current_time = time.time()
        
        if scenario == "stable":
            bandwidth = base_bandwidth + np.random.normal(0, 0.5)
            bandwidth_variance = 0.1
            latency = 5.0 + np.random.normal(0, 1.0)
            packet_loss = 0.001 + np.random.normal(0, 0.0005)
        elif scenario == "fluctuating":
            bandwidth = base_bandwidth + 5.0 * np.sin(step * 0.1) + np.random.normal(0, 2.0)
            bandwidth_variance = 0.5
            latency = 5.0 + 3.0 * np.sin(step * 0.2) + np.random.normal(0, 2.0)
            packet_loss = 0.01 + 0.005 * np.sin(step * 0.3) + np.random.normal(0, 0.002)
        elif scenario == "degrading":
            degradation = step * 0.01
            bandwidth = max(0.5, base_bandwidth - degradation + np.random.normal(0, 1.0))
            bandwidth_variance = 0.3
            latency = 5.0 + degradation * 2.0 + np.random.normal(0, 1.5)
            packet_loss = 0.001 + degradation * 0.01 + np.random.normal(0, 0.001)
        elif scenario == "improving":
            improvement = step * 0.02
            bandwidth = min(50.0, base_bandwidth + improvement + np.random.normal(0, 1.0))
            bandwidth_variance = 0.2
            latency = max(1.0, 5.0 - improvement * 0.5 + np.random.normal(0, 1.0))
            packet_loss = max(0.0001, 0.01 - improvement * 0.0001 + np.random.normal(0, 0.0005))
        else:
            bandwidth = base_bandwidth
            bandwidth_variance = 0.1
            latency = 5.0
            packet_loss = 0.001
        
        # 其他状态参数
        server_load = 0.3 + 0.4 * np.random.random()
        server_memory = 0.4 + 0.3 * np.random.random()
        edge_capability = 0.6 + 0.3 * np.random.random()
        edge_memory = 0.3 + 0.4 * np.random.random()
        battery_level = max(0.1, 1.0 - step * 0.0001)
        device_temperature = 0.3 + 0.4 * np.random.random()
        cpu_usage = 0.2 + 0.6 * np.random.random()
        
        task_priority = np.random.random()
        task_complexity = np.random.random()
        data_size = 0.1 + 0.9 * np.random.random()
        
        time_of_day = (current_time % 86400) / 86400 * 24  # 一天中的时间
        
        return EnhancedSystemState(
            bandwidth=max(0.1, bandwidth),
            bandwidth_variance=bandwidth_variance,
            latency=max(1.0, latency),
            packet_loss=max(0.0, min(0.1, packet_loss)),
            server_load=server_load,
            server_memory=server_memory,
            edge_capability=edge_capability,
            edge_memory=edge_memory,
            battery_level=battery_level,
            device_temperature=device_temperature,
            cpu_usage=cpu_usage,
            task_priority=task_priority,
            task_complexity=task_complexity,
            data_size=data_size,
            timestamp=current_time,
            time_of_day=time_of_day,
            recent_performance=0.8 + 0.2 * np.random.random(),
            adaptation_frequency=0.1 + 0.9 * np.random.random()
        )

class EnhancedSimulationExperiment:
    """增强的仿真实验"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rl_agent = EnhancedRLAgent()
        self.simulator = CloudEdgeSimulator()
        self.scenario_generator = NetworkScenarioGenerator()
        
        # 实验结果
        self.results = {
            'states': [],
            'actions': [],
            'rewards': [],
            'metrics': [],
            'learning_curve': []
        }
    
    def run_simulation(self) -> Dict:
        """运行仿真实验"""
        logger.info("开始增强的强化学习仿真实验...")
        
        for episode in range(self.config.learning_episodes):
            episode_rewards = []
            
            for step in range(self.config.duration):
                # 生成系统状态
                scenario = np.random.choice(self.config.network_scenarios)
                model_type = np.random.choice(self.config.models)
                
                state = self.scenario_generator.generate_enhanced_state(scenario, step)
                
                # 获取动作
                action = self.rl_agent.get_action(state)
                
                # 仿真性能
                metrics = self.simulator.simulate_performance(state, action, model_type)
                
                # 计算奖励
                reward = self.rl_agent.calculate_reward(state, action, metrics)
                
                # 生成下一个状态
                next_state = self.scenario_generator.generate_enhanced_state(scenario, step + 1)
                
                # 存储经验
                self.rl_agent.remember(state, action, reward, next_state, False)
                
                # 训练
                if step % 10 == 0:
                    self.rl_agent.replay()
                
                # 记录结果
                self.results['states'].append(state)
                self.results['actions'].append(action)
                self.results['rewards'].append(reward)
                self.results['metrics'].append(metrics)
                
                episode_rewards.append(reward)
            
            # 记录学习曲线
            avg_reward = np.mean(episode_rewards)
            self.results['learning_curve'].append(avg_reward)
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}, Average Reward: {avg_reward:.3f}")
        
        logger.info("仿真实验完成!")
        return self.results
    
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        if not self.results['rewards']:
            return "无实验结果"
        
        rewards = np.array(self.results['rewards'])
        metrics = self.results['metrics']
        
        # 计算统计信息
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        
        # 计算性能指标
        latencies = [m.latency for m in metrics]
        energies = [m.energy for m in metrics]
        accuracies = [m.accuracy for m in metrics]
        throughputs = [m.throughput for m in metrics]
        
        report = f"""
增强强化学习仿真实验结果报告
================================

总体统计:
- 平均奖励: {avg_reward:.3f} ± {std_reward:.3f}
- 最大奖励: {max_reward:.3f}
- 最小奖励: {min_reward:.3f}

性能指标:
- 平均延迟: {np.mean(latencies):.2f} ms
- 平均能耗: {np.mean(energies):.2f} mJ
- 平均准确性: {np.mean(accuracies):.3f}
- 平均吞吐量: {np.mean(throughputs):.2f} requests/s

学习效果:
- 初始平均奖励: {np.mean(self.results['learning_curve'][:10]):.3f}
- 最终平均奖励: {np.mean(self.results['learning_curve'][-10:]):.3f}
- 学习改善: {np.mean(self.results['learning_curve'][-10:]) - np.mean(self.results['learning_curve'][:10]):.3f}
        """
        
        return report
    
    def save_results(self, output_dir: str = "enhanced_results"):
        """保存实验结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        results_data = {
            'learning_curve': self.results['learning_curve'],
            'final_metrics': {
                'avg_latency': np.mean([m.latency for m in self.results['metrics']]),
                'avg_energy': np.mean([m.energy for m in self.results['metrics']]),
                'avg_accuracy': np.mean([m.accuracy for m in self.results['metrics']]),
                'avg_throughput': np.mean([m.throughput for m in self.results['metrics']]),
                'avg_reward': np.mean(self.results['rewards'])
            }
        }
        
        with open(os.path.join(output_dir, 'simulation_results.json'), 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # 保存模型
        self.rl_agent.save_model(os.path.join(output_dir, 'enhanced_rl_model.pth'))
        
        logger.info(f"结果已保存到: {output_dir}")

def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建实验配置
    config = SimulationConfig(
        duration=500,
        learning_episodes=50
    )
    
    # 创建实验
    experiment = EnhancedSimulationExperiment(config)
    
    # 运行仿真
    results = experiment.run_simulation()
    
    # 生成报告
    report = experiment.generate_performance_report()
    print(report)
    
    # 保存结果
    experiment.save_results()
    
    return results

if __name__ == "__main__":
    main()

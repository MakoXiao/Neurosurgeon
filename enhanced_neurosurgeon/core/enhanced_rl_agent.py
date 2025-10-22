"""
增强的强化学习智能体
Enhanced Reinforcement Learning Agent

基于状态-动作-奖励框架的深度优化，支持动态分割和压缩率调整
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSystemState:
    """增强的系统状态 - 包含更多环境因子"""
    # 网络状态
    bandwidth: float  # 网络带宽 (MB/s)
    bandwidth_variance: float  # 带宽方差
    latency: float  # 网络延迟 (ms)
    packet_loss: float  # 丢包率 (0-1)
    
    # 计算资源状态
    server_load: float  # 服务器负载 (0-1)
    server_memory: float  # 服务器内存使用率 (0-1)
    edge_capability: float  # 边缘设备计算能力 (0-1)
    edge_memory: float  # 边缘设备内存使用率 (0-1)
    
    # 设备状态
    battery_level: float  # 电池电量 (0-1)
    device_temperature: float  # 设备温度 (0-1)
    cpu_usage: float  # CPU使用率 (0-1)
    
    # 任务状态
    task_priority: float  # 任务优先级 (0-1)
    task_complexity: float  # 任务复杂度 (0-1)
    data_size: float  # 数据大小 (MB)
    
    # 时间信息
    timestamp: float
    time_of_day: float  # 一天中的时间 (0-24)
    
    # 历史信息
    recent_performance: float  # 最近性能表现
    adaptation_frequency: float  # 自适应频率

@dataclass
class EnhancedAction:
    """增强的动作空间 - 包含动态分割和压缩率调整"""
    partition_point: int  # 划分点 (0-20)
    compression_ratio: float  # 压缩率 (0.1-1.0)
    quantization_bits: int  # 量化位数 (4, 8, 16, 32)
    model_pruning_ratio: float  # 模型剪枝比例 (0-0.5)
    batch_size: int  # 批处理大小 (1, 2, 4, 8, 16)
    parallel_degree: int  # 并行度 (1, 2, 4, 8)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    latency: float  # 延迟 (ms)
    energy: float  # 能耗 (mJ)
    accuracy: float  # 准确性 (0-1)
    throughput: float  # 吞吐量 (requests/s)
    resource_utilization: float  # 资源利用率 (0-1)
    cost: float  # 成本

class EnhancedRewardFunction:
    """增强的奖励函数"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'latency': 0.3,
            'energy': 0.2,
            'accuracy': 0.25,
            'throughput': 0.15,
            'resource_utilization': 0.1
        }
        
    def calculate_reward(self, state: EnhancedSystemState, action: EnhancedAction, 
                        metrics: PerformanceMetrics) -> float:
        """计算奖励值"""
        
        # 基础性能奖励
        latency_reward = self._calculate_latency_reward(metrics.latency, state.bandwidth)
        energy_reward = self._calculate_energy_reward(metrics.energy, state.battery_level)
        accuracy_reward = metrics.accuracy
        throughput_reward = self._calculate_throughput_reward(metrics.throughput, state.server_load)
        resource_reward = self._calculate_resource_reward(metrics.resource_utilization, state.edge_capability)
        
        # 自适应奖励
        adaptation_reward = self._calculate_adaptation_reward(state, action)
        
        # 稳定性奖励
        stability_reward = self._calculate_stability_reward(state, metrics)
        
        # 综合奖励
        total_reward = (
            self.weights['latency'] * latency_reward +
            self.weights['energy'] * energy_reward +
            self.weights['accuracy'] * accuracy_reward +
            self.weights['throughput'] * throughput_reward +
            self.weights['resource_utilization'] * resource_reward +
            0.1 * adaptation_reward +
            0.1 * stability_reward
        )
        
        return total_reward
    
    def _calculate_latency_reward(self, latency: float, bandwidth: float) -> float:
        """计算延迟奖励"""
        # 基于带宽的延迟期望
        expected_latency = 100.0 / (bandwidth + 1.0)
        if latency <= expected_latency:
            return 1.0
        else:
            return max(0.0, 1.0 - (latency - expected_latency) / expected_latency)
    
    def _calculate_energy_reward(self, energy: float, battery_level: float) -> float:
        """计算能耗奖励"""
        # 电池电量越低，能耗权重越高
        energy_weight = 1.0 + (1.0 - battery_level) * 2.0
        expected_energy = 50.0 / energy_weight
        
        if energy <= expected_energy:
            return 1.0
        else:
            return max(0.0, 1.0 - (energy - expected_energy) / expected_energy)
    
    def _calculate_throughput_reward(self, throughput: float, server_load: float) -> float:
        """计算吞吐量奖励"""
        # 服务器负载越高，吞吐量要求越高
        expected_throughput = 10.0 * (1.0 + server_load)
        
        if throughput >= expected_throughput:
            return 1.0
        else:
            return max(0.0, throughput / expected_throughput)
    
    def _calculate_resource_reward(self, utilization: float, capability: float) -> float:
        """计算资源利用率奖励"""
        # 理想利用率在0.7-0.9之间
        if 0.7 <= utilization <= 0.9:
            return 1.0
        elif utilization < 0.7:
            return utilization / 0.7
        else:
            return max(0.0, 1.0 - (utilization - 0.9) / 0.1)
    
    def _calculate_adaptation_reward(self, state: EnhancedSystemState, action: EnhancedAction) -> float:
        """计算自适应奖励"""
        # 根据环境变化调整策略的奖励
        bandwidth_change = abs(state.bandwidth_variance)
        load_change = abs(state.server_load - 0.5)
        
        # 如果环境变化大，鼓励更多调整
        if bandwidth_change > 0.5 or load_change > 0.3:
            return 0.5
        else:
            return 0.2
    
    def _calculate_stability_reward(self, state: EnhancedSystemState, metrics: PerformanceMetrics) -> float:
        """计算稳定性奖励"""
        # 基于历史性能的稳定性
        if state.recent_performance > 0.8:
            return 0.3
        elif state.recent_performance > 0.6:
            return 0.2
        else:
            return 0.1

class DynamicPartitionStrategy:
    """动态分割策略"""
    
    def __init__(self):
        self.partition_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
        
    def get_optimal_partition(self, state: EnhancedSystemState, model_type: str) -> int:
        """获取最优划分点"""
        # 基于网络条件的动态调整
        if state.bandwidth < 1.0:  # 低带宽
            base_partition = 15
        elif state.bandwidth > 50.0:  # 高带宽
            base_partition = 0
        else:  # 中等带宽
            base_partition = 8
            
        # 基于服务器负载调整
        if state.server_load > 0.8:
            base_partition = min(20, base_partition + 3)
        elif state.server_load < 0.3:
            base_partition = max(0, base_partition - 2)
            
        # 基于电池电量调整
        if state.battery_level < 0.3:
            base_partition = max(0, base_partition - 2)
            
        # 基于任务复杂度调整
        if state.task_complexity > 0.8:
            base_partition = min(20, base_partition + 2)
            
        return max(0, min(20, base_partition))

class CompressionOptimizer:
    """压缩优化器"""
    
    def __init__(self):
        self.compression_profiles = {
            'aggressive': {'ratio': 0.1, 'bits': 4, 'pruning': 0.3},
            'balanced': {'ratio': 0.3, 'bits': 8, 'pruning': 0.1},
            'conservative': {'ratio': 0.6, 'bits': 16, 'pruning': 0.05},
            'none': {'ratio': 1.0, 'bits': 32, 'pruning': 0.0}
        }
        
    def get_optimal_compression(self, state: EnhancedSystemState) -> Dict[str, float]:
        """获取最优压缩策略"""
        # 基于网络带宽选择压缩策略
        if state.bandwidth < 2.0:
            return self.compression_profiles['aggressive']
        elif state.bandwidth < 10.0:
            return self.compression_profiles['balanced']
        elif state.bandwidth < 30.0:
            return self.compression_profiles['conservative']
        else:
            return self.compression_profiles['none']

class EnhancedRLAgent:
    """增强的强化学习智能体"""
    
    def __init__(self, state_dim: int = 15, action_dim: int = 1000, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # 增强的Q网络 - 使用更深的网络结构
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 目标网络
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # 复制参数到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 强化学习参数
        self.epsilon = 0.3  # 初始探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95  # 折扣因子
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_frequency = 10
        self.step_count = 0
        
        # 辅助模块
        self.reward_function = EnhancedRewardFunction()
        self.partition_strategy = DynamicPartitionStrategy()
        self.compression_optimizer = CompressionOptimizer()
        
    def state_to_tensor(self, state: EnhancedSystemState) -> torch.Tensor:
        """将状态转换为张量"""
        return torch.FloatTensor([
            state.bandwidth,
            state.bandwidth_variance,
            state.latency,
            state.packet_loss,
            state.server_load,
            state.server_memory,
            state.edge_capability,
            state.edge_memory,
            state.battery_level,
            state.device_temperature,
            state.cpu_usage,
            state.task_priority,
            state.task_complexity,
            state.data_size,
            state.time_of_day
        ]).unsqueeze(0)
    
    def action_to_enhanced(self, action_idx: int) -> EnhancedAction:
        """将动作索引转换为增强动作"""
        # 动作空间分解
        partition_point = action_idx % 21
        compression_idx = (action_idx // 21) % 4
        quantization_idx = (action_idx // 84) % 4
        pruning_idx = (action_idx // 336) % 5
        batch_idx = (action_idx // 1680) % 5
        parallel_idx = (action_idx // 8400) % 4
        
        # 映射到实际值
        compression_ratios = [0.1, 0.3, 0.6, 1.0]
        quantization_bits = [4, 8, 16, 32]
        pruning_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
        batch_sizes = [1, 2, 4, 8, 16]
        parallel_degrees = [1, 2, 4, 8]
        
        return EnhancedAction(
            partition_point=partition_point,
            compression_ratio=compression_ratios[compression_idx],
            quantization_bits=quantization_bits[quantization_idx],
            model_pruning_ratio=pruning_ratios[pruning_idx],
            batch_size=batch_sizes[batch_idx],
            parallel_degree=parallel_degrees[parallel_idx]
        )
    
    def get_action(self, state: EnhancedSystemState) -> EnhancedAction:
        """获取动作"""
        if np.random.random() < self.epsilon:
            # 随机探索
            action_idx = np.random.randint(0, self.action_dim)
        else:
            # 贪婪选择
            state_tensor = self.state_to_tensor(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = q_values.argmax(dim=1).item()
        
        return self.action_to_enhanced(action_idx)
    
    def remember(self, state: EnhancedSystemState, action: EnhancedAction, 
                reward: float, next_state: EnhancedSystemState, done: bool):
        """存储经验"""
        # 将动作转换为索引
        action_idx = self.enhanced_to_action_index(action)
        
        self.memory.append((
            self.state_to_tensor(state).squeeze(),
            action_idx,
            reward,
            self.state_to_tensor(next_state).squeeze(),
            done
        ))
    
    def enhanced_to_action_index(self, action: EnhancedAction) -> int:
        """将增强动作转换为索引"""
        # 反向映射
        compression_ratios = [0.1, 0.3, 0.6, 1.0]
        quantization_bits = [4, 8, 16, 32]
        pruning_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
        batch_sizes = [1, 2, 4, 8, 16]
        parallel_degrees = [1, 2, 4, 8]
        
        compression_idx = compression_ratios.index(action.compression_ratio)
        quantization_idx = quantization_bits.index(action.quantization_bits)
        pruning_idx = pruning_ratios.index(action.model_pruning_ratio)
        batch_idx = batch_sizes.index(action.batch_size)
        parallel_idx = parallel_degrees.index(action.parallel_degree)
        
        return (action.partition_point + 
                compression_idx * 21 +
                quantization_idx * 84 +
                pruning_idx * 336 +
                batch_idx * 1680 +
                parallel_idx * 8400)
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def calculate_reward(self, state: EnhancedSystemState, action: EnhancedAction, 
                        metrics: PerformanceMetrics) -> float:
        """计算奖励"""
        return self.reward_function.calculate_reward(state, action, metrics)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        logger.info(f"模型已从 {filepath} 加载")

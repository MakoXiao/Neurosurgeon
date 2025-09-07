"""
自适应划分器 - 核心决策模块
Adaptive Partitioner - Core Decision Module

实现基于强化学习的动态自适应划分策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle
import os
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemState:
    """系统状态数据结构"""
    bandwidth: float  # 网络带宽 (MB/s)
    server_load: float  # 服务器负载 (0-1)
    edge_capability: float  # 边缘设备计算能力 (0-1)
    battery_level: float  # 电池电量 (0-1)
    timestamp: float  # 时间戳
    
@dataclass
class PartitionDecision:
    """划分决策结果"""
    partition_point: int  # 划分点
    confidence: float  # 决策置信度
    predicted_latency: float  # 预测延迟
    predicted_energy: float  # 预测能耗
    reasoning: str  # 决策理由

class HistoryManager:
    """历史数据管理器"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.decisions_history = deque(maxlen=max_history_size)
        self.performance_history = deque(maxlen=max_history_size)
        self.state_history = deque(maxlen=max_history_size)
        
    def add_decision(self, state: SystemState, decision: PartitionDecision, 
                    actual_latency: float, actual_energy: float):
        """添加决策记录"""
        self.decisions_history.append({
            'state': state,
            'decision': decision,
            'actual_latency': actual_latency,
            'actual_energy': actual_energy,
            'timestamp': state.timestamp
        })
        
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取训练数据"""
        if len(self.decisions_history) < 10:
            return np.array([]), np.array([])
            
        states = []
        targets = []
        
        for record in self.decisions_history:
            state = record['state']
            decision = record['decision']
            actual_latency = record['actual_latency']
            
            # 特征向量: [bandwidth, server_load, edge_capability, battery_level]
            feature = np.array([
                state.bandwidth,
                state.server_load, 
                state.edge_capability,
                state.battery_level
            ])
            
            # 目标: 实际性能 (延迟的倒数，越小越好)
            target = 1.0 / (actual_latency + 1e-6)
            
            states.append(feature)
            targets.append(target)
            
        return np.array(states), np.array(targets)

class TimeSeriesPredictor:
    """时间序列预测器 - 预测网络状态变化"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.bandwidth_history = deque(maxlen=window_size)
        self.load_history = deque(maxlen=window_size)
        
    def update(self, bandwidth: float, server_load: float):
        """更新历史数据"""
        self.bandwidth_history.append(bandwidth)
        self.load_history.append(server_load)
        
    def predict_future_state(self, steps_ahead: int = 3) -> Tuple[float, float]:
        """预测未来状态"""
        if len(self.bandwidth_history) < 3:
            return self.bandwidth_history[-1] if self.bandwidth_history else 0.0, \
                   self.load_history[-1] if self.load_history else 0.0
                   
        # 简单的线性趋势预测
        bandwidth_trend = np.polyfit(range(len(self.bandwidth_history)), 
                                   list(self.bandwidth_history), 1)[0]
        load_trend = np.polyfit(range(len(self.load_history)), 
                              list(self.load_history), 1)[0]
        
        # 预测未来值
        future_bandwidth = max(0, self.bandwidth_history[-1] + bandwidth_trend * steps_ahead)
        future_load = max(0, min(1, self.load_history[-1] + load_trend * steps_ahead))
        
        return future_bandwidth, future_load

class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self):
        self.latency_weight = 0.5
        self.energy_weight = 0.3
        self.accuracy_weight = 0.2
        
    def set_preferences(self, latency_weight: float, energy_weight: float, accuracy_weight: float):
        """设置优化偏好"""
        total = latency_weight + energy_weight + accuracy_weight
        self.latency_weight = latency_weight / total
        self.energy_weight = energy_weight / total
        self.accuracy_weight = accuracy_weight / total
        
    def calculate_score(self, latency: float, energy: float, accuracy: float = 1.0) -> float:
        """计算综合得分"""
        # 归一化并计算加权得分
        latency_score = 1.0 / (latency + 1e-6)
        energy_score = 1.0 / (energy + 1e-6)
        accuracy_score = accuracy
        
        return (self.latency_weight * latency_score + 
                self.energy_weight * energy_score + 
                self.accuracy_weight * accuracy_score)

class MLPredictor:
    """机器学习预测器"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        if len(X) > 0:
            self.model.fit(X, y)
            self.is_trained = True
            logger.info(f"ML模型训练完成，样本数: {len(X)}")
            
    def predict(self, state: SystemState) -> float:
        """预测性能"""
        if not self.is_trained:
            return 0.5  # 默认值
            
        feature = np.array([[state.bandwidth, state.server_load, 
                           state.edge_capability, state.battery_level]])
        return self.model.predict(feature)[0]

class ReinforcementLearningAgent:
    """强化学习智能体"""
    
    def __init__(self, state_dim: int = 4, action_dim: int = 21, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # 简单的Q网络
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.epsilon = 0.1  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # 经验回放
        self.memory = deque(maxlen=1000)
        
    def get_action(self, state: SystemState) -> int:
        """获取动作（划分点）"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
            
        state_tensor = torch.FloatTensor([
            state.bandwidth, state.server_load, 
            state.edge_capability, state.battery_level
        ]).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def remember(self, state: SystemState, action: int, reward: float, 
                next_state: SystemState, done: bool):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size: int = 32):
        """经验回放训练"""
        if len(self.memory) < batch_size:
            return
            
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for i in batch:
            state, action, reward, next_state, done = self.memory[i]
            states.append([state.bandwidth, state.server_load, 
                          state.edge_capability, state.battery_level])
            actions.append(action)
            rewards.append(reward)
            next_states.append([next_state.bandwidth, next_state.server_load,
                              next_state.edge_capability, next_state.battery_level])
            dones.append(done)
            
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AdaptivePartitioner:
    """自适应划分器 - 主控制器"""
    
    def __init__(self, model_layers: int = 20):
        self.model_layers = model_layers
        self.history_manager = HistoryManager()
        self.time_series_predictor = TimeSeriesPredictor()
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        self.ml_predictor = MLPredictor()
        self.rl_agent = ReinforcementLearningAgent(action_dim=model_layers + 1)
        
        # 决策策略权重
        self.ml_weight = 0.4
        self.rl_weight = 0.4
        self.heuristic_weight = 0.2
        
    def update_system_state(self, bandwidth: float, server_load: float, 
                          edge_capability: float = 0.8, battery_level: float = 0.9):
        """更新系统状态"""
        state = SystemState(
            bandwidth=bandwidth,
            server_load=server_load,
            edge_capability=edge_capability,
            battery_level=battery_level,
            timestamp=time.time()
        )
        
        self.time_series_predictor.update(bandwidth, server_load)
        return state
        
    def make_decision(self, state: SystemState, model_type: str = "mobilenet") -> PartitionDecision:
        """做出划分决策"""
        # 1. 预测未来状态
        future_bandwidth, future_load = self.time_series_predictor.predict_future_state()
        
        # 2. 创建增强状态（包含未来预测）
        enhanced_state = SystemState(
            bandwidth=(state.bandwidth + future_bandwidth) / 2,
            server_load=(state.server_load + future_load) / 2,
            edge_capability=state.edge_capability,
            battery_level=state.battery_level,
            timestamp=state.timestamp
        )
        
        # 3. 获取多种预测结果
        ml_prediction = self.ml_predictor.predict(enhanced_state)
        rl_action = self.rl_agent.get_action(enhanced_state)
        heuristic_point = self._heuristic_decision(enhanced_state)
        
        # 4. 融合决策
        final_decision = self._fuse_decisions(ml_prediction, rl_action, heuristic_point, enhanced_state)
        
        return final_decision
        
    def _heuristic_decision(self, state: SystemState) -> int:
        """启发式决策"""
        # 基于带宽的简单启发式规则
        if state.bandwidth < 1.0:  # 低带宽
            return min(15, self.model_layers)  # 更多在边缘端
        elif state.bandwidth > 50.0:  # 高带宽
            return 0  # 全部在云端
        else:  # 中等带宽
            return min(8, self.model_layers // 2)  # 平衡划分
            
    def _fuse_decisions(self, ml_prediction: float, rl_action: int, 
                       heuristic_point: int, state: SystemState) -> PartitionDecision:
        """融合多种决策"""
        # 将ML预测转换为划分点
        ml_point = int(ml_prediction * self.model_layers)
        ml_point = max(0, min(self.model_layers, ml_point))
        
        # 加权融合
        final_point = int(
            self.ml_weight * ml_point +
            self.rl_weight * rl_action +
            self.heuristic_weight * heuristic_point
        )
        final_point = max(0, min(self.model_layers, final_point))
        
        # 计算置信度
        confidence = self._calculate_confidence(state, final_point)
        
        # 预测性能
        predicted_latency, predicted_energy = self._predict_performance(state, final_point)
        
        # 生成决策理由
        reasoning = self._generate_reasoning(state, final_point, ml_point, rl_action, heuristic_point)
        
        return PartitionDecision(
            partition_point=final_point,
            confidence=confidence,
            predicted_latency=predicted_latency,
            predicted_energy=predicted_energy,
            reasoning=reasoning
        )
        
    def _calculate_confidence(self, state: SystemState, partition_point: int) -> float:
        """计算决策置信度"""
        # 基于历史数据的相似性计算置信度
        if len(self.history_manager.decisions_history) < 5:
            return 0.5
            
        # 简化的置信度计算
        if len(self.history_manager.decisions_history) > 0:
            recent_bandwidths = [r['state'].bandwidth for r in list(self.history_manager.decisions_history)[-10:]]
            bandwidth_std = np.std(recent_bandwidths) if len(recent_bandwidths) > 1 else 0.0
            confidence = max(0.1, 1.0 - bandwidth_std / 100.0)
        else:
            confidence = 0.5
        
        return confidence
        
    def _predict_performance(self, state: SystemState, partition_point: int) -> Tuple[float, float]:
        """预测性能"""
        # 简化的性能预测模型
        base_latency = 100.0
        base_energy = 50.0
        
        # 基于划分点调整
        if partition_point == 0:  # 全部云端
            latency = base_latency + state.bandwidth * 0.1
            energy = base_energy * 0.3
        elif partition_point == self.model_layers:  # 全部边缘
            latency = base_latency * 2.0
            energy = base_energy * 1.5
        else:  # 协同
            latency = base_latency + (self.model_layers - partition_point) * 5.0
            energy = base_energy * (0.5 + partition_point / self.model_layers * 0.5)
            
        return latency, energy
        
    def _generate_reasoning(self, state: SystemState, final_point: int, 
                          ml_point: int, rl_action: int, heuristic_point: int) -> str:
        """生成决策理由"""
        reasons = []
        
        if state.bandwidth < 1.0:
            reasons.append("低带宽环境，优先边缘计算")
        elif state.bandwidth > 50.0:
            reasons.append("高带宽环境，云端计算更优")
        else:
            reasons.append("中等带宽，采用协同策略")
            
        if state.server_load > 0.8:
            reasons.append("服务器负载高，增加边缘计算")
            
        if state.battery_level < 0.3:
            reasons.append("电池电量低，减少边缘计算")
            
        return "; ".join(reasons)
        
    def learn_from_feedback(self, state: SystemState, decision: PartitionDecision, 
                          actual_latency: float, actual_energy: float):
        """从反馈中学习"""
        # 1. 更新历史记录
        self.history_manager.add_decision(state, decision, actual_latency, actual_energy)
        
        # 2. 训练ML模型
        X, y = self.history_manager.get_training_data()
        if len(X) > 0:
            self.ml_predictor.train(X, y)
            
        # 3. 更新RL智能体
        reward = self.multi_objective_optimizer.calculate_score(actual_latency, actual_energy)
        # 这里需要下一个状态，简化处理
        next_state = state  # 实际应用中应该有真实的下一个状态
        self.rl_agent.remember(state, decision.partition_point, reward, next_state, False)
        self.rl_agent.replay()
        
        logger.info(f"学习完成 - 实际延迟: {actual_latency:.2f}ms, 奖励: {reward:.3f}")
        
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'ml_predictor': self.ml_predictor.model if self.ml_predictor.is_trained else None,
            'rl_agent': self.rl_agent.q_network.state_dict(),
            'history': list(self.history_manager.decisions_history),
            'time_series': {
                'bandwidth_history': list(self.time_series_predictor.bandwidth_history),
                'load_history': list(self.time_series_predictor.load_history)
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"模型已保存到: {filepath}")
        
    def load_model(self, filepath: str):
        """加载模型"""
        if not os.path.exists(filepath):
            logger.warning(f"模型文件不存在: {filepath}")
            return
            
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        # 加载ML模型
        if model_data['ml_predictor'] is not None:
            self.ml_predictor.model = model_data['ml_predictor']
            self.ml_predictor.is_trained = True
            
        # 加载RL模型
        self.rl_agent.q_network.load_state_dict(model_data['rl_agent'])
        
        # 加载历史数据
        self.history_manager.decisions_history = deque(model_data['history'], maxlen=1000)
        
        # 加载时间序列数据
        ts_data = model_data['time_series']
        self.time_series_predictor.bandwidth_history = deque(ts_data['bandwidth_history'], maxlen=10)
        self.time_series_predictor.load_history = deque(ts_data['load_history'], maxlen=10)
        
        logger.info(f"模型已从 {filepath} 加载")

# 导入时间模块
import time

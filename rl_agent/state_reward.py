"""
29维状态空间和双目标奖励函数设计（创新点三）
"""
import torch
import numpy as np
import psutil
import time
from collections import deque


class StateSpace:
    """29维状态空间"""
    
    def __init__(self, history_window=10):
        """
        Args:
            history_window: 历史窗口大小
        """
        self.history_window = history_window
        
        # 历史记录
        self.latency_history = deque(maxlen=history_window)
        self.accuracy_history = deque(maxlen=history_window)
        self.partition_history = deque(maxlen=history_window)
        self.compression_history = deque(maxlen=history_window)
        
        # 归一化参数
        self.bandwidth_max = 1000.0  # MB/s
        self.latency_max = 5000.0  # ms
        self.feature_size_max = 100.0  # MB
        
    def get_device_state(self, edge_device=True):
        """
        获取设备状态（7维）
        
        Returns:
            device_state: [edge_cpu, edge_memory, edge_battery, edge_compute,
                          cloud_cpu, cloud_gpu, cloud_memory]
        """
        if edge_device:
            # 边缘设备状态
            edge_cpu = psutil.cpu_percent() / 100.0
            edge_memory = psutil.virtual_memory().percent / 100.0
            edge_battery = self._get_battery_level()
            edge_compute = 0.5  # 归一化的计算能力（可根据实际情况调整）
        else:
            edge_cpu = 0.3
            edge_memory = 0.3
            edge_battery = 1.0
            edge_compute = 0.5
        
        # 云端设备状态（模拟）
        cloud_cpu = 0.2  # 云端通常有更多资源
        cloud_gpu = 0.1
        cloud_memory = 0.15
        
        device_state = np.array([
            edge_cpu, edge_memory, edge_battery, edge_compute,
            cloud_cpu, cloud_gpu, cloud_memory
        ], dtype=np.float32)
        
        return device_state
    
    def get_network_state(self, bandwidth=100.0, latency=50.0, 
                         packet_loss=0.01, signal_strength=0.8,
                         network_type='wifi'):
        """
        获取网络状态（8维）
        
        Args:
            bandwidth: 带宽 (MB/s)
            latency: 延迟 (ms)
            packet_loss: 丢包率 [0, 1]
            signal_strength: 信号强度 [0, 1]
            network_type: 'wifi', 'lte', '5g'
        
        Returns:
            network_state: [bandwidth, latency, packet_loss, signal_strength,
                          network_quality, wifi, lte, 5g]
        """
        # 归一化
        bandwidth_norm = min(bandwidth / self.bandwidth_max, 1.0)
        latency_norm = min(latency / self.latency_max, 1.0)
        
        # 计算网络质量（基于多个指标的综合评分）
        network_quality = (bandwidth_norm + (1 - latency_norm) + 
                          (1 - packet_loss) + signal_strength) / 4.0
        
        # 网络类型one-hot编码
        network_type_map = {'wifi': [1, 0, 0], 'lte': [0, 1, 0], '5g': [0, 0, 1]}
        network_type_vec = network_type_map.get(network_type.lower(), [1, 0, 0])
        
        network_state = np.array([
            bandwidth_norm, latency_norm, packet_loss, signal_strength,
            network_quality
        ] + network_type_vec, dtype=np.float32)
        
        return network_state
    
    def get_task_state(self, input_size=3*224*224, model_complexity=1e7,
                      queue_length=0, expected_accuracy=0.95):
        """
        获取任务状态（4维）
        
        Args:
            input_size: 输入数据大小（像素数）
            model_complexity: 模型复杂度（参数量或FLOPs）
            queue_length: 任务队列长度
            expected_accuracy: 期望精度
        
        Returns:
            task_state: [input_size, model_complexity, queue_length, expected_accuracy]
        """
        # 归一化
        input_size_norm = (input_size / (3 * 224 * 224))  # 相对于标准输入
        model_complexity_norm = min(model_complexity / 1e8, 1.0)
        queue_length_norm = min(queue_length / 100.0, 1.0)
        
        task_state = np.array([
            input_size_norm, model_complexity_norm, 
            queue_length_norm, expected_accuracy
        ], dtype=np.float32)
        
        return task_state
    
    def get_history_state(self):
        """
        获取历史状态（6维）
        
        Returns:
            history_state: [last_partition, last_compression, last_latency,
                          last_accuracy, avg_latency, avg_accuracy]
        """
        if len(self.latency_history) == 0:
            # 初始状态
            return np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        
        last_partition = self.partition_history[-1] / 10.0  # 假设最多10个分割点
        last_compression = self.compression_history[-1]
        last_latency = min(self.latency_history[-1] / self.latency_max, 1.0)
        last_accuracy = self.accuracy_history[-1]
        
        avg_latency = min(np.mean(self.latency_history) / self.latency_max, 1.0)
        avg_accuracy = np.mean(self.accuracy_history)
        
        history_state = np.array([
            last_partition, last_compression, last_latency,
            last_accuracy, avg_latency, avg_accuracy
        ], dtype=np.float32)
        
        return history_state
    
    def get_feature_state(self, feature_tensor):
        """
        获取中间特征状态（4维）
        
        Args:
            feature_tensor: 中间特征张量
        
        Returns:
            feature_state: [feature_size, channels, sparsity, compressibility]
        """
        if feature_tensor is None:
            return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        
        # 特征大小（MB）
        feature_size = feature_tensor.numel() * feature_tensor.element_size() / (1024 * 1024)
        feature_size_norm = min(feature_size / self.feature_size_max, 1.0)
        
        # 通道数
        if len(feature_tensor.shape) == 4:
            channels = feature_tensor.shape[1]
            channels_norm = min(channels / 512.0, 1.0)
        else:
            channels_norm = 0.5
        
        # 稀疏度
        threshold = 1e-3
        sparsity = (torch.abs(feature_tensor) < threshold).float().mean().item()
        
        # 可压缩性（基于值的分布）
        std = torch.std(feature_tensor).item()
        compressibility = min(std / 10.0, 1.0)
        
        feature_state = np.array([
            feature_size_norm, channels_norm, sparsity, compressibility
        ], dtype=np.float32)
        
        return feature_state
    
    def get_full_state(self, feature_tensor=None, bandwidth=100.0, 
                      latency=50.0, network_type='wifi'):
        """
        获取完整的29维状态
        
        Args:
            feature_tensor: 中间特征张量
            bandwidth: 带宽
            latency: 网络延迟
            network_type: 网络类型
        
        Returns:
            state: 29维状态向量
        """
        device_state = self.get_device_state()
        network_state = self.get_network_state(bandwidth, latency, 
                                               network_type=network_type)
        task_state = self.get_task_state()
        history_state = self.get_history_state()
        feature_state = self.get_feature_state(feature_tensor)
        
        full_state = np.concatenate([
            device_state,    # 7维
            network_state,   # 8维
            task_state,      # 4维
            history_state,   # 6维
            feature_state    # 4维
        ])
        
        assert len(full_state) == 29, f"状态维度错误: {len(full_state)}"
        
        return full_state
    
    def update_history(self, partition_point, compression_rate, 
                      latency, accuracy):
        """
        更新历史记录
        
        Args:
            partition_point: 分割点
            compression_rate: 压缩率
            latency: 时延
            accuracy: 准确率
        """
        self.partition_history.append(partition_point)
        self.compression_history.append(compression_rate)
        self.latency_history.append(latency)
        self.accuracy_history.append(accuracy)
    
    def _get_battery_level(self):
        """获取电池电量（如果是移动设备）"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return battery.percent / 100.0
        except:
            pass
        return 1.0  # 默认满电


class RewardFunction:
    """双目标优化奖励函数"""
    
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2,
                 target_accuracy=0.90, max_latency=1000.0,
                 accuracy_weight=1.0, latency_weight=1.0):
        """
        Args:
            alpha: 准确率权重 (降低从0.6到0.5)
            beta: 时延权重 (降低从0.4到0.3)
            gamma: 压缩率权重 (新增0.2)
            target_accuracy: 目标准确率阈值
            max_latency: 最大可接受时延 (ms)
            accuracy_weight: 准确率缩放因子
            latency_weight: 时延缩放因子
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma  # 新增压缩率权重
        self.target_accuracy = target_accuracy
        self.max_latency = max_latency
        self.accuracy_weight = accuracy_weight
        self.latency_weight = latency_weight
        
        # 奖励归一化
        self.reward_history = deque(maxlen=100)
    
    def compute_reward(self, accuracy, latency, compression_rate=None):
        """
        计算奖励
        
        Args:
            accuracy: 推理准确率 [0, 1]
            latency: 推理时延 (ms)
            compression_rate: 压缩率（可选）
        
        Returns:
            reward: 奖励值
        """
        # 准确率奖励
        if accuracy >= self.target_accuracy:
            # 超过目标准确率，给予正奖励
            accuracy_reward = self.accuracy_weight * (
                (accuracy - self.target_accuracy) / (1 - self.target_accuracy)
            )
        else:
            # 低于目标准确率，给予严重惩罚（非对称惩罚）
            accuracy_reward = -self.accuracy_weight * (
                (self.target_accuracy - accuracy) / self.target_accuracy
            ) * 2.0  # 2倍惩罚
        
        # 时延奖励（越低越好）
        latency_normalized = latency / self.max_latency
        
        if latency <= self.max_latency:
            # 在可接受范围内
            latency_reward = -self.latency_weight * latency_normalized
        else:
            # 超过最大时延，给予中等惩罚
            latency_reward = -self.latency_weight * (
                1.0 + (latency - self.max_latency) / self.max_latency
            )
        
        # 压缩率奖励（更智能的压缩策略）
        compression_reward = 0.0
        if compression_rate is not None:
            # 根据准确率动态调整压缩奖励
            if accuracy >= self.target_accuracy * 0.95:
                # 准确率很好时，鼓励高压缩
                compression_reward = (1.0 - compression_rate) * 1.5
            elif accuracy >= self.target_accuracy * 0.9:
                # 准确率较好时，适度压缩
                compression_reward = (1.0 - compression_rate) * 1.0
            else:
                # 准确率不佳时，降低压缩奖励
                compression_reward = (1.0 - compression_rate) * 0.3
        
        # 综合奖励（使用新的权重分配）
        reward = (self.alpha * accuracy_reward + 
                 self.beta * latency_reward + 
                 self.gamma * compression_reward)
        
        return reward
    
    def compute_normalized_reward(self, accuracy, latency, compression_rate=None):
        """
        计算归一化奖励
        
        Args:
            accuracy: 推理准确率
            latency: 推理时延
            compression_rate: 压缩率
        
        Returns:
            normalized_reward: 归一化奖励
        """
        reward = self.compute_reward(accuracy, latency, compression_rate)
        
        self.reward_history.append(reward)
        
        if len(self.reward_history) > 1:
            mean_reward = np.mean(self.reward_history)
            std_reward = np.std(self.reward_history)
            
            if std_reward > 1e-6:
                normalized_reward = (reward - mean_reward) / std_reward
            else:
                normalized_reward = reward - mean_reward
        else:
            normalized_reward = reward
        
        return normalized_reward
    
    def compute_pareto_reward(self, accuracy, latency, 
                             reference_accuracy=0.85, reference_latency=500.0):
        """
        计算帕累托最优奖励
        
        Args:
            accuracy: 当前准确率
            latency: 当前时延
            reference_accuracy: 参考准确率（baseline）
            reference_latency: 参考时延（baseline）
        
        Returns:
            pareto_reward: 帕累托奖励
        """
        # 计算相对于baseline的改进
        accuracy_improvement = (accuracy - reference_accuracy) / reference_accuracy
        latency_improvement = (reference_latency - latency) / reference_latency
        
        # 帕累托奖励：两个目标都改进才给予正奖励
        if accuracy_improvement > 0 and latency_improvement > 0:
            pareto_reward = self.alpha * accuracy_improvement + self.beta * latency_improvement
        elif accuracy_improvement > 0:
            # 只有准确率改进
            pareto_reward = self.alpha * accuracy_improvement + self.beta * latency_improvement * 0.5
        elif latency_improvement > 0:
            # 只有时延改进
            pareto_reward = self.alpha * accuracy_improvement * 0.5 + self.beta * latency_improvement
        else:
            # 两个都没改进
            pareto_reward = self.alpha * accuracy_improvement + self.beta * latency_improvement
        
        return pareto_reward


if __name__ == '__main__':
    print("测试状态空间和奖励函数...")
    
    # 测试状态空间
    print("\n=== 测试状态空间 ===")
    state_space = StateSpace(history_window=10)
    
    # 创建一个模拟特征
    feature = torch.randn(1, 64, 28, 28)
    
    # 获取完整状态
    state = state_space.get_full_state(
        feature_tensor=feature,
        bandwidth=100.0,
        latency=50.0,
        network_type='wifi'
    )
    
    print(f"状态维度: {len(state)}")
    print(f"状态范围: [{state.min():.3f}, {state.max():.3f}]")
    
    # 分解状态
    print("\n状态分解:")
    print(f"  设备状态 (7维): {state[0:7]}")
    print(f"  网络状态 (8维): {state[7:15]}")
    print(f"  任务状态 (4维): {state[15:19]}")
    print(f"  历史状态 (6维): {state[19:25]}")
    print(f"  特征状态 (4维): {state[25:29]}")
    
    # 更新历史
    state_space.update_history(
        partition_point=3,
        compression_rate=0.5,
        latency=100.0,
        accuracy=0.92
    )
    
    state_updated = state_space.get_full_state(feature_tensor=feature)
    print(f"\n更新后的历史状态: {state_updated[19:25]}")
    
    # 测试奖励函数
    print("\n=== 测试奖励函数 ===")
    reward_fn = RewardFunction(
        alpha=0.6, beta=0.4,
        target_accuracy=0.90,
        max_latency=1000.0
    )
    
    # 测试不同场景
    scenarios = [
        (0.95, 500.0, "高准确率，中等时延"),
        (0.92, 800.0, "中等准确率，高时延"),
        (0.88, 300.0, "低准确率，低时延"),
        (0.85, 1200.0, "低准确率，超高时延"),
    ]
    
    print("\n奖励计算:")
    for accuracy, latency, desc in scenarios:
        reward = reward_fn.compute_reward(accuracy, latency, compression_rate=0.5)
        print(f"  {desc}")
        print(f"    准确率: {accuracy:.2f}, 时延: {latency:.0f}ms")
        print(f"    奖励: {reward:.4f}")
    
    # 测试帕累托奖励
    print("\n帕累托奖励:")
    pareto_reward = reward_fn.compute_pareto_reward(
        accuracy=0.92, latency=400.0,
        reference_accuracy=0.85, reference_latency=500.0
    )
    print(f"  相对baseline的帕累托奖励: {pareto_reward:.4f}")


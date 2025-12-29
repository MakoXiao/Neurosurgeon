"""
多进程边云协同推理环境
模拟边缘设备和云端设备的协同推理过程
"""
import torch
import torch.nn as nn
import numpy as np
import time
from multiprocessing import Process, Queue, Value
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_zoo import get_model
from compression.pruning_compression import AdaptivePruningCompressor
from rl_agent.state_reward import StateSpace, RewardFunction


class EdgeDevice:
    """边缘设备"""
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: 可分割的模型
            device: 计算设备
        """
        self.device = device
        self.model = model.to(device)
        self.compressor = AdaptivePruningCompressor()
    
    def forward_to_split(self, input_data, split_point):
        """
        在边缘端推理到分割点
        
        Args:
            input_data: 输入数据
            split_point: 分割点
        
        Returns:
            edge_output: 边缘端输出
            edge_time: 边缘端推理时间 (ms)
        """
        # 确保输入数据在正确的设备上
        input_data = input_data.to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            edge_output = self.model.forward_to_split(input_data, split_point)
        edge_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        return edge_output, edge_time
    
    def compress_feature(self, feature, compression_rate, pruning_type='structured'):
        """
        压缩中间特征
        
        Args:
            feature: 中间特征
            compression_rate: 压缩率
            pruning_type: 剪枝类型
        
        Returns:
            compressed_data: 压缩数据
            compress_time: 压缩时间 (ms)
        """
        start_time = time.time()
        compressed_data = self.compressor.compress(
            feature, compression_rate, pruning_type=pruning_type
        )
        compress_time = (time.time() - start_time) * 1000
        
        return compressed_data, compress_time


class CloudDevice:
    """云端设备"""
    
    def __init__(self, model, device='cpu'):
        """
        Args:
            model: 可分割的模型
            device: 计算设备
        """
        self.device = device
        self.model = model.to(device)
        self.compressor = AdaptivePruningCompressor()
    
    def decompress_feature(self, compressed_data):
        """
        解压缩中间特征
        
        Args:
            compressed_data: 压缩数据
        
        Returns:
            recovered_feature: 恢复的特征
            decompress_time: 解压缩时间 (ms)
        """
        start_time = time.time()
        recovered_feature = self.compressor.decompress(
            compressed_data, device=self.device
        )
        decompress_time = (time.time() - start_time) * 1000
        
        return recovered_feature, decompress_time
    
    def forward_from_split(self, feature, split_point):
        """
        从分割点继续推理
        
        Args:
            feature: 中间特征
            split_point: 分割点
        
        Returns:
            output: 最终输出
            cloud_time: 云端推理时间 (ms)
        """
        # 确保特征在正确的设备上
        feature = feature.to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            output = self.model.forward_from_split(feature, split_point)
        cloud_time = (time.time() - start_time) * 1000
        
        return output, cloud_time


class CollaborativeInferenceEnv:
    """协同推理环境"""
    
    def __init__(self, model_name, num_classes=101, 
                 edge_device='cpu', cloud_device='cpu',
                 bandwidth=100.0, network_latency=50.0,
                 target_accuracy=0.90, max_latency=1000.0):
        """
        Args:
            model_name: 模型名称
            num_classes: 类别数量
            edge_device: 边缘设备
            cloud_device: 云端设备
            bandwidth: 带宽 (MB/s)
            network_latency: 网络延迟 (ms)
            target_accuracy: 目标准确率
            max_latency: 最大时延
        """
        # 创建模型
        self.model = get_model(model_name, num_classes=num_classes, pretrained=False)
        self.model_name = model_name
        
        # 创建边缘和云端设备（使用独立的模型实例）
        edge_model = get_model(model_name, num_classes=num_classes, pretrained=False)
        cloud_model = get_model(model_name, num_classes=num_classes, pretrained=False)
        
        self.edge = EdgeDevice(edge_model, device=edge_device)
        self.cloud = CloudDevice(cloud_model, device=cloud_device)
        
        # 网络参数
        self.bandwidth = bandwidth
        self.network_latency = network_latency
        
        # 状态空间和奖励函数
        self.state_space = StateSpace()
        self.reward_fn = RewardFunction(
            target_accuracy=target_accuracy,
            max_latency=max_latency
        )
        
        # 当前状态
        self.current_input = None
        self.current_label = None
        self.current_feature = None
    
    def reset(self, input_data, label):
        """
        重置环境
        
        Args:
            input_data: 输入数据
            label: 标签
        
        Returns:
            state: 初始状态
        """
        self.current_input = input_data
        self.current_label = label
        self.current_feature = None
        
        # 获取初始状态
        state = self.state_space.get_full_state(
            feature_tensor=None,
            bandwidth=self.bandwidth,
            latency=self.network_latency
        )
        
        return state
    
    def step(self, action):
        """
        执行一步协同推理
        
        Args:
            action: (partition_point, compression_rate)
        
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        partition_point, compression_rate = action
        
        # 1. 边缘端推理
        edge_output, edge_time = self.edge.forward_to_split(
            self.current_input, partition_point
        )
        self.current_feature = edge_output
        
        # 2. 压缩中间特征
        compressed_data, compress_time = self.edge.compress_feature(
            edge_output, compression_rate
        )
        
        # 3. 计算传输时间
        transfer_time = self._compute_transfer_time(compressed_data)
        
        # 4. 云端解压缩
        recovered_feature, decompress_time = self.cloud.decompress_feature(
            compressed_data
        )
        
        # 5. 云端推理
        output, cloud_time = self.cloud.forward_from_split(
            recovered_feature, partition_point
        )
        
        # 6. 计算准确率（确保pred和label在同一设备上）
        pred = torch.argmax(output, dim=1).cpu()
        label = self.current_label.cpu() if isinstance(self.current_label, torch.Tensor) else torch.tensor([self.current_label])
        correct = (pred == label).float().mean().item()
        
        # 7. 计算总时延
        total_latency = (edge_time + compress_time + transfer_time + 
                        decompress_time + cloud_time)
        
        # 8. 计算奖励
        reward = self.reward_fn.compute_reward(
            accuracy=correct,
            latency=total_latency,
            compression_rate=compression_rate
        )
        
        # 9. 更新历史
        self.state_space.update_history(
            partition_point=partition_point,
            compression_rate=compression_rate,
            latency=total_latency,
            accuracy=correct
        )
        
        # 10. 获取下一个状态
        next_state = self.state_space.get_full_state(
            feature_tensor=edge_output,
            bandwidth=self.bandwidth,
            latency=self.network_latency
        )
        
        # 11. 额外信息
        info = {
            'edge_time': edge_time,
            'compress_time': compress_time,
            'transfer_time': transfer_time,
            'decompress_time': decompress_time,
            'cloud_time': cloud_time,
            'total_latency': total_latency,
            'accuracy': correct,
            'partition_point': partition_point,
            'compression_rate': compression_rate,
            'feature_size': edge_output.numel() * edge_output.element_size() / (1024 * 1024)
        }
        
        done = True  # 单步推理任务
        
        return next_state, reward, done, info
    
    def _compute_transfer_time(self, compressed_data):
        """
        计算传输时间
        
        Args:
            compressed_data: 压缩数据
        
        Returns:
            transfer_time: 传输时间 (ms)
        """
        # 计算数据大小（MB）
        if compressed_data['pruning_type'] == 'structured':
            data_size = (
                compressed_data['pruned_feature'].numel() * 
                compressed_data['pruned_feature'].element_size() +
                compressed_data['mask_indices'].numel() * 
                compressed_data['mask_indices'].element_size()
            ) / (1024 * 1024)
        else:
            data_size = (
                compressed_data['pruned_values'].numel() * 
                compressed_data['pruned_values'].element_size() +
                compressed_data['mask_indices'].numel() * 
                compressed_data['mask_indices'].element_size()
            ) / (1024 * 1024)
        
        # 传输时间 = 数据大小 / 带宽 + 网络延迟
        transfer_time = (data_size / self.bandwidth) * 1000 + self.network_latency
        
        return transfer_time
    
    def evaluate_baseline(self, input_data, label, baseline_type='all_edge'):
        """
        评估基线方法
        
        Args:
            input_data: 输入数据
            label: 标签
            baseline_type: 'all_edge' 或 'all_cloud'
        
        Returns:
            info: 评估信息
        """
        if baseline_type == 'all_edge':
            # 全边缘推理（使用边缘模型）
            start_time = time.time()
            with torch.no_grad():
                input_data = input_data.to(self.edge.device)
                output = self.edge.model(input_data)
            total_time = (time.time() - start_time) * 1000
            
            pred = torch.argmax(output, dim=1).cpu()
            label_cpu = label.cpu() if isinstance(label, torch.Tensor) else torch.tensor([label])
            accuracy = (pred == label_cpu).float().mean().item()
            
            return {
                'total_latency': total_time,
                'accuracy': accuracy,
                'type': 'all_edge'
            }
        
        elif baseline_type == 'all_cloud':
            # 全云端推理（需要传输原始输入，使用云端模型）
            input_size = input_data.numel() * input_data.element_size() / (1024 * 1024)
            transfer_time = (input_size / self.bandwidth) * 1000 + self.network_latency
            
            start_time = time.time()
            with torch.no_grad():
                input_data = input_data.to(self.cloud.device)
                output = self.cloud.model(input_data)
            cloud_time = (time.time() - start_time) * 1000
            
            total_time = transfer_time + cloud_time
            
            pred = torch.argmax(output, dim=1).cpu()
            label_cpu = label.cpu() if isinstance(label, torch.Tensor) else torch.tensor([label])
            accuracy = (pred == label_cpu).float().mean().item()
            
            return {
                'total_latency': total_time,
                'accuracy': accuracy,
                'transfer_time': transfer_time,
                'cloud_time': cloud_time,
                'type': 'all_cloud'
            }


def edge_process(input_queue, output_queue, model_name, split_point, 
                compression_rate, device='cpu'):
    """
    边缘设备进程
    
    Args:
        input_queue: 输入队列
        output_queue: 输出队列
        model_name: 模型名称
        split_point: 分割点
        compression_rate: 压缩率
        device: 设备
    """
    # 创建模型和压缩器
    model = get_model(model_name, num_classes=101, pretrained=False)
    edge = EdgeDevice(model, device=device)
    
    while True:
        # 从队列获取输入
        data = input_queue.get()
        
        if data is None:  # 结束信号
            break
        
        input_data, task_id = data
        
        # 边缘端推理
        edge_output, edge_time = edge.forward_to_split(input_data, split_point)
        
        # 压缩
        compressed_data, compress_time = edge.compress_feature(
            edge_output, compression_rate
        )
        
        # 发送到云端
        output_queue.put((compressed_data, split_point, task_id, 
                         edge_time, compress_time))


def cloud_process(input_queue, output_queue, model_name, device='cpu'):
    """
    云端设备进程
    
    Args:
        input_queue: 输入队列
        output_queue: 输出队列
        model_name: 模型名称
        device: 设备
    """
    # 创建模型
    model = get_model(model_name, num_classes=101, pretrained=False)
    cloud = CloudDevice(model, device=device)
    
    while True:
        # 从队列获取数据
        data = input_queue.get()
        
        if data is None:  # 结束信号
            break
        
        compressed_data, split_point, task_id, edge_time, compress_time = data
        
        # 解压缩
        recovered_feature, decompress_time = cloud.decompress_feature(
            compressed_data
        )
        
        # 云端推理
        output, cloud_time = cloud.forward_from_split(
            recovered_feature, split_point
        )
        
        # 发送结果
        output_queue.put((output, task_id, edge_time, compress_time, 
                         decompress_time, cloud_time))


if __name__ == '__main__':
    print("测试协同推理环境...")
    
    # 创建环境
    env = CollaborativeInferenceEnv(
        model_name='resnet18',
        num_classes=101,
        edge_device='cpu',
        cloud_device='cpu',
        bandwidth=100.0,
        network_latency=50.0
    )
    
    print(f"模型: {env.model_name}")
    print(f"分割点数量: {len(env.model.get_split_points())}")
    
    # 创建测试数据
    input_data = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 101, (2,))
    
    # 重置环境
    state = env.reset(input_data, labels)
    print(f"\n初始状态形状: {state.shape}")
    
    # 测试不同的动作
    print("\n测试不同的分割点和压缩率:")
    for partition_point in [1, 3, 5]:
        for compression_rate in [0.3, 0.5, 0.7]:
            action = (partition_point, compression_rate)
            next_state, reward, done, info = env.step(action)
            
            print(f"\n动作: 分割点={partition_point}, 压缩率={compression_rate:.1f}")
            print(f"  总时延: {info['total_latency']:.2f}ms")
            print(f"    边缘时间: {info['edge_time']:.2f}ms")
            print(f"    压缩时间: {info['compress_time']:.2f}ms")
            print(f"    传输时间: {info['transfer_time']:.2f}ms")
            print(f"    解压时间: {info['decompress_time']:.2f}ms")
            print(f"    云端时间: {info['cloud_time']:.2f}ms")
            print(f"  准确率: {info['accuracy']:.2f}")
            print(f"  奖励: {reward:.4f}")
            print(f"  特征大小: {info['feature_size']:.2f}MB")
    
    # 测试基线方法
    print("\n\n测试基线方法:")
    for baseline_type in ['all_edge', 'all_cloud']:
        info = env.evaluate_baseline(input_data, labels, baseline_type)
        print(f"\n{info['type']}:")
        print(f"  总时延: {info['total_latency']:.2f}ms")
        print(f"  准确率: {info['accuracy']:.2f}")


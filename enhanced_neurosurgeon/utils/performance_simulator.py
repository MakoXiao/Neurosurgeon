"""
性能模拟器
Performance Simulator

模拟不同划分策略下的性能表现
"""

import numpy as np
import torch
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class PerformanceSimulator:
    """性能模拟器"""
    
    def __init__(self):
        # 模型性能基准数据
        self.model_profiles = {
            "mobilenet": {
                "base_latency": 80.0,
                "base_energy": 40.0,
                "layer_complexity": [1.0, 1.2, 1.1, 1.3, 1.0, 1.2, 1.1, 1.3, 1.0, 1.2, 
                                   1.1, 1.3, 1.0, 1.2, 1.1, 1.3, 1.0, 1.2, 1.1, 1.3],
                "data_sizes": [0.1, 0.2, 0.15, 0.25, 0.1, 0.2, 0.15, 0.25, 0.1, 0.2,
                             0.15, 0.25, 0.1, 0.2, 0.15, 0.25, 0.1, 0.2, 0.15, 0.25]
            },
            "vggnet": {
                "base_latency": 200.0,
                "base_energy": 100.0,
                "layer_complexity": [2.0, 2.5, 2.2, 2.8, 2.0, 2.5, 2.2, 2.8, 2.0, 2.5,
                                   2.2, 2.8, 2.0, 2.5, 2.2, 2.8, 2.0, 2.5, 2.2, 2.8],
                "data_sizes": [0.3, 0.4, 0.35, 0.45, 0.3, 0.4, 0.35, 0.45, 0.3, 0.4,
                             0.35, 0.45, 0.3, 0.4, 0.35, 0.45, 0.3, 0.4, 0.35, 0.45]
            },
            "alexnet": {
                "base_latency": 120.0,
                "base_energy": 60.0,
                "layer_complexity": [1.5, 1.8, 1.6, 1.9, 1.5, 1.8, 1.6, 1.9, 1.5, 1.8,
                                   1.6, 1.9, 1.5, 1.8, 1.6, 1.9, 1.5, 1.8, 1.6, 1.9],
                "data_sizes": [0.2, 0.3, 0.25, 0.35, 0.2, 0.3, 0.25, 0.35, 0.2, 0.3,
                             0.25, 0.35, 0.2, 0.3, 0.25, 0.35, 0.2, 0.3, 0.25, 0.35]
            },
            "lenet": {
                "base_latency": 50.0,
                "base_energy": 25.0,
                "layer_complexity": [0.8, 1.0, 0.9, 1.1, 0.8, 1.0, 0.9, 1.1, 0.8, 1.0,
                                   0.9, 1.1, 0.8, 1.0, 0.9, 1.1, 0.8, 1.0, 0.9, 1.1],
                "data_sizes": [0.05, 0.1, 0.08, 0.12, 0.05, 0.1, 0.08, 0.12, 0.05, 0.1,
                             0.08, 0.12, 0.05, 0.1, 0.08, 0.12, 0.05, 0.1, 0.08, 0.12]
            }
        }
        
        # 设备性能参数
        self.device_profiles = {
            "edge": {
                "compute_power": 0.8,  # 相对计算能力
                "energy_efficiency": 1.2,  # 能耗效率
                "memory_bandwidth": 0.6
            },
            "cloud": {
                "compute_power": 2.0,
                "energy_efficiency": 0.8,
                "memory_bandwidth": 2.5
            }
        }
        
    def simulate_performance(self, partition_point: int, bandwidth: float, 
                           model_type: str) -> Tuple[float, float]:
        """模拟性能"""
        if model_type not in self.model_profiles:
            model_type = "mobilenet"  # 默认模型
            
        profile = self.model_profiles[model_type]
        edge_profile = self.device_profiles["edge"]
        cloud_profile = self.device_profiles["cloud"]
        
        # 计算边缘端性能
        edge_latency = self._calculate_edge_latency(partition_point, profile, edge_profile)
        edge_energy = self._calculate_edge_energy(partition_point, profile, edge_profile)
        
        # 计算传输性能
        transmission_latency, transmission_energy = self._calculate_transmission(
            partition_point, profile, bandwidth
        )
        
        # 计算云端性能
        cloud_latency = self._calculate_cloud_latency(partition_point, profile, cloud_profile)
        cloud_energy = self._calculate_cloud_energy(partition_point, profile, cloud_profile)
        
        # 总性能
        total_latency = edge_latency + transmission_latency + cloud_latency
        total_energy = edge_energy + transmission_energy + cloud_energy
        
        return total_latency, total_energy
        
    def _calculate_edge_latency(self, partition_point: int, model_profile: Dict, 
                              device_profile: Dict) -> float:
        """计算边缘端延迟"""
        if partition_point == 0:
            return 0.0
            
        edge_layers = min(partition_point, len(model_profile["layer_complexity"]))
        complexity_sum = sum(model_profile["layer_complexity"][:edge_layers])
        
        base_latency = model_profile["base_latency"]
        edge_latency = (base_latency * complexity_sum / len(model_profile["layer_complexity"]) 
                       / device_profile["compute_power"])
        
        return edge_latency
        
    def _calculate_edge_energy(self, partition_point: int, model_profile: Dict, 
                             device_profile: Dict) -> float:
        """计算边缘端能耗"""
        if partition_point == 0:
            return 0.0
            
        edge_layers = min(partition_point, len(model_profile["layer_complexity"]))
        complexity_sum = sum(model_profile["layer_complexity"][:edge_layers])
        
        base_energy = model_profile["base_energy"]
        edge_energy = (base_energy * complexity_sum / len(model_profile["layer_complexity"]) 
                      * device_profile["energy_efficiency"])
        
        return edge_energy
        
    def _calculate_transmission(self, partition_point: int, model_profile: Dict, 
                              bandwidth: float) -> Tuple[float, float]:
        """计算传输性能"""
        if partition_point == 0:
            # 传输原始输入
            data_size = 0.1  # MB
        elif partition_point >= len(model_profile["data_sizes"]):
            # 传输最终输出
            data_size = 0.01  # MB
        else:
            # 传输中间结果
            data_size = model_profile["data_sizes"][partition_point - 1]
            
        # 传输延迟 (ms)
        transmission_latency = (data_size * 8) / bandwidth  # 转换为Mbps
        
        # 传输能耗 (mJ)
        transmission_energy = data_size * 0.1  # 简化的能耗模型
        
        return transmission_latency, transmission_energy
        
    def _calculate_cloud_latency(self, partition_point: int, model_profile: Dict, 
                               device_profile: Dict) -> float:
        """计算云端延迟"""
        if partition_point >= len(model_profile["layer_complexity"]):
            return 0.0
            
        cloud_layers = len(model_profile["layer_complexity"]) - partition_point
        complexity_sum = sum(model_profile["layer_complexity"][partition_point:])
        
        base_latency = model_profile["base_latency"]
        cloud_latency = (base_latency * complexity_sum / len(model_profile["layer_complexity"]) 
                        / device_profile["compute_power"])
        
        return cloud_latency
        
    def _calculate_cloud_energy(self, partition_point: int, model_profile: Dict, 
                              device_profile: Dict) -> float:
        """计算云端能耗"""
        if partition_point >= len(model_profile["layer_complexity"]):
            return 0.0
            
        cloud_layers = len(model_profile["layer_complexity"]) - partition_point
        complexity_sum = sum(model_profile["layer_complexity"][partition_point:])
        
        base_energy = model_profile["base_energy"]
        cloud_energy = (base_energy * complexity_sum / len(model_profile["layer_complexity"]) 
                       * device_profile["energy_efficiency"])
        
        return cloud_energy
        
    def get_optimal_partition(self, bandwidth: float, model_type: str, 
                            optimization_target: str = "latency") -> int:
        """获取最优划分点"""
        profile = self.model_profiles.get(model_type, self.model_profiles["mobilenet"])
        num_layers = len(profile["layer_complexity"])
        
        best_point = 0
        best_score = float('inf')
        
        for partition_point in range(num_layers + 1):
            latency, energy = self.simulate_performance(partition_point, bandwidth, model_type)
            
            if optimization_target == "latency":
                score = latency
            elif optimization_target == "energy":
                score = energy
            else:  # balanced
                score = latency * 0.7 + energy * 0.3
                
            if score < best_score:
                best_score = score
                best_point = partition_point
                
        return best_point
        
    def analyze_sensitivity(self, model_type: str, bandwidth_range: Tuple[float, float] = (0.5, 100.0)) -> Dict:
        """分析敏感性"""
        profile = self.model_profiles.get(model_type, self.model_profiles["mobilenet"])
        num_layers = len(profile["layer_complexity"])
        
        bandwidths = np.linspace(bandwidth_range[0], bandwidth_range[1], 20)
        results = {
            "bandwidths": bandwidths.tolist(),
            "optimal_points": [],
            "latencies": [],
            "energies": []
        }
        
        for bandwidth in bandwidths:
            optimal_point = self.get_optimal_partition(bandwidth, model_type, "balanced")
            latency, energy = self.simulate_performance(optimal_point, bandwidth, model_type)
            
            results["optimal_points"].append(optimal_point)
            results["latencies"].append(latency)
            results["energies"].append(energy)
            
        return results

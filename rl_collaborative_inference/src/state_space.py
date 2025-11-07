"""
State space definition and normalization
"""
import torch
import numpy as np
import platform

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class StateSpace:
    """State space for reinforcement learning"""
    
    def __init__(self):
        self.state_dim = 29  # Total state dimension
        self.device_state_dim = 7
        self.network_state_dim = 8
        self.task_state_dim = 4
        self.history_state_dim = 6
        self.feature_state_dim = 4
        
        # Normalization factors
        self.max_bandwidth = 100.0  # MB/s
        self.max_latency = 1000.0  # ms
        self.max_feature_size = 50.0  # MB
    
    def get_device_state(self, edge_device_info=None, cloud_device_info=None):
        """
        Get device state
        :param edge_device_info: edge device info dict
        :param cloud_device_info: cloud device info dict
        :return: device state vector [7]
        """
        # Default values if not provided
        if edge_device_info is None:
            if HAS_PSUTIL:
                cpu_util = psutil.cpu_percent() / 100.0
                memory_util = psutil.virtual_memory().percent / 100.0
            else:
                cpu_util = 0.5
                memory_util = 0.5
            edge_device_info = {
                'cpu_util': cpu_util,
                'memory_util': memory_util,
                'battery': 1.0,  # Assume full battery
                'compute_cap': 1.0  # Normalized compute capability
            }
        
        if cloud_device_info is None:
            cloud_device_info = {
                'cpu_util': 0.5,
                'gpu_util': 0.5,
                'memory_util': 0.5
            }
        
        state = [
            edge_device_info.get('cpu_util', 0.5),
            edge_device_info.get('memory_util', 0.5),
            edge_device_info.get('battery', 1.0),
            edge_device_info.get('compute_cap', 1.0),
            cloud_device_info.get('cpu_util', 0.5),
            cloud_device_info.get('gpu_util', 0.5),
            cloud_device_info.get('memory_util', 0.5)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def get_network_state(self, network_info=None):
        """
        Get network state
        :param network_info: network info dict
        :return: network state vector [8]
        """
        if network_info is None:
            network_info = {
                'bandwidth': 10.0,  # MB/s
                'latency': 50.0,  # ms
                'packet_loss': 0.0,
                'signal_strength': 1.0,
                'network_type': 'wifi'  # 0: wifi, 1: lte, 2: 5g
            }
        
        # Normalize bandwidth
        bandwidth_norm = min(network_info.get('bandwidth', 10.0) / self.max_bandwidth, 1.0)
        
        # Normalize latency
        latency_norm = min(network_info.get('latency', 50.0) / self.max_latency, 1.0)
        
        # Network type one-hot encoding
        network_type = network_info.get('network_type', 'wifi')
        if network_type == 'wifi':
            type_onehot = [1, 0, 0]
        elif network_type == 'lte':
            type_onehot = [0, 1, 0]
        else:  # 5g
            type_onehot = [0, 0, 1]
        
        state = [
            bandwidth_norm,
            latency_norm,
            network_info.get('packet_loss', 0.0),
            network_info.get('signal_strength', 1.0)
        ] + type_onehot
        
        return np.array(state, dtype=np.float32)
    
    def get_task_state(self, task_info=None):
        """
        Get task state
        :param task_info: task info dict
        :return: task state vector [4]
        """
        if task_info is None:
            task_info = {
                'input_size': 0.1,  # MB
                'model_complexity': 0.5,
                'task_queue_length': 1,
                'expected_accuracy': 0.9
            }
        
        # Normalize input size (assuming max 10MB)
        input_size_norm = min(task_info.get('input_size', 0.1) / 10.0, 1.0)
        
        # Normalize task queue (assuming max 10 tasks)
        queue_norm = min(task_info.get('task_queue_length', 1) / 10.0, 1.0)
        
        state = [
            input_size_norm,
            task_info.get('model_complexity', 0.5),
            queue_norm,
            task_info.get('expected_accuracy', 0.9)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def get_history_state(self, history_info=None):
        """
        Get history state
        :param history_info: history info dict
        :return: history state vector [6]
        """
        if history_info is None:
            history_info = {
                'last_partition_point': 0.5,
                'last_compression_rate': 0.5,
                'last_latency': 0.5,
                'last_accuracy': 0.9,
                'avg_latency_window': 0.5,
                'avg_accuracy_window': 0.9
            }
        
        # Normalize partition point (assuming max 10 partition points)
        partition_norm = history_info.get('last_partition_point', 5) / 10.0
        
        # Normalize latency
        latency_norm = min(history_info.get('last_latency', 500.0) / self.max_latency, 1.0)
        avg_latency_norm = min(history_info.get('avg_latency_window', 500.0) / self.max_latency, 1.0)
        
        state = [
            partition_norm,
            history_info.get('last_compression_rate', 0.5),
            latency_norm,
            history_info.get('last_accuracy', 0.9),
            avg_latency_norm,
            history_info.get('avg_accuracy_window', 0.9)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def get_feature_state(self, feature_info=None):
        """
        Get feature state
        :param feature_info: feature info dict
        :return: feature state vector [4]
        """
        if feature_info is None:
            feature_info = {
                'feature_size': 1.0,  # MB
                'feature_channels': 256,
                'feature_sparsity': 0.1,
                'compressibility': 0.5
            }
        
        # Normalize feature size
        feature_size_norm = min(feature_info.get('feature_size', 1.0) / self.max_feature_size, 1.0)
        
        # Normalize channels (assuming max 1024 channels)
        channels_norm = min(feature_info.get('feature_channels', 256) / 1024.0, 1.0)
        
        state = [
            feature_size_norm,
            channels_norm,
            feature_info.get('feature_sparsity', 0.1),
            feature_info.get('compressibility', 0.5)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def build_state(self, device_info=None, network_info=None, task_info=None,
                   history_info=None, feature_info=None):
        """
        Build complete state vector
        :return: complete state vector [29]
        """
        device_state = self.get_device_state(
            device_info.get('edge', None) if device_info else None,
            device_info.get('cloud', None) if device_info else None
        )
        network_state = self.get_network_state(network_info)
        task_state = self.get_task_state(task_info)
        history_state = self.get_history_state(history_info)
        feature_state = self.get_feature_state(feature_info)
        
        complete_state = np.concatenate([
            device_state,
            network_state,
            task_state,
            history_state,
            feature_state
        ])
        
        return complete_state.astype(np.float32)


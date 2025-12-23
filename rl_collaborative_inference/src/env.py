"""
Reinforcement learning environment for collaborative inference
"""
import torch
import torch.nn as nn
import numpy as np
import time
import pickle
from typing import Dict, Tuple, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.pruning import PruningManager
from src.model_partition import ModelPartitioner
from src.state_space import StateSpace
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import inference_utils


class CollaborativeInferenceEnv:
    """Environment for collaborative inference with RL"""
    
    def __init__(self, model, dataset, edge_device='cpu', cloud_device='cpu',
                 network_bandwidth=10.0, pruning_type='structured',
                 target_accuracy=0.95, max_latency=1.0,
                 alpha=0.6, beta=0.4):
        """
        :param model: DNN model
        :param dataset: dataset for evaluation
        :param edge_device: edge device ('cpu' or 'cuda')
        :param cloud_device: cloud device ('cpu' or 'cuda')
        :param network_bandwidth: network bandwidth (MB/s)
        :param pruning_type: 'structured' or 'unstructured'
        :param target_accuracy: target accuracy threshold
        :param max_latency: maximum acceptable latency (seconds)
        :param alpha: accuracy weight in reward
        :param beta: latency weight in reward
        """
        self.model = model
        self.dataset = dataset
        self.edge_device = edge_device
        self.cloud_device = cloud_device
        self.network_bandwidth = network_bandwidth
        self.target_accuracy = target_accuracy
        self.max_latency = max_latency
        self.alpha = alpha
        self.beta = beta
        
        # Initialize components
        self.partitioner = ModelPartitioner(model)
        self.pruning_manager = PruningManager(pruning_type=pruning_type)
        self.state_space = StateSpace()
        
        # Get valid partition points
        self.valid_partition_points = self.partitioner.valid_partition_points
        self.num_partition_points = len(self.valid_partition_points)
        
        # History for state
        self.history = {
            'last_partition_point': 0,
            'last_compression_rate': 0.5,
            'last_latency': 0.0,
            'last_accuracy': 0.0,
            'latency_window': [],
            'accuracy_window': []
        }
        
        # Dataset iterator with random sampling
        import random
        self.dataset_indices = list(range(len(dataset)))
        random.shuffle(self.dataset_indices)
        self.current_idx = 0
        self.current_sample = None
        self.current_label = None
        
    def reset(self):
        """Reset environment"""
        # Get new sample using shuffled indices
        idx = self.dataset_indices[self.current_idx]
        self.current_idx = (self.current_idx + 1) % len(self.dataset_indices)
        
        # If we've gone through all samples, reshuffle
        if self.current_idx == 0:
            import random
            random.shuffle(self.dataset_indices)
        
        sample, label = self.dataset[idx]
        self.current_sample = sample.unsqueeze(0) if sample.dim() == 3 else sample
        self.current_label = label
        
        # Reset history
        self.history['latency_window'] = []
        self.history['accuracy_window'] = []
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step
        :param action: action dict with 'partition_point' and 'compression_rate'
        :return: next_state, reward, done, info
        """
        partition_point_idx = action['partition_point']
        compression_rate = action['compression_rate']
        
        # Map partition point index to actual partition point
        if partition_point_idx >= len(self.valid_partition_points):
            partition_point_idx = len(self.valid_partition_points) - 1
        actual_partition_point = self.valid_partition_points[partition_point_idx]
        
        # Partition model
        edge_model, cloud_model = self.partitioner.partition(actual_partition_point)
        edge_model = edge_model.to(self.edge_device)
        cloud_model = cloud_model.to(self.cloud_device)
        
        # Edge inference
        input_data = self.current_sample.to(self.edge_device)
        edge_start_time = time.time()
        
        with torch.no_grad():
            edge_output = edge_model(input_data)
        edge_time = time.time() - edge_start_time
        
        # Prune intermediate feature
        pruned_feature, pruning_info = self.pruning_manager.compress(
            edge_output, compression_rate
        )
        
        # Calculate transmission time
        # Estimate size of pruned feature
        if pruned_feature.is_sparse:
            feature_size_bytes = pruned_feature._values().numel() * 4  # float32
        else:
            feature_size_bytes = pruned_feature.numel() * 4
        
        # Add mask size
        mask_size_bytes = pruning_info['mask'].numel() * 1  # bool = 1 byte
        total_size_bytes = feature_size_bytes + mask_size_bytes
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        transmission_time = total_size_mb / self.network_bandwidth  # seconds
        
        # Cloud recovery and inference
        cloud_start_time = time.time()
        
        with torch.no_grad():
            recovered_feature = self.pruning_manager.decompress(
                pruned_feature, pruning_info, self.cloud_device
            )
            cloud_output = cloud_model(recovered_feature)
        
        cloud_time = time.time() - cloud_start_time
        
        # Calculate total latency
        total_latency = edge_time + transmission_time + cloud_time
        
        # Calculate accuracy
        pred = torch.argmax(cloud_output, dim=1)
        # Ensure label is a tensor and on the same device
        if isinstance(self.current_label, (int, np.integer, torch.Tensor)):
            if isinstance(self.current_label, torch.Tensor):
                label_tensor = self.current_label.to(pred.device)
            else:
                label_tensor = torch.tensor([self.current_label], device=pred.device, dtype=torch.long)
        else:
            label_tensor = torch.tensor([self.current_label], device=pred.device, dtype=torch.long)
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)
        # Ensure same shape
        if pred.shape != label_tensor.shape:
            if label_tensor.numel() == 1:
                label_tensor = label_tensor.expand_as(pred)
        
        # Debug: check if shapes match
        if pred.shape != label_tensor.shape:
            # Try to fix shape mismatch
            if pred.numel() == 1 and label_tensor.numel() == 1:
                pred = pred.view(1)
                label_tensor = label_tensor.view(1)
            elif pred.numel() > 1 and label_tensor.numel() == 1:
                label_tensor = label_tensor.expand_as(pred)
            elif pred.numel() == 1 and label_tensor.numel() > 1:
                pred = pred.expand_as(label_tensor)
        
        accuracy = (pred == label_tensor).float().item()
        
        # Calculate reward
        reward = self._compute_reward(accuracy, total_latency)
        
        # Update history
        self.history['last_partition_point'] = actual_partition_point
        self.history['last_compression_rate'] = compression_rate
        self.history['last_latency'] = total_latency * 1000  # Convert to ms
        self.history['last_accuracy'] = accuracy
        self.history['latency_window'].append(total_latency * 1000)
        self.history['accuracy_window'].append(accuracy)
        
        # Keep window size
        if len(self.history['latency_window']) > 10:
            self.history['latency_window'].pop(0)
        if len(self.history['accuracy_window']) > 10:
            self.history['accuracy_window'].pop(0)
        
        # Get next state
        next_state = self._get_state()
        
        # Done flag (always False for continuous learning)
        done = False
        
        # Info
        info = {
            'latency': total_latency,
            'edge_time': edge_time,
            'transmission_time': transmission_time,
            'cloud_time': cloud_time,
            'accuracy': accuracy,
            'compression_ratio': self.pruning_manager.calculate_compression_ratio(
                edge_output, pruned_feature
            ),
            'partition_point': actual_partition_point,
            'compression_rate': compression_rate
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Get current state"""
        # Device info
        device_info = {
            'edge': {
                'cpu_util': 0.5,
                'memory_util': 0.5,
                'battery': 1.0,
                'compute_cap': 1.0
            },
            'cloud': {
                'cpu_util': 0.3,
                'gpu_util': 0.2,
                'memory_util': 0.4
            }
        }
        
        # Network info
        network_info = {
            'bandwidth': self.network_bandwidth,
            'latency': 50.0,
            'packet_loss': 0.0,
            'signal_strength': 1.0,
            'network_type': 'wifi'
        }
        
        # Task info
        task_info = {
            'input_size': 0.1,
            'model_complexity': 0.5,
            'task_queue_length': 1,
            'expected_accuracy': 0.9
        }
        
        # History info
        history_info = {
            'last_partition_point': self.history['last_partition_point'],
            'last_compression_rate': self.history['last_compression_rate'],
            'last_latency': self.history['last_latency'],
            'last_accuracy': self.history['last_accuracy'],
            'avg_latency_window': np.mean(self.history['latency_window']) if self.history['latency_window'] else 0.0,
            'avg_accuracy_window': np.mean(self.history['accuracy_window']) if self.history['accuracy_window'] else 0.9
        }
        
        # Feature info (estimated)
        feature_info = {
            'feature_size': 1.0,
            'feature_channels': 256,
            'feature_sparsity': 0.1,
            'compressibility': 0.5
        }
        
        state = self.state_space.build_state(
            device_info=device_info,
            network_info=network_info,
            task_info=task_info,
            history_info=history_info,
            feature_info=feature_info
        )
        
        return state
    
    def _compute_reward(self, accuracy, latency):
        """
        Compute reward
        :param accuracy: accuracy [0, 1]
        :param latency: latency in seconds
        :return: reward
        """
        # Accuracy reward
        if accuracy >= self.target_accuracy:
            accuracy_reward = (accuracy - self.target_accuracy) / (1 - self.target_accuracy)
        else:
            # Penalty for low accuracy
            accuracy_reward = -2.0 * (self.target_accuracy - accuracy) / self.target_accuracy
        
        # Latency reward (negative, lower is better)
        latency_norm = min(latency / self.max_latency, 1.0)
        latency_reward = -latency_norm
        
        # Combined reward
        reward = self.alpha * accuracy_reward + self.beta * latency_reward
        
        return reward


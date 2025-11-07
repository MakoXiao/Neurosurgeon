"""
Model partition module for collaborative inference
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils import inference_utils


class ModelPartitioner:
    """Partition DNN model for collaborative inference"""
    
    def __init__(self, model):
        """
        :param model: DNN model to partition
        """
        self.model = model
        self.num_layers = len(model)
        self.valid_partition_points = self._get_valid_partition_points()
    
    def _get_valid_partition_points(self):
        """Get valid partition points (skip ReLU, BatchNorm, Dropout)"""
        valid_points = [0]  # Can partition at the beginning
        for i in range(1, self.num_layers):
            layer = self.model[i - 1]
            # Skip non-computational layers
            if not isinstance(layer, (nn.ReLU, nn.ReLU6, nn.BatchNorm2d, nn.Dropout)):
                valid_points.append(i)
        valid_points.append(self.num_layers)  # Can partition at the end
        return valid_points
    
    def partition(self, partition_point):
        """
        Partition model at given point
        :param partition_point: partition point index
        :return: edge_model, cloud_model
        """
        if partition_point < 0 or partition_point > self.num_layers:
            raise ValueError(f"Invalid partition point: {partition_point}")
        
        edge_model, cloud_model = inference_utils.model_partition(self.model, partition_point)
        return edge_model, cloud_model
    
    def get_partition_info(self, partition_point):
        """
        Get information about partition
        :param partition_point: partition point
        :return: partition info dict
        """
        edge_model, cloud_model = self.partition(partition_point)
        
        # Count parameters
        edge_params = sum(p.numel() for p in edge_model.parameters())
        cloud_params = sum(p.numel() for p in cloud_model.parameters())
        
        return {
            'partition_point': partition_point,
            'edge_params': edge_params,
            'cloud_params': cloud_params,
            'edge_layers': len(edge_model),
            'cloud_layers': len(cloud_model)
        }


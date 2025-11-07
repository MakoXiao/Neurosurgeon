"""
Pruning module for intermediate feature compression
Supports both structured and unstructured pruning with recovery mechanism
"""
import torch
import torch.nn as nn
import numpy as np


class StructuredPruner:
    """Structured pruning: channel-wise pruning"""
    
    @staticmethod
    def prune(feature_tensor, compression_rate):
        """
        Structured pruning: prune channels
        :param feature_tensor: intermediate feature [B, C, H, W]
        :param compression_rate: compression rate [0.1, 1.0]
        :return: pruned_feature, mask, indices
        """
        B, C, H, W = feature_tensor.shape
        
        # Calculate number of channels to keep
        keep_channels = max(1, int(C * compression_rate))
        
        # Calculate channel importance using L2 norm
        # Flatten spatial dimensions and compute norm per channel
        channel_importance = torch.norm(feature_tensor.view(B, C, -1), dim=2)  # [B, C] -> [C] after mean
        channel_importance = channel_importance.mean(dim=0)  # Average over batch
        
        # Select top-k important channels
        _, top_indices = torch.topk(channel_importance, keep_channels)
        top_indices = torch.sort(top_indices)[0]  # Sort to maintain order
        
        # Create mask
        mask = torch.zeros(C, dtype=torch.bool, device=feature_tensor.device)
        mask[top_indices] = True
        
        # Prune
        pruned_feature = feature_tensor[:, mask, :, :]
        
        return pruned_feature, mask, top_indices
    
    @staticmethod
    def recover(pruned_feature, mask, original_channels, device):
        """
        Recover structured pruned feature
        :param pruned_feature: pruned feature [B, C', H, W]
        :param mask: pruning mask [C]
        :param original_channels: original number of channels
        :param device: device
        :return: recovered feature [B, C, H, W]
        """
        B, C_pruned, H, W = pruned_feature.shape
        recovered = torch.zeros(B, original_channels, H, W, device=device)
        
        # Place pruned features back to original positions
        recovered[:, mask, :, :] = pruned_feature
        
        return recovered


class UnstructuredPruner:
    """Unstructured pruning: element-wise pruning"""
    
    @staticmethod
    def prune(feature_tensor, compression_rate):
        """
        Unstructured pruning: prune elements
        :param feature_tensor: intermediate feature [B, C, H, W]
        :param compression_rate: compression rate [0.1, 1.0]
        :return: pruned_feature (sparse), mask, indices
        """
        # Calculate number of elements to keep
        total_elements = feature_tensor.numel()
        keep_elements = max(1, int(total_elements * compression_rate))
        
        # Calculate element importance using absolute value
        importance = torch.abs(feature_tensor.flatten())
        
        # Select top-k important elements
        _, top_indices = torch.topk(importance, keep_elements)
        
        # Create mask
        mask = torch.zeros_like(feature_tensor, dtype=torch.bool)
        mask_flatten = mask.flatten()
        mask_flatten[top_indices] = True
        mask = mask_flatten.reshape(feature_tensor.shape)
        
        # Prune (create sparse tensor)
        pruned_feature = feature_tensor * mask
        
        return pruned_feature, mask, top_indices
    
    @staticmethod
    def recover(pruned_feature, mask, device):
        """
        Recover unstructured pruned feature
        :param pruned_feature: pruned sparse feature [B, C, H, W]
        :param mask: pruning mask [B, C, H, W]
        :param device: device
        :return: recovered feature [B, C, H, W]
        """
        # Unstructured pruning recovery is straightforward
        # since sparse tensor already contains position information
        if pruned_feature.is_sparse:
            recovered = pruned_feature.to_dense()
        else:
            recovered = pruned_feature
        
        return recovered


class PruningManager:
    """Manager for pruning operations"""
    
    def __init__(self, pruning_type='structured'):
        """
        :param pruning_type: 'structured' or 'unstructured'
        """
        self.pruning_type = pruning_type
        if pruning_type == 'structured':
            self.pruner = StructuredPruner()
        else:
            self.pruner = UnstructuredPruner()
    
    def compress(self, feature_tensor, compression_rate):
        """
        Compress intermediate feature
        :param feature_tensor: feature to compress
        :param compression_rate: compression rate [0.1, 1.0]
        :return: pruned_feature, pruning_info
        """
        compression_rate = max(0.1, min(1.0, compression_rate))
        
        if self.pruning_type == 'structured':
            pruned_feature, mask, indices = self.pruner.prune(feature_tensor, compression_rate)
            pruning_info = {
                'mask': mask,
                'indices': indices,
                'original_shape': feature_tensor.shape,
                'pruning_type': 'structured',
                'compression_rate': compression_rate
            }
        else:
            pruned_feature, mask, indices = self.pruner.prune(feature_tensor, compression_rate)
            pruning_info = {
                'mask': mask,
                'indices': indices,
                'original_shape': feature_tensor.shape,
                'pruning_type': 'unstructured',
                'compression_rate': compression_rate
            }
        
        return pruned_feature, pruning_info
    
    def decompress(self, pruned_feature, pruning_info, device):
        """
        Decompress pruned feature
        :param pruned_feature: pruned feature
        :param pruning_info: pruning information
        :param device: device
        :return: recovered feature
        """
        if pruning_info['pruning_type'] == 'structured':
            original_channels = pruning_info['original_shape'][1]
            recovered = self.pruner.recover(
                pruned_feature,
                pruning_info['mask'],
                original_channels,
                device
            )
        else:
            recovered = self.pruner.recover(
                pruned_feature,
                pruning_info['mask'],
                device
            )
        
        return recovered
    
    def calculate_compression_ratio(self, original_tensor, pruned_tensor):
        """
        Calculate compression ratio
        :param original_tensor: original tensor
        :param pruned_tensor: pruned tensor
        :return: compression ratio
        """
        original_size = original_tensor.numel() * 4  # float32 = 4 bytes
        if pruned_tensor.is_sparse:
            pruned_size = pruned_tensor._values().numel() * 4
        else:
            pruned_size = pruned_tensor.numel() * 4
        
        return original_size / max(pruned_size, 1)


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
        :param feature_tensor: intermediate feature [B, C, H, W] or [B, features]
        :param compression_rate: compression rate [0.1, 1.0]
        :return: pruned_feature, mask, indices
        """
        # Handle different feature shapes
        if feature_tensor.dim() == 4:
            # 4D feature: [B, C, H, W]
            B, C, H, W = feature_tensor.shape
            original_shape = (B, C, H, W)
            
            # Calculate number of channels to keep
            keep_channels = max(1, int(C * compression_rate))
            
            # Calculate channel importance using L2 norm
            channel_importance = torch.norm(feature_tensor.view(B, C, -1), dim=2)  # [B, C]
            channel_importance = channel_importance.mean(dim=0)  # [C]
            
            # Select top-k important channels
            _, top_indices = torch.topk(channel_importance, keep_channels)
            top_indices = torch.sort(top_indices)[0]  # Sort to maintain order
            
            # Create mask
            mask = torch.zeros(C, dtype=torch.bool, device=feature_tensor.device)
            mask[top_indices] = True
            
            # Prune
            pruned_feature = feature_tensor[:, mask, :, :]
            
        elif feature_tensor.dim() == 2:
            # 2D feature: [B, features] - prune features directly
            B, num_features = feature_tensor.shape
            original_shape = (B, num_features)
            
            # Calculate number of features to keep
            keep_features = max(1, int(num_features * compression_rate))
            
            # Calculate feature importance using L2 norm
            feature_importance = torch.norm(feature_tensor, dim=0)  # [features]
            
            # Select top-k important features
            _, top_indices = torch.topk(feature_importance, keep_features)
            top_indices = torch.sort(top_indices)[0]  # Sort to maintain order
            
            # Create mask
            mask = torch.zeros(num_features, dtype=torch.bool, device=feature_tensor.device)
            mask[top_indices] = True
            
            # Prune
            pruned_feature = feature_tensor[:, mask]
            
        else:
            # Flatten to 2D and handle
            B = feature_tensor.shape[0]
            feature_flat = feature_tensor.view(B, -1)
            num_features = feature_flat.shape[1]
            
            # Calculate number of features to keep
            keep_features = max(1, int(num_features * compression_rate))
            
            # Calculate feature importance
            feature_importance = torch.norm(feature_flat, dim=0)
            
            # Select top-k important features
            _, top_indices = torch.topk(feature_importance, keep_features)
            top_indices = torch.sort(top_indices)[0]
            
            # Create mask
            mask = torch.zeros(num_features, dtype=torch.bool, device=feature_tensor.device)
            mask[top_indices] = True
            
            # Prune and reshape
            pruned_flat = feature_flat[:, mask]
            # Try to maintain original shape as much as possible
            pruned_feature = pruned_flat.view(B, -1)
        
        return pruned_feature, mask, top_indices
    
    @staticmethod
    def recover(pruned_feature, mask, original_channels, device):
        """
        Recover structured pruned feature
        :param pruned_feature: pruned feature [B, C', H, W] or [B, features]
        :param mask: pruning mask [C] or [features]
        :param original_channels: original number of channels/features
        :param device: device
        :return: recovered feature
        """
        B = pruned_feature.shape[0]
        dtype = pruned_feature.dtype
        
        # Handle different feature dimensions
        if pruned_feature.dim() == 4:
            # 4D feature: [B, C', H, W]
            B, C_pruned, H, W = pruned_feature.shape
            recovered = torch.zeros(B, original_channels, H, W, device=device, dtype=dtype)
            recovered[:, mask, :, :] = pruned_feature
        elif pruned_feature.dim() == 2:
            # 2D feature: [B, features]
            B, num_features_pruned = pruned_feature.shape
            recovered = torch.zeros(B, original_channels, device=device, dtype=dtype)
            recovered[:, mask] = pruned_feature
        else:
            # Flatten and recover
            pruned_flat = pruned_feature.view(B, -1)
            num_features_pruned = pruned_flat.shape[1]
            recovered_flat = torch.zeros(B, original_channels, device=device, dtype=dtype)
            recovered_flat[:, mask] = pruned_flat
            # Try to reshape to original dimensions if possible
            # For now, return flattened version
            recovered = recovered_flat
        
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
            original_shape = pruning_info['original_shape']
            # Get the dimension that was pruned (usually the second dimension)
            if len(original_shape) >= 2:
                original_channels = original_shape[1]
            else:
                original_channels = original_shape[-1] if len(original_shape) > 0 else pruned_feature.shape[1]
            
            recovered = self.pruner.recover(
                pruned_feature,
                pruning_info['mask'],
                original_channels,
                device
            )
            
            # If original shape was different from recovered shape, try to reshape
            if recovered.dim() != len(original_shape):
                # Try to reshape to original shape
                try:
                    recovered = recovered.view(*original_shape)
                except:
                    # If reshaping fails, return as is
                    pass
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


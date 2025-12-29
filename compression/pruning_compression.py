"""
可恢复剪枝压缩机制（创新点二）
包括结构化剪枝和非结构化剪枝，支持云端精确恢复
"""
import torch
import torch.nn as nn
import numpy as np


class PruningCompressor:
    """剪枝压缩器基类"""
    
    def __init__(self, pruning_type='structured'):
        """
        Args:
            pruning_type: 剪枝类型 ('structured' 或 'unstructured')
        """
        self.pruning_type = pruning_type
    
    def compress(self, feature_tensor, compression_rate):
        """
        压缩特征张量
        
        Args:
            feature_tensor: 输入特征 [B, C, H, W]
            compression_rate: 压缩率 [0.1, 1.0]
        
        Returns:
            compressed_data: 压缩后的数据字典
        """
        raise NotImplementedError
    
    def decompress(self, compressed_data, device='cpu'):
        """
        解压缩特征张量
        
        Args:
            compressed_data: 压缩数据字典
            device: 目标设备
        
        Returns:
            recovered_tensor: 恢复后的特征张量
        """
        raise NotImplementedError


class StructuredPruningCompressor(PruningCompressor):
    """结构化剪枝压缩器（通道级剪枝）"""
    
    def __init__(self):
        super(StructuredPruningCompressor, self).__init__(pruning_type='structured')
    
    def compress(self, feature_tensor, compression_rate):
        """
        结构化剪枝压缩
        
        Args:
            feature_tensor: 输入特征 [B, C, H, W]
            compression_rate: 压缩率 [0.1, 1.0]
        
        Returns:
            compressed_data: {
                'pruned_feature': 剪枝后的特征,
                'mask_indices': 保留的通道索引,
                'original_shape': 原始形状,
                'compression_rate': 压缩率,
                'pruning_type': 'structured'
            }
        """
        if len(feature_tensor.shape) == 2:
            # 处理全连接层输出 [B, D]
            return self._compress_linear(feature_tensor, compression_rate)
        
        B, C, H, W = feature_tensor.shape
        
        # 计算保留的通道数
        keep_channels = max(1, int(C * compression_rate))
        
        # 计算每个通道的重要性（使用L2范数）
        channel_importance = torch.norm(
            feature_tensor.view(B, C, -1), 
            dim=(0, 2)
        )  # [C]
        
        # 选择最重要的通道
        _, top_indices = torch.topk(channel_importance, keep_channels)
        top_indices = torch.sort(top_indices)[0]  # 排序以保持顺序
        
        # 剪枝
        pruned_feature = feature_tensor[:, top_indices, :, :]
        
        compressed_data = {
            'pruned_feature': pruned_feature,
            'mask_indices': top_indices,
            'original_shape': feature_tensor.shape,
            'compression_rate': compression_rate,
            'pruning_type': 'structured'
        }
        
        return compressed_data
    
    def _compress_linear(self, feature_tensor, compression_rate):
        """压缩全连接层输出"""
        B, D = feature_tensor.shape
        
        # 计算保留的维度数
        keep_dims = max(1, int(D * compression_rate))
        
        # 计算每个维度的重要性
        importance = torch.abs(feature_tensor).mean(dim=0)  # [D]
        
        # 选择最重要的维度
        _, top_indices = torch.topk(importance, keep_dims)
        top_indices = torch.sort(top_indices)[0]
        
        # 剪枝
        pruned_feature = feature_tensor[:, top_indices]
        
        compressed_data = {
            'pruned_feature': pruned_feature,
            'mask_indices': top_indices,
            'original_shape': feature_tensor.shape,
            'compression_rate': compression_rate,
            'pruning_type': 'structured'
        }
        
        return compressed_data
    
    def decompress(self, compressed_data, device='cpu'):
        """
        恢复结构化剪枝的特征
        
        Args:
            compressed_data: 压缩数据字典
            device: 目标设备
        
        Returns:
            recovered_tensor: 恢复后的特征张量
        """
        pruned_feature = compressed_data['pruned_feature'].to(device)
        mask_indices = compressed_data['mask_indices'].to(device)
        original_shape = compressed_data['original_shape']
        
        if len(original_shape) == 2:
            # 恢复全连接层输出
            B, D = original_shape
            recovered = torch.zeros(B, D, device=device)
            recovered[:, mask_indices] = pruned_feature
        else:
            # 恢复卷积层输出
            B, C, H, W = original_shape
            recovered = torch.zeros(B, C, H, W, device=device)
            recovered[:, mask_indices, :, :] = pruned_feature
        
        return recovered


class UnstructuredPruningCompressor(PruningCompressor):
    """非结构化剪枝压缩器（元素级剪枝）"""
    
    def __init__(self):
        super(UnstructuredPruningCompressor, self).__init__(pruning_type='unstructured')
    
    def compress(self, feature_tensor, compression_rate):
        """
        非结构化剪枝压缩
        
        Args:
            feature_tensor: 输入特征 [B, C, H, W] 或 [B, D]
            compression_rate: 压缩率 [0.1, 1.0]
        
        Returns:
            compressed_data: {
                'pruned_values': 保留的值,
                'mask_indices': 保留的位置索引,
                'original_shape': 原始形状,
                'compression_rate': 压缩率,
                'pruning_type': 'unstructured'
            }
        """
        original_shape = feature_tensor.shape
        
        # 展平张量
        flat_tensor = feature_tensor.flatten()
        total_elements = flat_tensor.numel()
        
        # 计算保留的元素数
        keep_elements = max(1, int(total_elements * compression_rate))
        
        # 计算每个元素的重要性（使用绝对值）
        importance = torch.abs(flat_tensor)
        
        # 选择最重要的元素
        _, top_indices = torch.topk(importance, keep_elements)
        top_indices = torch.sort(top_indices)[0]
        
        # 获取保留的值
        pruned_values = flat_tensor[top_indices]
        
        compressed_data = {
            'pruned_values': pruned_values,
            'mask_indices': top_indices,
            'original_shape': original_shape,
            'compression_rate': compression_rate,
            'pruning_type': 'unstructured'
        }
        
        return compressed_data
    
    def decompress(self, compressed_data, device='cpu'):
        """
        恢复非结构化剪枝的特征
        
        Args:
            compressed_data: 压缩数据字典
            device: 目标设备
        
        Returns:
            recovered_tensor: 恢复后的特征张量
        """
        pruned_values = compressed_data['pruned_values'].to(device)
        mask_indices = compressed_data['mask_indices'].to(device)
        original_shape = compressed_data['original_shape']
        
        # 创建零张量
        total_elements = np.prod(original_shape)
        recovered_flat = torch.zeros(total_elements, device=device)
        
        # 恢复保留的值
        recovered_flat[mask_indices] = pruned_values
        
        # 恢复原始形状
        recovered = recovered_flat.reshape(original_shape)
        
        return recovered


class AdaptivePruningCompressor:
    """自适应剪枝压缩器（根据特征特性选择剪枝策略）"""
    
    def __init__(self):
        self.structured_compressor = StructuredPruningCompressor()
        self.unstructured_compressor = UnstructuredPruningCompressor()
    
    def compute_feature_sparsity(self, feature_tensor):
        """计算特征稀疏度"""
        threshold = 1e-3
        sparse_ratio = (torch.abs(feature_tensor) < threshold).float().mean().item()
        return sparse_ratio
    
    def select_pruning_strategy(self, feature_tensor, compression_rate):
        """
        根据特征特性选择剪枝策略
        
        Args:
            feature_tensor: 输入特征
            compression_rate: 压缩率
        
        Returns:
            pruning_type: 'structured' 或 'unstructured'
        """
        # 计算特征稀疏度
        sparsity = self.compute_feature_sparsity(feature_tensor)
        
        # 如果特征本身就很稀疏，使用非结构化剪枝
        if sparsity > 0.5:
            return 'unstructured'
        
        # 如果压缩率较低（需要大量压缩），使用结构化剪枝
        if compression_rate < 0.3:
            return 'structured'
        
        # 默认使用结构化剪枝
        return 'structured'
    
    def compress(self, feature_tensor, compression_rate, pruning_type='auto'):
        """
        自适应压缩
        
        Args:
            feature_tensor: 输入特征
            compression_rate: 压缩率
            pruning_type: 'auto', 'structured', 或 'unstructured'
        
        Returns:
            compressed_data: 压缩数据字典
        """
        if pruning_type == 'auto':
            pruning_type = self.select_pruning_strategy(feature_tensor, compression_rate)
        
        if pruning_type == 'structured':
            return self.structured_compressor.compress(feature_tensor, compression_rate)
        else:
            return self.unstructured_compressor.compress(feature_tensor, compression_rate)
    
    def decompress(self, compressed_data, device='cpu'):
        """
        解压缩
        
        Args:
            compressed_data: 压缩数据字典
            device: 目标设备
        
        Returns:
            recovered_tensor: 恢复后的特征张量
        """
        pruning_type = compressed_data['pruning_type']
        
        if pruning_type == 'structured':
            return self.structured_compressor.decompress(compressed_data, device)
        else:
            return self.unstructured_compressor.decompress(compressed_data, device)


def compute_compression_ratio(original_tensor, compressed_data):
    """
    计算实际压缩比
    
    Args:
        original_tensor: 原始张量
        compressed_data: 压缩数据
    
    Returns:
        compression_ratio: 压缩比
    """
    original_size = original_tensor.numel() * original_tensor.element_size()
    
    if compressed_data['pruning_type'] == 'structured':
        compressed_size = (
            compressed_data['pruned_feature'].numel() * 
            compressed_data['pruned_feature'].element_size() +
            compressed_data['mask_indices'].numel() * 
            compressed_data['mask_indices'].element_size()
        )
    else:
        compressed_size = (
            compressed_data['pruned_values'].numel() * 
            compressed_data['pruned_values'].element_size() +
            compressed_data['mask_indices'].numel() * 
            compressed_data['mask_indices'].element_size()
        )
    
    compression_ratio = original_size / compressed_size
    return compression_ratio


if __name__ == '__main__':
    print("测试剪枝压缩机制...")
    
    # 测试结构化剪枝
    print("\n=== 测试结构化剪枝 ===")
    structured_compressor = StructuredPruningCompressor()
    
    feature = torch.randn(2, 64, 28, 28)
    print(f"原始特征形状: {feature.shape}")
    
    for compression_rate in [0.3, 0.5, 0.7, 1.0]:
        compressed = structured_compressor.compress(feature, compression_rate)
        recovered = structured_compressor.decompress(compressed)
        
        ratio = compute_compression_ratio(feature, compressed)
        error = torch.norm(feature - recovered) / torch.norm(feature)
        
        print(f"压缩率: {compression_rate:.1f}")
        print(f"  压缩后形状: {compressed['pruned_feature'].shape}")
        print(f"  实际压缩比: {ratio:.2f}x")
        print(f"  恢复误差: {error:.6f}")
    
    # 测试非结构化剪枝
    print("\n=== 测试非结构化剪枝 ===")
    unstructured_compressor = UnstructuredPruningCompressor()
    
    for compression_rate in [0.3, 0.5, 0.7, 1.0]:
        compressed = unstructured_compressor.compress(feature, compression_rate)
        recovered = unstructured_compressor.decompress(compressed)
        
        ratio = compute_compression_ratio(feature, compressed)
        error = torch.norm(feature - recovered) / torch.norm(feature)
        
        print(f"压缩率: {compression_rate:.1f}")
        print(f"  保留元素数: {compressed['pruned_values'].numel()}")
        print(f"  实际压缩比: {ratio:.2f}x")
        print(f"  恢复误差: {error:.6f}")
    
    # 测试自适应剪枝
    print("\n=== 测试自适应剪枝 ===")
    adaptive_compressor = AdaptivePruningCompressor()
    
    # 创建稀疏特征
    sparse_feature = torch.randn(2, 64, 28, 28)
    sparse_feature[sparse_feature.abs() < 0.5] = 0
    
    print(f"稀疏特征稀疏度: {adaptive_compressor.compute_feature_sparsity(sparse_feature):.3f}")
    
    compressed = adaptive_compressor.compress(sparse_feature, 0.5, pruning_type='auto')
    print(f"选择的剪枝类型: {compressed['pruning_type']}")
    
    recovered = adaptive_compressor.decompress(compressed)
    error = torch.norm(sparse_feature - recovered) / torch.norm(sparse_feature)
    print(f"恢复误差: {error:.6f}")


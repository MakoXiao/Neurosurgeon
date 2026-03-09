"""
剪枝/压缩模块:
- 卷积层: 结构化通道剪枝 (按L1范数排序，保留top-k通道)
- 全连接层: 权重剪枝 (按权重绝对值，保留top-k神经元)
"""
import torch
import torch.nn as nn
import copy
import numpy as np


def prune_conv_layer(layer, compression_ratio):
    """
    对Conv2d层进行结构化通道剪枝
    compression_ratio: 保留的通道比例, 如0.5表示保留50%的通道

    Returns:
        pruned_layer: 剪枝后的Conv2d层
        kept_channels: 保留的通道索引
    """
    if compression_ratio >= 1.0:
        return copy.deepcopy(layer), list(range(layer.out_channels))
    if compression_ratio <= 0.0:
        compression_ratio = 0.05  # 最少保留5%

    weight = layer.weight.data  # (out_channels, in_channels, H, W)
    num_channels = weight.shape[0]
    num_keep = max(1, int(num_channels * compression_ratio))

    # 按L1范数排序
    l1_norms = torch.norm(weight.view(num_channels, -1), p=1, dim=1)
    _, indices = torch.sort(l1_norms, descending=True)
    kept_channels = sorted(indices[:num_keep].tolist())

    # 创建新的剪枝后层
    new_weight = weight[kept_channels]
    pruned_layer = nn.Conv2d(
        in_channels=layer.in_channels,
        out_channels=num_keep,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        bias=layer.bias is not None
    )
    pruned_layer.weight.data = new_weight
    if layer.bias is not None:
        pruned_layer.bias.data = layer.bias.data[kept_channels]

    return pruned_layer, kept_channels


def prune_fc_layer(layer, compression_ratio):
    """
    对Linear层进行神经元剪枝
    compression_ratio: 保留的神经元比例

    Returns:
        pruned_layer: 剪枝后的Linear层
        kept_neurons: 保留的神经元索引
    """
    if compression_ratio >= 1.0:
        return copy.deepcopy(layer), list(range(layer.out_features))
    if compression_ratio <= 0.0:
        compression_ratio = 0.05

    weight = layer.weight.data  # (out_features, in_features)
    num_neurons = weight.shape[0]
    num_keep = max(1, int(num_neurons * compression_ratio))

    # 按L1范数排序
    l1_norms = torch.norm(weight, p=1, dim=1)
    _, indices = torch.sort(l1_norms, descending=True)
    kept_neurons = sorted(indices[:num_keep].tolist())

    pruned_layer = nn.Linear(
        in_features=layer.in_features,
        out_features=num_keep,
        bias=layer.bias is not None
    )
    pruned_layer.weight.data = weight[kept_neurons]
    if layer.bias is not None:
        pruned_layer.bias.data = layer.bias.data[kept_neurons]

    return pruned_layer, kept_neurons


def compress_intermediate_feature(feature_tensor, compression_ratio, layer_type="conv"):
    """
    对中间特征进行压缩 (模拟剪枝后传输的效果)

    对于Conv输出: 按通道重要性保留top-k通道
    对于FC输出: 按神经元重要性保留top-k神经元

    Args:
        feature_tensor: 中间特征 (batch, channels, H, W) 或 (batch, features)
        compression_ratio: 保留比例 (0, 1]
        layer_type: "conv" 或 "fc"

    Returns:
        compressed_feature: 压缩后的特征
        kept_indices: 保留的索引
        original_shape: 原始形状 (用于恢复)
    """
    if compression_ratio >= 1.0:
        return feature_tensor.clone(), None, feature_tensor.shape

    original_shape = feature_tensor.shape

    if layer_type == "conv" and len(feature_tensor.shape) == 4:
        # (batch, channels, H, W)
        num_channels = feature_tensor.shape[1]
        num_keep = max(1, int(num_channels * compression_ratio))

        # 按通道的L2范数排序
        channel_norms = torch.norm(
            feature_tensor.view(feature_tensor.shape[0], num_channels, -1),
            p=2, dim=2
        ).mean(dim=0)
        _, indices = torch.sort(channel_norms, descending=True)
        kept_indices = sorted(indices[:num_keep].tolist())

        compressed_feature = feature_tensor[:, kept_indices, :, :]
    else:
        # (batch, features)
        if len(feature_tensor.shape) == 1:
            feature_tensor = feature_tensor.unsqueeze(0)
        num_features = feature_tensor.shape[-1]
        num_keep = max(1, int(num_features * compression_ratio))

        abs_vals = torch.abs(feature_tensor).mean(dim=0)
        _, indices = torch.sort(abs_vals, descending=True)
        kept_indices = sorted(indices[:num_keep].tolist())

        compressed_feature = feature_tensor[..., kept_indices]

    return compressed_feature, kept_indices, original_shape


def restore_intermediate_feature(compressed_feature, kept_indices, original_shape):
    """
    在云端恢复压缩后的中间特征 (用零填充被剪枝的部分)

    Args:
        compressed_feature: 压缩后的特征
        kept_indices: 保留的索引
        original_shape: 原始形状

    Returns:
        restored_feature: 恢复后的特征 (与原始形状相同)
    """
    if kept_indices is None:
        return compressed_feature

    restored = torch.zeros(original_shape)

    if len(original_shape) == 4:
        # Conv特征
        restored[:, kept_indices, :, :] = compressed_feature
    else:
        # FC特征
        restored[..., kept_indices] = compressed_feature

    return restored


def estimate_accuracy_after_compression(model, layer_profiles, partition_point,
                                         compression_ratio, base_accuracy=0.85,
                                         input_shape=(1, 3, 224, 224)):
    """
    估算给定压缩率下的模型准确率

    基于剪枝文献的经验模型:
    - 适度剪枝(0.6-1.0)对准确率影响很小 (<2%)
    - 激进剪枝(<0.3)才会造成显著准确率下降
    - 模型后端层(高级语义特征)对压缩更鲁棒
    - 卷积层中间层的结构化剪枝影响较小

    使用 sigmoid-like 衰减: acc = base * (1 - k * sigmoid(-s*(cr - c0)))
    k: 最大准确率下降幅度
    s: 衰减陡度
    c0: 衰减中心点

    Args:
        model: DNN模型
        layer_profiles: 层profiling信息
        partition_point: 划分点
        compression_ratio: 压缩率
        base_accuracy: 不压缩时的基准准确率

    Returns:
        estimated_accuracy: 估算准确率
    """
    if partition_point == 0 or partition_point >= len(layer_profiles):
        return base_accuracy

    if compression_ratio >= 1.0:
        return base_accuracy

    relative_position = partition_point / len(layer_profiles)
    layer_info = layer_profiles[partition_point - 1]

    # 基于层位置和类型的参数化
    # 后端层对压缩更鲁棒, 前端层更敏感
    # 参考文献: 结构化剪枝在适度压缩率(0.3-0.6)下准确率下降通常<2%
    if layer_info["layer_type"] == "conv":
        # 卷积层: 结构化通道剪枝, 中后端层对适度压缩鲁棒
        k = 0.03 + 0.05 * (1 - relative_position)  # 最大下降 3%-8%
        s = 12.0   # 较陡, 只在极低CR时快速衰减
        c0 = 0.20  # 在CR=0.2左右才开始显著下降
    else:
        # 全连接层: 位于模型后端, 对适度压缩鲁棒
        k = 0.02 + 0.04 * (1 - relative_position)  # 最大下降 2%-6%
        s = 14.0
        c0 = 0.15

    # Sigmoid衰减: CR高时接近0(不影响), CR低时接近k(最大影响)
    sigmoid_val = 1.0 / (1.0 + np.exp(s * (compression_ratio - c0)))
    acc_drop = k * sigmoid_val
    estimated_accuracy = base_accuracy * (1.0 - acc_drop)

    # 轻微噪声
    noise = np.random.normal(0, 0.001)
    estimated_accuracy = np.clip(estimated_accuracy + noise, 0.0, base_accuracy)

    return estimated_accuracy

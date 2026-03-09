"""
Model Profiler: 对DNN模型的每一层进行性能分析
- 每层推理延迟 (edge/cloud)
- 每层输出中间特征大小
- 每层类型标记 (Conv2d / Linear / other)
"""
import torch
import torch.nn as nn
import time
import pickle
import copy
import numpy as np


def get_layer_type(layer):
    """判断层类型: conv / fc / other"""
    if isinstance(layer, nn.Conv2d):
        return "conv"
    elif isinstance(layer, nn.Linear):
        return "fc"
    else:
        return "other"


def is_compressible(layer):
    """判断该层是否可压缩（有权重参数的层）"""
    return isinstance(layer, (nn.Conv2d, nn.Linear))


def measure_layer_latency(layer, input_data, device="cpu", num_runs=30):
    """测量单层推理延迟 (ms)"""
    layer = layer.to(device)
    input_data = input_data.to(device)

    with torch.no_grad():
        for _ in range(5):
            _ = layer(input_data)

        times = []
        for _ in range(num_runs):
            x = input_data.clone()
            if device == "cuda":
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = layer(x)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
            else:
                start = time.perf_counter()
                _ = layer(x)
                end = time.perf_counter()
                times.append((end - start) * 1000)

    return np.mean(times)


def get_intermediate_size_bytes(tensor):
    """获取中间特征的序列化字节大小"""
    return len(pickle.dumps(tensor))


def get_intermediate_size_mb(tensor):
    """获取中间特征的大小 (MB), 基于浮点数数量"""
    total_num = 1
    for s in tensor.shape:
        total_num *= s
    return total_num * 4 / 1e6


def profile_model(model, input_shape=(1, 3, 224, 224), device="cpu",
                  edge_device="cpu", cloud_device="cpu"):
    """
    对模型每一层进行profiling, 返回每层的详细信息列表

    Returns:
        layer_profiles: list of dict, 每个dict包含:
            - index: 层索引 (0-based)
            - layer: 层对象
            - layer_name: 层名称
            - layer_type: conv/fc/other
            - is_compressible: 是否可压缩
            - edge_latency_ms: 边端推理延迟
            - cloud_latency_ms: 云端推理延迟
            - output_shape: 输出形状
            - output_size_mb: 输出大小(MB)
            - output_size_bytes: 序列化字节大小
    """
    layer_profiles = []
    x = torch.rand(size=input_shape, requires_grad=False)

    idx = 0
    for layer in model:
        layer_name = layer.__class__.__name__
        layer_type = get_layer_type(layer)
        compressible = is_compressible(layer)

        edge_lat = measure_layer_latency(layer, x, device=edge_device)
        cloud_lat = measure_layer_latency(layer, x, device=cloud_device)

        with torch.no_grad():
            x = layer.to("cpu")(x.to("cpu"))

        output_size_mb = get_intermediate_size_mb(x)
        output_size_bytes = get_intermediate_size_bytes(x)

        layer_profiles.append({
            "index": idx,
            "layer": layer,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "is_compressible": compressible,
            "edge_latency_ms": edge_lat,
            "cloud_latency_ms": cloud_lat,
            "output_shape": tuple(x.shape),
            "output_size_mb": output_size_mb,
            "output_size_bytes": output_size_bytes,
        })
        idx += 1

    return layer_profiles


def compute_partition_latency(layer_profiles, partition_point, compression_ratio,
                              bandwidth_mbps, edge_device="cpu", cloud_device="cpu"):
    """
    计算给定划分点和压缩率下的总推理延迟

    Args:
        layer_profiles: profile_model的输出
        partition_point: 划分点 (0=全云端, len=全本地)
        compression_ratio: 压缩率 (0,1], 1表示不压缩
        bandwidth_mbps: 带宽 (Mbps)

    Returns:
        total_latency_ms, edge_latency_ms, transmission_latency_ms, cloud_latency_ms
    """
    num_layers = len(layer_profiles)
    bandwidth_mbytes = bandwidth_mbps / 8.0  # MB/s

    # 边端推理延迟: 前partition_point层
    edge_latency = 0.0
    for i in range(partition_point):
        edge_latency += layer_profiles[i]["edge_latency_ms"]

    # 传输延迟
    if partition_point == 0:
        # 传输原始输入
        input_size_mb = 1 * 3 * 224 * 224 * 4 / 1e6
        transmission_size_mb = input_size_mb * compression_ratio
    elif partition_point >= num_layers:
        # 全本地, 无传输
        transmission_size_mb = 0.0
    else:
        # 传输中间特征 (压缩后)
        transmission_size_mb = layer_profiles[partition_point - 1]["output_size_mb"] * compression_ratio

    if bandwidth_mbytes > 0 and transmission_size_mb > 0:
        transmission_latency = (transmission_size_mb / bandwidth_mbytes) * 1000  # ms
    else:
        transmission_latency = 0.0

    # 云端推理延迟: partition_point之后的层
    cloud_latency = 0.0
    for i in range(partition_point, num_layers):
        cloud_latency += layer_profiles[i]["cloud_latency_ms"]

    total_latency = edge_latency + transmission_latency + cloud_latency
    return total_latency, edge_latency, transmission_latency, cloud_latency

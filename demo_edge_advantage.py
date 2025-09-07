#!/usr/bin/env python3
"""
演示云边协同优势的脚本
通过模拟不同的网络条件和设备性能来展示划分策略的变化
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import inference_utils
from deployment import neuron_surgeon_deployment
from predictor import predictor_utils
import pickle

def simulate_edge_advantage():
    """
    模拟不同场景下的云边协同优势
    """
    print("=" * 80)
    print("Neurosurgeon 云边协同优势演示")
    print("=" * 80)
    
    # 测试不同的模型
    models = ["alex_net", "vgg_net", "le_net", "mobile_net"]
    
    # 模拟不同的网络条件 (MB/s)
    network_conditions = [
        {"name": "4G网络", "bandwidth": 10, "description": "移动4G网络环境"},
        {"name": "3G网络", "bandwidth": 2, "description": "移动3G网络环境"},
        {"name": "WiFi网络", "bandwidth": 50, "description": "家庭WiFi环境"},
        {"name": "低带宽", "bandwidth": 0.5, "description": "网络拥堵环境"},
    ]
    
    for model_name in models:
        print(f"\n🔍 测试模型: {model_name.upper()}")
        print("-" * 60)
        
        model = inference_utils.get_dnn_model(model_name)
        
        for condition in network_conditions:
            print(f"\n📡 网络条件: {condition['name']} ({condition['bandwidth']} MB/s)")
            print(f"   描述: {condition['description']}")
            
            # 显示详细的划分过程
            partition_point = neuron_surgeon_deployment(
                model, 
                network_type="wifi", 
                define_speed=condition['bandwidth'], 
                show=True  # 显示详细过程
            )
            
            # 分析结果
            if partition_point == 0:
                print("   🎯 策略: 全部云端执行 (边缘端计算能力不足或网络带宽充足)")
            elif partition_point == len(model):
                print("   🎯 策略: 全部边缘端执行 (网络带宽严重不足)")
            else:
                print(f"   🎯 策略: 云边协同 (第{partition_point}层后划分)")
                print("   ✅ 优势: 平衡了计算延迟和传输延迟")
            
            print()

def compare_strategies():
    """
    对比不同策略的性能
    """
    print("\n" + "=" * 80)
    print("策略对比分析")
    print("=" * 80)
    
    model_name = "vgg_net"
    model = inference_utils.get_dnn_model(model_name)
    
    # 模拟一个中等带宽的网络环境
    bandwidth = 5  # MB/s
    
    print(f"模型: {model_name}")
    print(f"网络带宽: {bandwidth} MB/s")
    print("-" * 60)
    
    # 策略1: 全部云端执行
    print("策略1: 全部云端执行")
    edge_model, cloud_model = inference_utils.model_partition(model, 0)
    x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)
    
    # 计算传输时间 (原始输入数据)
    input_size = len(pickle.dumps(x))
    transmission_time = input_size / (bandwidth * 1024 * 1024) * 1000  # ms
    
    # 预测云端计算时间
    predictor_dict = {}
    cloud_lat = predictor_utils.predict_model_latency(x, cloud_model, device="cloud", predictor_dict=predictor_dict)
    total_time_cloud = transmission_time + cloud_lat
    
    print(f"  传输时间: {transmission_time:.2f} ms")
    print(f"  云端计算时间: {cloud_lat:.2f} ms")
    print(f"  总时间: {total_time_cloud:.2f} ms")
    
    # 策略2: 云边协同
    print("\n策略2: 云边协同")
    partition_point = neuron_surgeon_deployment(model, "wifi", bandwidth, show=False)
    edge_model, cloud_model = inference_utils.model_partition(model, partition_point)
    
    # 边缘端计算
    edge_lat = predictor_utils.predict_model_latency(x, edge_model, device="edge", predictor_dict=predictor_dict)
    edge_output = edge_model(x)
    
    # 传输中间结果
    output_size = len(pickle.dumps(edge_output))
    transmission_time = output_size / (bandwidth * 1024 * 1024) * 1000  # ms
    
    # 云端计算
    cloud_lat = predictor_utils.predict_model_latency(edge_output, cloud_model, device="cloud", predictor_dict=predictor_dict)
    total_time_collaborative = edge_lat + transmission_time + cloud_lat
    
    print(f"  边缘端计算时间: {edge_lat:.2f} ms")
    print(f"  传输时间: {transmission_time:.2f} ms")
    print(f"  云端计算时间: {cloud_lat:.2f} ms")
    print(f"  总时间: {total_time_collaborative:.2f} ms")
    
    # 性能提升分析
    improvement = (total_time_cloud - total_time_collaborative) / total_time_cloud * 100
    print(f"\n📊 性能分析:")
    print(f"  云边协同相比全部云端执行提升: {improvement:.1f}%")
    
    if improvement > 0:
        print("  ✅ 云边协同策略更优")
    else:
        print("  ❌ 全部云端执行更优")

def demonstrate_edge_computing_benefits():
    """
    演示边缘计算的优势场景
    """
    print("\n" + "=" * 80)
    print("边缘计算优势场景演示")
    print("=" * 80)
    
    # 场景1: 网络延迟敏感应用
    print("场景1: 实时视频分析 (低延迟要求)")
    print("  需求: 延迟 < 50ms")
    print("  网络: 3G (2 MB/s)")
    print("  结果: 云边协同可以满足实时性要求")
    
    # 场景2: 隐私保护
    print("\n场景2: 隐私敏感数据处理")
    print("  需求: 敏感数据不离开本地")
    print("  策略: 在边缘端处理敏感层，云端处理非敏感层")
    print("  结果: 既保护隐私又利用云端计算能力")
    
    # 场景3: 网络不稳定
    print("\n场景3: 网络不稳定环境")
    print("  需求: 在网络中断时仍能提供基本服务")
    print("  策略: 边缘端保留关键计算能力")
    print("  结果: 提高系统可靠性")

if __name__ == "__main__":
    try:
        # 演示云边协同优势
        simulate_edge_advantage()
        
        # 对比不同策略
        compare_strategies()
        
        # 演示边缘计算优势场景
        demonstrate_edge_computing_benefits()
        
        print("\n" + "=" * 80)
        print("总结:")
        print("1. 在网络带宽较低时，云边协同策略明显优于全部云端执行")
        print("2. 边缘计算可以减少数据传输，降低延迟")
        print("3. 智能划分策略能够根据网络条件动态调整")
        print("4. 云边协同在实时性、隐私保护、可靠性方面都有优势")
        print("=" * 80)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

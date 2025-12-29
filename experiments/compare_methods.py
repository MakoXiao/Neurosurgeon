"""
对比实验脚本
对比不同方法和创新点的性能
"""
import torch
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.caltech101_loader import get_caltech101_dataloaders
from environment.collaborative_env import CollaborativeInferenceEnv
from rl_agent.hybrid_ppo import HybridPPO


class BaselineMethod:
    """基线方法"""
    
    @staticmethod
    def all_edge(env, input_data, label):
        """全边缘推理"""
        return env.evaluate_baseline(input_data, label, 'all_edge')
    
    @staticmethod
    def all_cloud(env, input_data, label):
        """全云端推理"""
        return env.evaluate_baseline(input_data, label, 'all_cloud')
    
    @staticmethod
    def fixed_partition(env, input_data, label, partition_point=3, compression_rate=0.5):
        """固定分割点和压缩率"""
        state = env.reset(input_data, label)
        action = (partition_point, compression_rate)
        _, _, _, info = env.step(action)
        return info
    
    @staticmethod
    def neurosurgeon(env, input_data, label, bandwidth):
        """
        Neurosurgeon方法：基于带宽选择最优分割点，不压缩
        """
        # 简化的Neurosurgeon：根据带宽选择分割点
        if bandwidth > 200:
            partition_point = 1  # 高带宽，早分割
        elif bandwidth > 100:
            partition_point = 3  # 中带宽，中间分割
        else:
            partition_point = 5  # 低带宽，晚分割
        
        state = env.reset(input_data, label)
        action = (partition_point, 1.0)  # 不压缩
        _, _, _, info = env.step(action)
        return info


def evaluate_method(env, test_loader, method_name, method_func, 
                   num_samples=200, **kwargs):
    """
    评估方法
    
    Args:
        env: 环境
        test_loader: 测试数据加载器
        method_name: 方法名称
        method_func: 方法函数
        num_samples: 测试样本数
        **kwargs: 方法参数
    
    Returns:
        results: 评估结果
    """
    print(f"\n评估方法: {method_name}")
    
    total_latency = 0
    total_accuracy = 0
    latencies = []
    accuracies = []
    
    sample_count = 0
    
    for inputs, labels in tqdm(test_loader, desc=method_name):
        if sample_count >= num_samples:
            break
        
        info = method_func(env, inputs, labels, **kwargs)
        
        total_latency += info['total_latency']
        total_accuracy += info['accuracy']
        latencies.append(info['total_latency'])
        accuracies.append(info['accuracy'])
        
        sample_count += 1
    
    avg_latency = total_latency / sample_count
    avg_accuracy = total_accuracy / sample_count
    
    results = {
        'method': method_name,
        'avg_latency': avg_latency,
        'avg_accuracy': avg_accuracy,
        'std_latency': np.std(latencies),
        'std_accuracy': np.std(accuracies),
        'latencies': latencies,
        'accuracies': accuracies
    }
    
    print(f"  平均时延: {avg_latency:.2f} ± {results['std_latency']:.2f} ms")
    print(f"  平均准确率: {avg_accuracy:.4f} ± {results['std_accuracy']:.4f}")
    
    return results


def evaluate_rl_agent(env, agent, test_loader, num_samples=200):
    """
    评估RL智能体
    
    Args:
        env: 环境
        agent: RL智能体
        test_loader: 测试数据加载器
        num_samples: 测试样本数
    
    Returns:
        results: 评估结果
    """
    print(f"\n评估方法: RL Agent (Hybrid PPO)")
    
    total_latency = 0
    total_accuracy = 0
    latencies = []
    accuracies = []
    partition_points = []
    compression_rates = []
    
    sample_count = 0
    
    for inputs, labels in tqdm(test_loader, desc='RL Agent'):
        if sample_count >= num_samples:
            break
        
        state = env.reset(inputs, labels)
        
        # 使用确定性策略
        partition_point, compression_rate = agent.select_action(state, deterministic=True)
        action = (partition_point, compression_rate)
        
        _, _, _, info = env.step(action)
        
        total_latency += info['total_latency']
        total_accuracy += info['accuracy']
        latencies.append(info['total_latency'])
        accuracies.append(info['accuracy'])
        partition_points.append(partition_point)
        compression_rates.append(compression_rate)
        
        sample_count += 1
    
    avg_latency = total_latency / sample_count
    avg_accuracy = total_accuracy / sample_count
    
    results = {
        'method': 'RL Agent (Hybrid PPO)',
        'avg_latency': avg_latency,
        'avg_accuracy': avg_accuracy,
        'std_latency': np.std(latencies),
        'std_accuracy': np.std(accuracies),
        'avg_partition_point': np.mean(partition_points),
        'avg_compression_rate': np.mean(compression_rates),
        'latencies': latencies,
        'accuracies': accuracies,
        'partition_points': partition_points,
        'compression_rates': compression_rates
    }
    
    print(f"  平均时延: {avg_latency:.2f} ± {results['std_latency']:.2f} ms")
    print(f"  平均准确率: {avg_accuracy:.4f} ± {results['std_accuracy']:.4f}")
    print(f"  平均分割点: {results['avg_partition_point']:.2f}")
    print(f"  平均压缩率: {results['avg_compression_rate']:.3f}")
    
    return results


def compare_all_methods(model_name, data_dir, checkpoint_dir, rl_agent_dir,
                       num_samples=200, device='cuda'):
    """
    对比所有方法
    
    Args:
        model_name: 模型名称
        data_dir: 数据目录
        checkpoint_dir: 模型检查点目录
        rl_agent_dir: RL智能体目录
        num_samples: 测试样本数
        device: 设备
    
    Returns:
        all_results: 所有方法的结果
    """
    print(f"\n{'='*60}")
    print(f"对比实验: {model_name}")
    print(f"{'='*60}\n")
    
    # 加载数据
    print("加载数据...")
    _, test_loader, num_classes = get_caltech101_dataloaders(
        data_dir, batch_size=1, num_workers=2
    )
    
    # 创建环境
    print("创建环境...")
    env = CollaborativeInferenceEnv(
        model_name=model_name,
        num_classes=num_classes,
        edge_device='cpu',
        cloud_device=device,
        bandwidth=100.0,
        network_latency=50.0,
        target_accuracy=0.85,
        max_latency=1000.0
    )
    
    # 加载模型权重
    model_checkpoint = os.path.join(checkpoint_dir, model_name, 'best_model.pth')
    if os.path.exists(model_checkpoint):
        print(f"加载模型权重: {model_checkpoint}")
        
        # 先将所有模型移到CPU，然后加载权重
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
        
        env.model = env.model.to('cpu')
        env.edge.model = env.edge.model.to('cpu')
        env.cloud.model = env.cloud.model.to('cpu')
        
        env.model.load_state_dict(checkpoint['model_state_dict'])
        env.edge.model.load_state_dict(checkpoint['model_state_dict'])
        env.cloud.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 将云端模型移到GPU（边缘模型保持在CPU）
        env.cloud.model = env.cloud.model.to(device)
        env.cloud.device = device
        
        print(f"  ✅ 模型已加载并配置到正确设备")
        print(f"  边缘设备: {env.edge.device}")
        print(f"  云端设备: {env.cloud.device}")
    
    all_results = {}
    
    # 1. 全边缘推理
    results = evaluate_method(
        env, test_loader, 'All Edge', 
        BaselineMethod.all_edge, num_samples
    )
    all_results['all_edge'] = results
    
    # 2. 全云端推理
    results = evaluate_method(
        env, test_loader, 'All Cloud',
        BaselineMethod.all_cloud, num_samples
    )
    all_results['all_cloud'] = results
    
    # 3. Neurosurgeon (固定分割，不压缩)
    results = evaluate_method(
        env, test_loader, 'Neurosurgeon',
        BaselineMethod.neurosurgeon, num_samples,
        bandwidth=env.bandwidth
    )
    all_results['neurosurgeon'] = results
    
    # 4. 固定分割点 + 固定压缩率 (0.3)
    results = evaluate_method(
        env, test_loader, 'Fixed (sp=3, cr=0.3)',
        BaselineMethod.fixed_partition, num_samples,
        partition_point=3, compression_rate=0.3
    )
    all_results['fixed_0.3'] = results
    
    # 5. 固定分割点 + 固定压缩率 (0.5)
    results = evaluate_method(
        env, test_loader, 'Fixed (sp=3, cr=0.5)',
        BaselineMethod.fixed_partition, num_samples,
        partition_point=3, compression_rate=0.5
    )
    all_results['fixed_0.5'] = results
    
    # 6. 固定分割点 + 固定压缩率 (0.7)
    results = evaluate_method(
        env, test_loader, 'Fixed (sp=3, cr=0.7)',
        BaselineMethod.fixed_partition, num_samples,
        partition_point=3, compression_rate=0.7
    )
    all_results['fixed_0.7'] = results
    
    # 7. RL智能体 (我们的方法)
    agent_path = os.path.join(rl_agent_dir, f'rl_agent_{model_name}', 'best_agent.pth')
    if os.path.exists(agent_path):
        print(f"\n加载RL智能体: {agent_path}")
        
        state_dim = 29
        num_partition_points = len(env.model.get_split_points())
        
        agent = HybridPPO(
            state_dim=state_dim,
            num_partition_points=num_partition_points,
            device=device
        )
        agent.load(agent_path)
        
        results = evaluate_rl_agent(env, agent, test_loader, num_samples)
        all_results['rl_agent'] = results
    else:
        print(f"\n警告: 未找到RL智能体 {agent_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='对比实验')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'vgg11', 'mobilenetv2', 'alexnet', 'all'],
                       help='模型名称')
    parser.add_argument('--data_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101',
                       help='数据目录')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/checkpoints',
                       help='模型检查点目录')
    parser.add_argument('--rl_agent_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/rl_agents',
                       help='RL智能体目录')
    parser.add_argument('--save_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/results',
                       help='结果保存目录')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='测试样本数')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print(f"\n对比实验配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  检查点目录: {args.checkpoint_dir}")
    print(f"  RL智能体目录: {args.rl_agent_dir}")
    print(f"  结果保存目录: {args.save_dir}")
    print(f"  测试样本数: {args.num_samples}")
    print(f"  设备: {args.device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 对比实验
    if args.model == 'all':
        models = ['resnet18', 'vgg11', 'mobilenetv2', 'alexnet']
    else:
        models = [args.model]
    
    all_model_results = {}
    
    for model_name in models:
        results = compare_all_methods(
            model_name=model_name,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            rl_agent_dir=args.rl_agent_dir,
            num_samples=args.num_samples,
            device=args.device
        )
        
        all_model_results[model_name] = results
        
        # 保存单个模型的结果
        result_path = os.path.join(args.save_dir, f'{model_name}_results.json')
        with open(result_path, 'w') as f:
            # 转换numpy类型为Python类型
            results_serializable = {}
            for method, res in results.items():
                results_serializable[method] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else 
                        float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in res.items()
                }
            json.dump(results_serializable, f, indent=4)
        
        print(f"\n结果已保存到: {result_path}")
    
    # 保存所有模型的结果
    summary_path = os.path.join(args.save_dir, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        summary = {}
        for model_name, results in all_model_results.items():
            summary[model_name] = {
                method: {
                    'avg_latency': float(res['avg_latency']),
                    'avg_accuracy': float(res['avg_accuracy']),
                    'std_latency': float(res['std_latency']),
                    'std_accuracy': float(res['std_accuracy'])
                }
                for method, res in results.items()
            }
        json.dump(summary, f, indent=4)
    
    # 打印总结
    print("\n" + "="*80)
    print("对比实验总结")
    print("="*80)
    
    for model_name, results in all_model_results.items():
        print(f"\n模型: {model_name}")
        print("-" * 80)
        print(f"{'方法':<25s} {'时延(ms)':<15s} {'准确率':<15s}")
        print("-" * 80)
        
        for method, res in results.items():
            latency_str = f"{res['avg_latency']:.2f} ± {res['std_latency']:.2f}"
            accuracy_str = f"{res['avg_accuracy']:.4f} ± {res['std_accuracy']:.4f}"
            print(f"{res['method']:<25s} {latency_str:<15s} {accuracy_str:<15s}")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()


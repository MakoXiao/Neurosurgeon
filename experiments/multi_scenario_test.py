"""
多场景对比实验
测试不同网络条件下的性能
"""
import torch
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.caltech101_loader import get_caltech101_dataloaders
from environment.collaborative_env import CollaborativeInferenceEnv
from rl_agent.hybrid_ppo import HybridPPO


# 定义多种网络场景
NETWORK_SCENARIOS = {
    'high_bandwidth': {
        'bandwidth': 100.0,   # MB/s
        'latency': 20.0,      # ms
        'name': 'High Bandwidth (100MB/s, 20ms)'
    },
    'medium_bandwidth': {
        'bandwidth': 50.0,
        'latency': 50.0,
        'name': 'Medium Bandwidth (50MB/s, 50ms)'
    },
    'low_bandwidth': {
        'bandwidth': 20.0,
        'latency': 100.0,
        'name': 'Low Bandwidth (20MB/s, 100ms)'
    },
    'very_low_bandwidth': {
        'bandwidth': 10.0,
        'latency': 200.0,
        'name': 'Very Low Bandwidth (10MB/s, 200ms)'
    },
    'edge_network': {
        'bandwidth': 5.0,
        'latency': 300.0,
        'name': 'Edge Network (5MB/s, 300ms)'
    }
}


def test_scenario(env, test_loader, rl_agent, scenario_name, scenario_config, num_samples=500):
    """测试单个网络场景"""
    print(f"\n{'='*80}")
    print(f"测试场景: {scenario_config['name']}")
    print(f"{'='*80}")
    
    # 更新环境的网络参数
    env.bandwidth = scenario_config['bandwidth']
    env.network_latency = scenario_config['latency']
    
    results = {
        'all_edge': {'latencies': [], 'accuracies': []},
        'all_cloud': {'latencies': [], 'accuracies': []},
        'rl_agent': {'latencies': [], 'accuracies': [], 'partition_points': [], 'compression_rates': []}
    }
    
    sample_count = 0
    
    for inputs, labels in tqdm(test_loader, desc=f"{scenario_name}"):
        if sample_count >= num_samples:
            break
        
        # 1. All Edge
        info = env.evaluate_baseline(inputs, labels, 'all_edge')
        results['all_edge']['latencies'].append(info['total_latency'])
        results['all_edge']['accuracies'].append(info['accuracy'])
        
        # 2. All Cloud
        info = env.evaluate_baseline(inputs, labels, 'all_cloud')
        results['all_cloud']['latencies'].append(info['total_latency'])
        results['all_cloud']['accuracies'].append(info['accuracy'])
        
        # 3. RL Agent
        state = env.reset(inputs, labels)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(rl_agent.device)
        
        with torch.no_grad():
            partition_point, compression_rate, _ = rl_agent.actor.sample_action(state_tensor)
            partition_point = partition_point.cpu().item()
            compression_rate = compression_rate.cpu().item()
        
        action = (partition_point, compression_rate)
        _, _, _, info = env.step(action)
        
        results['rl_agent']['latencies'].append(info['total_latency'])
        results['rl_agent']['accuracies'].append(info['accuracy'])
        results['rl_agent']['partition_points'].append(partition_point)
        results['rl_agent']['compression_rates'].append(compression_rate)
        
        sample_count += 1
    
    # 计算统计数据
    summary = {}
    for method in results:
        summary[method] = {
            'avg_latency': np.mean(results[method]['latencies']),
            'std_latency': np.std(results[method]['latencies']),
            'avg_accuracy': np.mean(results[method]['accuracies']),
            'std_accuracy': np.std(results[method]['accuracies']),
            'latencies': results[method]['latencies'],
            'accuracies': results[method]['accuracies']
        }
        
        if method == 'rl_agent':
            summary[method]['avg_partition_point'] = np.mean(results[method]['partition_points'])
            summary[method]['avg_compression_rate'] = np.mean(results[method]['compression_rates'])
            summary[method]['partition_points'] = results[method]['partition_points']
            summary[method]['compression_rates'] = results[method]['compression_rates']
    
    # 打印结果
    print(f"\n{'方法':<20} {'时延(ms)':<20} {'准确率':<20}")
    print('-'*80)
    for method, stats in summary.items():
        print(f"{method:<20} {stats['avg_latency']:>8.2f}±{stats['std_latency']:>5.2f}    "
              f"{stats['avg_accuracy']:>8.4f}±{stats['std_accuracy']:>6.4f}")
    
    if 'rl_agent' in summary:
        print(f"\nRL Agent平均分割点: {summary['rl_agent']['avg_partition_point']:.2f}")
        print(f"RL Agent平均压缩率: {summary['rl_agent']['avg_compression_rate']:.3f}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='多场景对比实验')
    parser.add_argument('--model', type=str, default='vgg11',
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
                       default='/opt/03-ai/01-proj/Neurosurgeon/results/multi_scenario',
                       help='结果保存目录')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='每个场景的测试样本数')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"多场景对比实验: {args.model}")
    print(f"{'='*80}\n")
    
    # 加载数据
    print("加载数据...")
    _, test_loader, num_classes = get_caltech101_dataloaders(
        args.data_dir, batch_size=1, num_workers=2
    )
    
    # 创建环境（使用默认网络参数，后续会动态修改）
    print("创建环境...")
    env = CollaborativeInferenceEnv(
        model_name=args.model,
        num_classes=num_classes,
        edge_device='cpu',
        cloud_device=args.device,
        bandwidth=100.0,
        network_latency=50.0,
        target_accuracy=0.85,
        max_latency=1000.0
    )
    
    # 加载模型权重
    model_checkpoint = os.path.join(args.checkpoint_dir, args.model, 'best_model.pth')
    if os.path.exists(model_checkpoint):
        print(f"加载模型权重: {model_checkpoint}")
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
        
        env.model = env.model.to('cpu')
        env.edge.model = env.edge.model.to('cpu')
        env.cloud.model = env.cloud.model.to('cpu')
        
        env.model.load_state_dict(checkpoint['model_state_dict'])
        env.edge.model.load_state_dict(checkpoint['model_state_dict'])
        env.cloud.model.load_state_dict(checkpoint['model_state_dict'])
        
        env.cloud.model = env.cloud.model.to(args.device)
        env.cloud.device = args.device
        
        print(f"  ✅ 模型已加载")
    
    # 加载RL智能体
    rl_agent_path = os.path.join(args.rl_agent_dir, f'rl_agent_{args.model}', 'best_agent.pth')
    
    rl_agent = None
    if os.path.exists(rl_agent_path):
        print(f"加载RL智能体: {rl_agent_path}")
        
        state_dim = 29
        num_partition_points = len(env.model.get_split_points())
        
        rl_agent = HybridPPO(
            state_dim=state_dim,
            num_partition_points=num_partition_points,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            clip_epsilon=0.2,
            device=args.device
        )
        
        checkpoint = torch.load(rl_agent_path, map_location=args.device)
        rl_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        rl_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        print(f"  ✅ RL智能体已加载")
    else:
        print(f"  ⚠️  未找到RL智能体: {rl_agent_path}")
        print("  将只测试All Edge和All Cloud")
    
    # 测试所有场景
    all_results = {}
    
    for scenario_name, scenario_config in NETWORK_SCENARIOS.items():
        results = test_scenario(
            env, test_loader, rl_agent, scenario_name, scenario_config, args.num_samples
        )
        all_results[scenario_name] = {
            'config': scenario_config,
            'results': results
        }
        
        # 保存单个场景的结果
        scenario_file = os.path.join(args.save_dir, f'{args.model}_{scenario_name}.json')
        with open(scenario_file, 'w') as f:
            json.dump(all_results[scenario_name], f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        print(f"✅ 场景结果已保存: {scenario_file}")
    
    # 保存汇总结果
    summary_file = os.path.join(args.save_dir, f'{args.model}_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    print(f"\n{'='*80}")
    print(f"所有场景测试完成！")
    print(f"汇总结果已保存: {summary_file}")
    print(f"{'='*80}\n")
    
    return all_results


if __name__ == '__main__':
    main()


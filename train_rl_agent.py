"""
强化学习智能体训练脚本
训练混合动作空间PPO智能体用于协同推理优化
"""
import torch
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import time

from dataset.caltech101_loader import get_caltech101_dataloaders
from environment.collaborative_env import CollaborativeInferenceEnv
from rl_agent.hybrid_ppo import HybridPPO


def train_rl_agent(model_name, data_dir, checkpoint_dir, save_dir,
                   num_episodes=1000, max_steps_per_episode=100,
                   update_interval=10, device='cuda'):
    """
    训练RL智能体
    
    Args:
        model_name: 模型名称
        data_dir: 数据目录
        checkpoint_dir: 模型检查点目录
        save_dir: RL智能体保存目录
        num_episodes: 训练回合数
        max_steps_per_episode: 每回合最大步数
        update_interval: 更新间隔
        device: 设备
    """
    print(f"\n{'='*60}")
    print(f"训练RL智能体: {model_name}")
    print(f"{'='*60}\n")
    
    # 创建保存目录
    agent_save_dir = os.path.join(save_dir, f'rl_agent_{model_name}')
    os.makedirs(agent_save_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader, num_classes = get_caltech101_dataloaders(
        data_dir, batch_size=32, num_workers=4
    )
    
    # 创建环境
    print("创建协同推理环境...")
    env = CollaborativeInferenceEnv(
        model_name=model_name,
        num_classes=num_classes,
        edge_device='cpu',  # 边缘设备通常是CPU
        cloud_device=device,  # 云端可以使用GPU
        bandwidth=100.0,
        network_latency=50.0,
        target_accuracy=0.85,
        max_latency=1000.0
    )
    
    # 加载训练好的模型权重
    model_checkpoint = os.path.join(checkpoint_dir, model_name, 'best_model.pth')
    if os.path.exists(model_checkpoint):
        print(f"加载模型权重: {model_checkpoint}")
        
        # 加载权重到CPU
        checkpoint = torch.load(model_checkpoint, map_location='cpu')
        
        # 先将所有模型移到CPU，然后加载权重
        env.model = env.model.to('cpu')
        env.edge.model = env.edge.model.to('cpu')
        env.cloud.model = env.cloud.model.to('cpu')
        
        # 加载权重
        env.model.load_state_dict(checkpoint['model_state_dict'])
        env.edge.model.load_state_dict(checkpoint['model_state_dict'])
        env.cloud.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 将云端模型移动到GPU
        env.cloud.model = env.cloud.model.to(device)
        env.cloud.device = device
        
        print(f"  模型已加载并移动到正确设备")
        print(f"  边缘设备: {env.edge.device}")
        print(f"  云端设备: {env.cloud.device}")
        print(f"  边缘模型实际设备: {next(env.edge.model.parameters()).device}")
        print(f"  云端模型实际设备: {next(env.cloud.model.parameters()).device}")
    else:
        print(f"警告: 未找到模型权重 {model_checkpoint}，使用随机初始化")
        # 确保设备设置正确
        env.edge.model = env.edge.model.to('cpu')
        env.edge.device = 'cpu'
        env.cloud.model = env.cloud.model.to(device)
        env.cloud.device = device
    
    # 创建RL智能体
    state_dim = 29
    num_partition_points = len(env.model.get_split_points())
    
    print(f"创建PPO智能体...")
    print(f"  状态维度: {state_dim}")
    print(f"  分割点数量: {num_partition_points}")
    
    agent = HybridPPO(
        state_dim=state_dim,
        num_partition_points=num_partition_points,
        compression_min=0.1,
        compression_max=1.0,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        device=device
    )
    
    # 训练历史
    history = {
        'episode_rewards': [],
        'episode_latencies': [],
        'episode_accuracies': [],
        'actor_losses': [],
        'critic_losses': [],
        'entropies': [],
        'best_reward': -float('inf'),
        'best_episode': 0
    }
    
    best_reward = -float('inf')
    
    # 训练循环
    print(f"\n开始训练...")
    data_iter = iter(train_loader)
    
    for episode in range(1, num_episodes + 1):
        episode_reward = 0
        episode_latency = 0
        episode_accuracy = 0
        step_count = 0
        
        pbar = tqdm(range(max_steps_per_episode), 
                   desc=f'Episode {episode}/{num_episodes}',
                   leave=False)
        
        for step in pbar:
            # 获取数据
            try:
                inputs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, labels = next(data_iter)
            
            # 重置环境
            state = env.reset(inputs, labels)
            
            # 选择动作
            partition_point, compression_rate = agent.select_action(state)
            action = (partition_point, compression_rate)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储转移
            agent.store_transition(reward, done)
            
            # 统计
            episode_reward += reward
            episode_latency += info['total_latency']
            episode_accuracy += info['accuracy']
            step_count += 1
            
            # 更新进度条
            pbar.set_postfix({
                'reward': episode_reward / step_count,
                'latency': episode_latency / step_count,
                'acc': episode_accuracy / step_count
            })
        
        # 计算平均值
        avg_reward = episode_reward / step_count
        avg_latency = episode_latency / step_count
        avg_accuracy = episode_accuracy / step_count
        
        # 更新策略
        if episode % update_interval == 0:
            metrics = agent.update(next_state, num_epochs=10, batch_size=64)
            
            if metrics:
                history['actor_losses'].append(metrics['actor_loss'])
                history['critic_losses'].append(metrics['critic_loss'])
                history['entropies'].append(metrics['entropy'])
        
        # 记录历史
        history['episode_rewards'].append(avg_reward)
        history['episode_latencies'].append(avg_latency)
        history['episode_accuracies'].append(avg_accuracy)
        
        # 打印结果
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  平均奖励: {avg_reward:.4f}")
            print(f"  平均时延: {avg_latency:.2f}ms")
            print(f"  平均准确率: {avg_accuracy:.4f}")
            if history['actor_losses']:
                print(f"  Actor损失: {history['actor_losses'][-1]:.4f}")
                print(f"  Critic损失: {history['critic_losses'][-1]:.4f}")
        
        # 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            history['best_reward'] = best_reward
            history['best_episode'] = episode
            
            best_agent_path = os.path.join(agent_save_dir, 'best_agent.pth')
            agent.save(best_agent_path)
            print(f"  保存最佳智能体 (奖励: {best_reward:.4f})")
        
        # 定期保存检查点
        if episode % 100 == 0:
            checkpoint_path = os.path.join(agent_save_dir, f'agent_episode_{episode}.pth')
            agent.save(checkpoint_path)
    
    # 保存训练历史
    history_path = os.path.join(agent_save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"最佳奖励: {best_reward:.4f} (Episode {history['best_episode']})")
    print(f"智能体保存在: {agent_save_dir}")
    print(f"{'='*60}\n")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='训练RL智能体')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'vgg11', 'mobilenetv2', 'alexnet', 'all'],
                       help='模型名称')
    parser.add_argument('--data_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/data/caltech-101',
                       help='数据目录')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/checkpoints',
                       help='模型检查点目录')
    parser.add_argument('--save_dir', type=str,
                       default='/opt/03-ai/01-proj/Neurosurgeon/rl_agents',
                       help='RL智能体保存目录')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='训练回合数')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='每回合最大步数')
    parser.add_argument('--update_interval', type=int, default=10,
                       help='策略更新间隔')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    print(f"\nRL训练配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  检查点目录: {args.checkpoint_dir}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  训练回合数: {args.episodes}")
    print(f"  每回合步数: {args.max_steps}")
    print(f"  更新间隔: {args.update_interval}")
    print(f"  设备: {args.device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练智能体
    if args.model == 'all':
        models = ['resnet18', 'vgg11', 'mobilenetv2', 'alexnet']
    else:
        models = [args.model]
    
    all_histories = {}
    
    for model_name in models:
        start_time = time.time()
        
        history = train_rl_agent(
            model_name=model_name,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            save_dir=args.save_dir,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            update_interval=args.update_interval,
            device=args.device
        )
        
        elapsed_time = time.time() - start_time
        print(f"{model_name} RL训练耗时: {elapsed_time/60:.2f} 分钟\n")
        
        all_histories[model_name] = history
    
    # 保存所有智能体的训练历史
    summary_path = os.path.join(args.save_dir, 'rl_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_histories, f, indent=4)
    
    # 打印总结
    print("\n" + "="*60)
    print("RL训练总结")
    print("="*60)
    for model_name, history in all_histories.items():
        print(f"{model_name:15s}: 最佳奖励 {history['best_reward']:.4f} (Episode {history['best_episode']})")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()


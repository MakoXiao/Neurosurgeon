"""
详细诊断准确率问题
"""
import os
import sys
import torch
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models.AlexNet import AlexNet
from src.env import CollaborativeInferenceEnv
from src.actor_critic import Actor, Critic
from src.ppo import PPO
from src.dataset_loader import get_caltech101_dataloader

def detailed_diagnosis():
    """详细诊断"""
    print("="*80)
    print("详细诊断准确率问题")
    print("="*80)
    
    # 加载模型
    model = AlexNet(3, 101)
    model.load_state_dict(torch.load('./alexnet_caltech101.pth', map_location='cpu'))
    model.eval()
    
    # 加载数据集
    _, test_dataset = get_caltech101_dataloader('../data/caltech-101', batch_size=1, split='test')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建环境
    env = CollaborativeInferenceEnv(
        model=model,
        dataset=test_dataset,
        edge_device=device,
        cloud_device=device,
        network_bandwidth=10.0,
        pruning_type='structured'
    )
    
    # 加载RL模型
    rl_model_path = './experiments/comparison/train_20251203_090732/final_model.pt'
    state_dim = 29
    num_partition_points = env.num_partition_points
    actor = Actor(state_dim, num_partition_points).to(device)
    critic = Critic(state_dim).to(device)
    agent = PPO(actor, critic)
    agent.load(rl_model_path)
    
    print("\n测试前10个样本的详细预测:")
    print("-"*80)
    
    for i in range(10):
        state = env.reset()
        action, _, _, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        
        # 手动检查预测
        with torch.no_grad():
            # 获取edge和cloud模型
            from src.model_partitioner import ModelPartitioner
            partitioner = ModelPartitioner(model)
            edge_model, cloud_model = partitioner.partition(action['partition_point'])
            edge_model = edge_model.to(device)
            cloud_model = cloud_model.to(device)
            
            # Edge inference
            edge_output = edge_model(env.current_sample.to(device))
            
            # Prune
            from src.pruning import PruningManager
            pruning_manager = PruningManager(pruning_type='structured')
            pruned_feature, pruning_info = pruning_manager.compress(edge_output, action['compression_rate'])
            
            # Decompress
            recovered_feature = pruning_manager.decompress(pruned_feature, pruning_info, device)
            
            # Cloud inference
            cloud_output = cloud_model(recovered_feature)
            pred = torch.argmax(cloud_output, dim=1).item()
        
        label = env.current_label
        acc = info['accuracy']
        
        print(f"Sample {i}: label={label}, pred={pred}, accuracy={acc}, match={pred==label}")
        if i >= 4:
            break
    
    print("\n" + "="*80)

if __name__ == "__main__":
    detailed_diagnosis()



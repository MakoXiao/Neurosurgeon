"""
诊断JALAD和Proposed RL的准确率问题
"""
import os
import sys
import torch
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models.AlexNet import AlexNet
from baselines.jalad_baseline import JALADBaseline
from baselines.local_baseline import LocalBaseline
from src.dataset_loader import get_caltech101_dataloader
from src.env import CollaborativeInferenceEnv
from src.actor_critic import Actor, Critic
from src.ppo import PPO

def test_accuracy_calculation():
    """测试准确率计算"""
    print("="*80)
    print("诊断准确率计算问题")
    print("="*80)
    
    # 加载模型
    model = AlexNet(3, 101)
    model_path = './alexnet_caltech101.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✓ 模型已加载: {model_path}")
    else:
        print(f"✗ 模型文件不存在: {model_path}")
        return
    
    model.eval()
    
    # 加载数据集
    _, test_dataset = get_caltech101_dataloader('../data/caltech-101', batch_size=1, split='test')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ 使用设备: {device}")
    print(f"✓ 测试集大小: {len(test_dataset)}")
    
    # 测试前10个样本
    print("\n" + "="*80)
    print("测试前10个样本的准确率计算")
    print("="*80)
    
    # 1. Local Baseline
    print("\n--- Local Baseline ---")
    local = LocalBaseline(model, device=device)
    local_correct = 0
    for i in range(10):
        data, label = test_dataset[i]
        data = data.unsqueeze(0) if data.dim() == 3 else data
        acc, lat, info = local.inference(data, label)
        if acc > 0:
            local_correct += 1
        if i < 3:
            print(f"  Sample {i}: label={label}, accuracy={acc}, pred={torch.argmax(model(data.to(device)), dim=1).item()}")
    print(f"  Local准确率: {local_correct}/10 = {local_correct/10:.2%}")
    
    # 2. JALAD Baseline
    print("\n--- JALAD Baseline ---")
    jalad = JALADBaseline(
        model=model,
        dataset=test_dataset,
        edge_device=device,
        cloud_device=device,
        network_bandwidth=10.0,
        compression_ratio=0.5,
        partition_point=4
    )
    jalad_correct = 0
    for i in range(10):
        data, label = test_dataset[i]
        data = data.unsqueeze(0) if data.dim() == 3 else data
        try:
            acc, lat, info = jalad.inference(data, label)
            if acc > 0:
                jalad_correct += 1
            if i < 3:
                # 手动检查预测
                with torch.no_grad():
                    edge_output = jalad.edge_model(data.to(device))
                    # 压缩和解压缩
                    feature_flat = edge_output.view(1, -1)
                    if feature_flat.shape[1] > 100000:
                        indices = torch.randperm(feature_flat.shape[1])[:100000]
                        feature_flat = feature_flat[:, indices]
                    compressed = jalad.autoencoder.encode(feature_flat.cpu())
                    decompressed = jalad.autoencoder.decode(compressed)
                    # 恢复形状
                    decompressed = decompressed.to(device)
                    if decompressed.shape[1] < edge_output.view(1, -1).shape[1]:
                        padding = torch.zeros(1, edge_output.view(1, -1).shape[1] - decompressed.shape[1], device=device)
                        decompressed = torch.cat([decompressed, padding], dim=1)
                    decompressed = decompressed.view(edge_output.shape)
                    cloud_output = jalad.cloud_model(decompressed)
                    pred = torch.argmax(cloud_output, dim=1).item()
                print(f"  Sample {i}: label={label}, accuracy={acc}, pred={pred}, match={pred==label}")
        except Exception as e:
            print(f"  Sample {i}: Error - {e}")
            import traceback
            traceback.print_exc()
    print(f"  JALAD准确率: {jalad_correct}/10 = {jalad_correct/10:.2%}")
    
    # 3. Proposed RL (使用环境)
    print("\n--- Proposed RL (Environment) ---")
    env = CollaborativeInferenceEnv(
        model=model,
        dataset=test_dataset,
        edge_device=device,
        cloud_device=device,
        network_bandwidth=10.0,
        pruning_type='structured'
    )
    
    # 加载训练好的RL模型
    rl_model_path = './experiments/comparison/train_20251203_090732/final_model.pt'
    if os.path.exists(rl_model_path):
        state_dim = 29
        num_partition_points = env.num_partition_points
        actor = Actor(state_dim, num_partition_points).to(device)
        critic = Critic(state_dim).to(device)
        agent = PPO(actor, critic)
        try:
            agent.load(rl_model_path)
            print(f"✓ RL模型已加载: {rl_model_path}")
        except Exception as e:
            print(f"✗ 无法加载RL模型: {e}")
            return
    else:
        print(f"✗ RL模型文件不存在: {rl_model_path}")
        return
    
    rl_correct = 0
    state = env.reset()
    for i in range(10):
        action, _, _, _ = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        
        acc = info['accuracy']
        if acc > 0:
            rl_correct += 1
        
        if i < 3:
            print(f"  Sample {i}: accuracy={acc}, reward={reward:.4f}, latency={info['latency']*1000:.2f}ms")
            # 检查环境中的标签和预测
            print(f"    current_label={env.current_label}, partition_point={action['partition_point']}")
        
        state = next_state
        if done:
            state = env.reset()
    
    print(f"  Proposed RL准确率: {rl_correct}/10 = {rl_correct/10:.2%}")
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)

if __name__ == "__main__":
    test_accuracy_calculation()



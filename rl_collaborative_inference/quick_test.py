"""
Quick test script to verify the implementation
"""
import os
import sys
import torch

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.actor_critic import Actor, Critic
from src.pruning import PruningManager
from src.model_partition import ModelPartitioner
from src.state_space import StateSpace
from models.AlexNet import AlexNet

print("Testing modules...")

# Test 1: Model creation
print("\n1. Testing model creation...")
model = AlexNet(input_channels=3, num_classes=101)
print(f"   ✓ Model created: {len(model)} layers")

# Test 2: Model partitioner
print("\n2. Testing model partitioner...")
partitioner = ModelPartitioner(model)
print(f"   ✓ Valid partition points: {len(partitioner.valid_partition_points)}")
edge_model, cloud_model = partitioner.partition(4)
print(f"   ✓ Partition successful: edge={len(edge_model)} layers, cloud={len(cloud_model)} layers")

# Test 3: Pruning manager
print("\n3. Testing pruning manager...")
pruning_manager = PruningManager(pruning_type='structured')
test_tensor = torch.randn(1, 256, 14, 14)
pruned, info = pruning_manager.compress(test_tensor, 0.5)
recovered = pruning_manager.decompress(pruned, info, 'cpu')
print(f"   ✓ Pruning successful: {test_tensor.shape} -> {pruned.shape} -> {recovered.shape}")

# Test 4: State space
print("\n4. Testing state space...")
state_space = StateSpace()
state = state_space.build_state()
print(f"   ✓ State created: shape={state.shape}, dim={len(state)}")

# Test 5: Actor-Critic
print("\n5. Testing Actor-Critic networks...")
state_dim = 29
num_partition_points = len(partitioner.valid_partition_points)
actor = Actor(state_dim, num_partition_points)
critic = Critic(state_dim)
test_state = torch.randn(1, state_dim)
action = actor.select_action(test_state)
value = critic(test_state)
print(f"   ✓ Actor action: partition={action['partition_point']}, compression={action['compression_rate']:.3f}")
print(f"   ✓ Critic value: {value.item():.3f}")

print("\n✓ All tests passed!")


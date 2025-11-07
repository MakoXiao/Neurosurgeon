"""
Training script for RL-based collaborative inference
"""
import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.actor_critic import Actor, Critic
from src.env import CollaborativeInferenceEnv
from src.ppo import PPO
from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet


def train(args):
    """Training function"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"train_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    print("Loading Caltech-101 dataset...")
    dataloader, dataset = get_caltech101_dataloader(
        args.data_dir,
        batch_size=1,
        split='train',
        num_workers=0
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    print("Creating model...")
    model = AlexNet(input_channels=3, num_classes=101)
    model.eval()
    
    # Create environment
    print("Creating environment...")
    env = CollaborativeInferenceEnv(
        model=model,
        dataset=dataset,
        edge_device=device,
        cloud_device=device,
        network_bandwidth=args.network_bandwidth,
        pruning_type=args.pruning_type,
        target_accuracy=args.target_accuracy,
        max_latency=args.max_latency,
        alpha=args.alpha,
        beta=args.beta
    )
    
    # Get state and action dimensions
    state_dim = 29
    num_partition_points = env.num_partition_points
    
    # Create networks
    print("Creating networks...")
    actor = Actor(state_dim, num_partition_points).to(device)
    critic = Critic(state_dim).to(device)
    
    # Create PPO agent
    agent = PPO(
        actor=actor,
        critic=critic,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef
    )
    
    # Training loop
    print("Starting training...")
    episode_rewards = []
    episode_accuracies = []
    episode_latencies = []
    
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    for step in tqdm(range(args.max_steps)):
        # Select action
        action, log_prob, value, entropy = agent.select_action(state)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Store experience
        agent.buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        # Update policy
        if len(agent.buffer) >= args.batch_size and step % args.update_freq == 0:
            agent.update(batch_size=args.batch_size)
        
        # Reset environment periodically
        if done or episode_steps >= args.max_episode_steps:
            episode_rewards.append(episode_reward)
            episode_accuracies.append(info.get('accuracy', 0.0))
            episode_latencies.append(info.get('latency', 0.0))
            
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
        
        # Logging
        if step % args.log_freq == 0 and episode_rewards:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_accuracy = np.mean(episode_accuracies[-100:])
            avg_latency = np.mean(episode_latencies[-100:])
            
            print(f"\nStep {step}:")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Accuracy: {avg_accuracy:.4f}")
            print(f"  Avg Latency: {avg_latency:.4f}s")
            
            # Save checkpoint
            if step % args.save_freq == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pt")
                agent.save(checkpoint_path)
                print(f"  Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    agent.save(final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")
    
    # Save training history
    history = {
        'rewards': episode_rewards,
        'accuracies': episode_accuracies,
        'latencies': episode_latencies
    }
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--data_dir', type=str, default='../data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    
    # Training
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum training steps')
    parser.add_argument('--max_episode_steps', type=int, default=100,
                       help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--update_freq', type=int, default=10,
                       help='Update frequency')
    
    # PPO
    parser.add_argument('--lr_actor', type=float, default=3e-4,
                       help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4,
                       help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                       help='PPO clip parameter')
    parser.add_argument('--k_epochs', type=int, default=10,
                       help='Number of update epochs')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                       help='Entropy coefficient')
    
    # Environment
    parser.add_argument('--network_bandwidth', type=float, default=10.0,
                       help='Network bandwidth (MB/s)')
    parser.add_argument('--pruning_type', type=str, default='structured',
                       choices=['structured', 'unstructured'],
                       help='Pruning type')
    parser.add_argument('--target_accuracy', type=float, default=0.95,
                       help='Target accuracy')
    parser.add_argument('--max_latency', type=float, default=1.0,
                       help='Maximum latency (seconds)')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Accuracy weight in reward')
    parser.add_argument('--beta', type=float, default=0.4,
                       help='Latency weight in reward')
    
    # Logging
    parser.add_argument('--log_freq', type=int, default=100,
                       help='Logging frequency')
    parser.add_argument('--save_freq', type=int, default=1000,
                       help='Save frequency')
    
    # Device
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    train(args)


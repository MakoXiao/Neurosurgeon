"""
Enhanced training script with comprehensive tracking for paper experiments
Records cumulative rewards, value loss, and other metrics for generating paper figures
"""
import os
import sys
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from collections import deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.actor_critic import Actor, Critic
from src.env import CollaborativeInferenceEnv
from src.ppo import PPO
from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet


class TrainingTracker:
    """Track training metrics for paper figures"""
    
    def __init__(self, log_freq=100, window_size=100):
        self.log_freq = log_freq
        self.window_size = window_size
        
        # Cumulative metrics
        self.cumulative_reward = 0.0
        self.cumulative_rewards_history = []
        
        # Episode metrics
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_accuracies = deque(maxlen=window_size)
        self.episode_latencies = deque(maxlen=window_size)
        
        # Step-level metrics
        self.step_rewards = []
        self.step_values = []
        self.value_losses = []
        
        # Time frame tracking
        self.time_frames = []
        
    def update(self, step, reward, value, value_loss=None, info=None):
        """Update tracking metrics"""
        self.cumulative_reward += reward
        self.step_rewards.append(reward)
        self.step_values.append(value)
        
        if value_loss is not None:
            self.value_losses.append(value_loss)
        
        # Log at specified frequency
        if step % self.log_freq == 0:
            self.cumulative_rewards_history.append({
                'time_frame': step,
                'cumulative_reward': self.cumulative_reward,
                'avg_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0.0,
                'avg_accuracy': np.mean(list(self.episode_accuracies)) if self.episode_accuracies else 0.0,
                'avg_latency': np.mean(list(self.episode_latencies)) if self.episode_latencies else 0.0,
                'value_loss': self.value_losses[-1] if self.value_losses else None
            })
            self.time_frames.append(step)
    
    def record_episode(self, reward, accuracy, latency):
        """Record episode-level metrics"""
        self.episode_rewards.append(reward)
        self.episode_accuracies.append(accuracy)
        self.episode_latencies.append(latency)
    
    def get_history(self):
        """Get training history"""
        return {
            'cumulative_rewards': self.cumulative_rewards_history,
            'episode_rewards': list(self.episode_rewards),
            'episode_accuracies': list(self.episode_accuracies),
            'episode_latencies': list(self.episode_latencies),
            'value_losses': self.value_losses,
            'time_frames': self.time_frames
        }


def train(args):
    """Training function with comprehensive tracking"""
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
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
    
    # Create tracker
    tracker = TrainingTracker(log_freq=args.log_freq, window_size=100)
    
    # Training loop
    print("Starting training...")
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    last_value_loss = None
    
    for step in tqdm(range(args.max_steps), desc="Training"):
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
            # Modified PPO update to return value loss
            value_loss = agent.update(batch_size=args.batch_size)
            if value_loss is not None:
                last_value_loss = value_loss
        
        # Reset environment periodically
        if done or episode_steps >= args.max_episode_steps:
            tracker.record_episode(
                episode_reward,
                info.get('accuracy', 0.0),
                info.get('latency', 0.0)
            )
            
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
        
        # Update tracker
        tracker.update(step, reward, value, last_value_loss, info)
        
        # Logging
        if step % args.log_freq == 0 and tracker.episode_rewards:
            avg_reward = np.mean(list(tracker.episode_rewards))
            avg_accuracy = np.mean(list(tracker.episode_accuracies))
            avg_latency = np.mean(list(tracker.episode_latencies))
            
            print(f"\nStep {step}:")
            print(f"  Cumulative Reward: {tracker.cumulative_reward:.4f}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Accuracy: {avg_accuracy:.4f}")
            print(f"  Avg Latency: {avg_latency:.4f}s")
            if last_value_loss is not None:
                print(f"  Value Loss: {last_value_loss:.6f}")
            
            # Save checkpoint
            if step % args.save_freq == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pt")
                agent.save(checkpoint_path)
                
                # Save training history
                history = tracker.get_history()
                with open(os.path.join(output_dir, f'training_history_{step}.json'), 'w') as f:
                    json.dump(history, f, indent=2)
    
    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    agent.save(final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")
    
    # Save final training history
    history = tracker.get_history()
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to {os.path.join(output_dir, 'training_history.json')}")
    
    return output_dir, tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced training with tracking')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='../data/caltech-101',
                       help='Path to Caltech-101 dataset')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory')
    
    # Training
    parser.add_argument('--max_steps', type=int, default=500000,
                       help='Maximum training steps (500K for paper experiments)')
    parser.add_argument('--max_episode_steps', type=int, default=100,
                       help='Maximum steps per episode')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--update_freq', type=int, default=10,
                       help='Update frequency')
    
    # PPO
    parser.add_argument('--lr_actor', type=float, default=0.0001,
                       help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=0.0001,
                       help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--eps_clip', type=float, default=0.2,
                       help='PPO clip parameter')
    parser.add_argument('--k_epochs', type=int, default=10,
                       help='Number of update epochs (reuse time)')
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
    parser.add_argument('--log_freq', type=int, default=1000,
                       help='Logging frequency')
    parser.add_argument('--save_freq', type=int, default=50000,
                       help='Save frequency')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    # Device
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA if available')
    
    args = parser.parse_args()
    
    train(args)


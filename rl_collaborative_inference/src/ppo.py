"""
PPO (Proximal Policy Optimization) algorithm
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque


class PPOBuffer:
    """Buffer for PPO algorithm"""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, log_prob, value):
        """Push experience to buffer"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def sample(self, batch_size):
        """Sample batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class PPO:
    """PPO algorithm"""
    
    def __init__(self, actor, critic, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, eps_clip=0.2, k_epochs=10, entropy_coef=0.01):
        """
        :param actor: Actor network
        :param critic: Critic network
        :param lr_actor: Actor learning rate
        :param lr_critic: Critic learning rate
        :param gamma: Discount factor
        :param eps_clip: PPO clip parameter
        :param k_epochs: Number of update epochs
        :param entropy_coef: Entropy coefficient
        """
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)
        
        self.buffer = PPOBuffer()
    
    def select_action(self, state, deterministic=False):
        """
        Select action
        :param state: state
        :param deterministic: whether to use deterministic action
        :return: action, log_prob, value
        """
        if isinstance(state, torch.Tensor):
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 确保state tensor在actor所在的设备上
        device = next(self.actor.parameters()).device
        state_tensor = state_tensor.to(device)
        
        # Get action from actor
        action = self.actor.select_action(state_tensor, deterministic=deterministic)
        
        # Get log probability
        log_prob, entropy = self.actor.evaluate_action(state_tensor, action)
        
        # Get value from critic
        value = self.critic(state_tensor)
        
        return action, log_prob.item(), value.item(), entropy.item()
    
    def update(self, batch_size=64):
        """
        Update policy
        :param batch_size: batch size
        """
        if len(self.buffer) < batch_size:
            return
        
        # Sample batch
        batch = self.buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([b['state'] for b in batch])
        actions = [b['action'] for b in batch]
        rewards = torch.FloatTensor([b['reward'] for b in batch])
        next_states = torch.FloatTensor([b['next_state'] for b in batch])
        dones = torch.FloatTensor([b['done'] for b in batch])
        old_log_probs = torch.FloatTensor([b['log_prob'] for b in batch])
        old_values = torch.FloatTensor([b['value'] for b in batch])
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, dones, old_values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update for k epochs
        for _ in range(self.k_epochs):
            # Evaluate actions
            new_log_probs = []
            entropies = []
            for i, state in enumerate(states):
                log_prob, entropy = self.actor.evaluate_action(state.unsqueeze(0), actions[i])
                new_log_probs.append(log_prob)
                entropies.append(entropy)
            
            new_log_probs = torch.stack(new_log_probs).squeeze()
            entropies = torch.stack(entropies).squeeze()
            
            # Compute new values
            new_values = self.critic(states).squeeze()
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropies.mean()
            
            # Critic loss
            critic_loss = F.mse_loss(new_values, returns)
            
            # Update networks
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()
            
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()
        
        # Clear buffer
        self.buffer.clear()
    
    def _compute_gae(self, rewards, dones, values, next_value=0):
        """
        Compute Generalized Advantage Estimation (GAE)
        :param rewards: rewards
        :param dones: done flags
        :param values: values
        :param next_value: next value
        :return: returns, advantages
        """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * 0.95 * gae  # lambda = 0.95
            
            returns[t] = gae + values[t]
            advantages[t] = gae
            next_value = values[t]
        
        return returns, advantages
    
    def save(self, filepath):
        """Save models"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load models"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])


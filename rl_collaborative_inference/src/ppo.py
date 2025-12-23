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
        # Get device from actor model
        device = next(self.actor.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
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
            return None
        
        # Sample batch
        batch = self.buffer.sample(batch_size)
        
        # Get device from actor model
        device = next(self.actor.parameters()).device
        
        # Convert to tensors and move to correct device
        # Ensure states are properly shaped as [batch_size, state_dim]
        state_list = []
        for b in batch:
            s = b['state']
            # Convert to numpy array first if needed
            if isinstance(s, torch.Tensor):
                s = s.cpu().numpy()
            elif not isinstance(s, np.ndarray):
                s = np.array(s)
            # Flatten to 1D if needed
            s = s.flatten()
            state_list.append(s)
        
        # Convert to tensor and ensure 2D shape [batch_size, state_dim]
        states = torch.FloatTensor(np.array(state_list)).to(device)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        
        actions = [b['action'] for b in batch]
        rewards = torch.FloatTensor([b['reward'] for b in batch]).to(device)
        # Handle next_states similarly
        next_states_list = []
        for b in batch:
            s = b['next_state']
            if isinstance(s, torch.Tensor):
                s = s.cpu().numpy()
            elif not isinstance(s, np.ndarray):
                s = np.array(s)
            s = s.flatten()
            next_states_list.append(s)
        next_states = torch.FloatTensor(np.array(next_states_list)).to(device)
        if next_states.dim() == 1:
            next_states = next_states.unsqueeze(0)
        
        dones = torch.FloatTensor([b['done'] for b in batch]).to(device)
        old_log_probs = torch.FloatTensor([b['log_prob'] for b in batch]).to(device)
        old_values = torch.FloatTensor([b['value'] for b in batch]).to(device)
        
        # Check for NaN or Inf values
        if torch.isnan(states).any() or torch.isinf(states).any():
            print(f"Warning: NaN or Inf values in states, skipping update")
            self.buffer.clear()
            return None
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, dones, old_values)
        
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update for k epochs
        avg_critic_loss = 0.0
        for _ in range(self.k_epochs):
            # Evaluate actions
            new_log_probs = []
            entropies = []
            for i in range(len(states)):
                # Get single state and ensure it's 2D [1, state_dim]
                state = states[i:i+1]  # Keep batch dimension
                log_prob, entropy = self.actor.evaluate_action(state, actions[i])
                new_log_probs.append(log_prob)
                entropies.append(entropy)
            
            new_log_probs = torch.stack(new_log_probs).squeeze()
            entropies = torch.stack(entropies).squeeze()
            
            # Compute new values - states should already be 2D [batch_size, state_dim]
            # Verify states shape before passing to critic
            if states.dim() != 2:
                print(f"Error: states has wrong dimension: {states.shape}, expected 2D")
                self.buffer.clear()
                return None
            
            # Verify state_dim matches critic network input size
            expected_state_dim = self.critic.net[0].in_features
            if states.shape[1] != expected_state_dim:
                print(f"Error: states shape mismatch: got {states.shape[1]}, expected {expected_state_dim}")
                self.buffer.clear()
                return None
            
            # Check for invalid values before CUDA operation
            if torch.isnan(states).any() or torch.isinf(states).any():
                print(f"Warning: NaN or Inf in states before critic, skipping")
                continue
            
            # Clamp states to reasonable range to avoid numerical issues
            states = torch.clamp(states, -10.0, 10.0)
            
            try:
                new_values = self.critic(states)
                if new_values.dim() > 1:
                    new_values = new_values.squeeze()
            except RuntimeError as e:
                print(f"CUDA error in critic forward: {e}")
                print(f"States shape: {states.shape}, dtype: {states.dtype}, device: {states.device}")
                if states.numel() > 0:
                    print(f"States stats: min={states.min().item():.4f}, max={states.max().item():.4f}, mean={states.mean().item():.4f}")
                # Clear buffer and return to avoid further errors
                self.buffer.clear()
                return None
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropies.mean()
            
            # Critic loss
            critic_loss = F.mse_loss(new_values, returns)
            avg_critic_loss += critic_loss.item()
            
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
        
        return avg_critic_loss / self.k_epochs
    
    def update_with_loss(self, batch_size=64):
        """
        Update policy and return value loss
        :param batch_size: batch size
        :return: average value loss
        """
        return self.update(batch_size)
    
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


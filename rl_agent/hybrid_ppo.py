"""
混合动作空间的PPO算法（创新点一）
同时优化分割点选择（离散）和压缩率选择（连续）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np


class ActorNetwork(nn.Module):
    """Actor网络：输出混合动作空间的策略"""
    
    def __init__(self, state_dim, num_partition_points, 
                 compression_min=0.1, compression_max=1.0):
        """
        Args:
            state_dim: 状态空间维度
            num_partition_points: 分割点数量
            compression_min: 最小压缩率
            compression_max: 最大压缩率
        """
        super(ActorNetwork, self).__init__()
        
        self.compression_min = compression_min
        self.compression_max = compression_max
        
        # 共享基础网络
        self.base = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 分割点头部（离散动作）
        self.partition_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_partition_points)
        )
        
        # 压缩率头部（连续动作）- 均值
        self.compression_mu_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出[0, 1]
        )
        
        # 压缩率头部（连续动作）- 标准差
        self.compression_sigma_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 保证正值
        )
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
        
        Returns:
            partition_logits: 分割点logits [batch_size, num_partition_points]
            compression_mu: 压缩率均值 [batch_size, 1]
            compression_sigma: 压缩率标准差 [batch_size, 1]
        """
        code = self.base(state)
        
        # 分割点logits
        partition_logits = self.partition_head(code)
        
        # 压缩率分布参数
        compression_mu_normalized = self.compression_mu_head(code)
        compression_mu = (compression_mu_normalized * 
                         (self.compression_max - self.compression_min) + 
                         self.compression_min)
        compression_sigma = self.compression_sigma_head(code) + 1e-5
        
        return partition_logits, compression_mu, compression_sigma
    
    def sample_action(self, state):
        """
        采样动作
        
        Args:
            state: 状态张量
        
        Returns:
            partition_point: 分割点
            compression_rate: 压缩率
            log_prob: 对数概率
        """
        partition_logits, compression_mu, compression_sigma = self.forward(state)
        
        # 采样分割点
        partition_dist = Categorical(logits=partition_logits)
        partition_point = partition_dist.sample()
        partition_log_prob = partition_dist.log_prob(partition_point)
        
        # 采样压缩率
        compression_dist = Normal(compression_mu, compression_sigma)
        compression_rate = compression_dist.sample()
        compression_rate = torch.clamp(compression_rate, 
                                      self.compression_min, 
                                      self.compression_max)
        compression_log_prob = compression_dist.log_prob(compression_rate)
        
        # 总对数概率
        log_prob = partition_log_prob + compression_log_prob.squeeze(-1)
        
        return partition_point, compression_rate.squeeze(-1), log_prob
    
    def evaluate_action(self, state, partition_point, compression_rate):
        """
        评估动作的对数概率和熵
        
        Args:
            state: 状态张量
            partition_point: 分割点
            compression_rate: 压缩率
        
        Returns:
            log_prob: 对数概率
            entropy: 熵
        """
        partition_logits, compression_mu, compression_sigma = self.forward(state)
        
        # 评估分割点
        partition_dist = Categorical(logits=partition_logits)
        partition_log_prob = partition_dist.log_prob(partition_point)
        partition_entropy = partition_dist.entropy()
        
        # 评估压缩率
        compression_dist = Normal(compression_mu, compression_sigma)
        compression_log_prob = compression_dist.log_prob(compression_rate.unsqueeze(-1))
        compression_entropy = compression_dist.entropy()
        
        # 总对数概率和熵
        log_prob = partition_log_prob + compression_log_prob.squeeze(-1)
        entropy = partition_entropy + compression_entropy.squeeze(-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Critic网络：评估状态价值"""
    
    def __init__(self, state_dim):
        """
        Args:
            state_dim: 状态空间维度
        """
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
        
        Returns:
            value: 状态价值 [batch_size, 1]
        """
        return self.network(state)


class HybridPPO:
    """混合动作空间的PPO算法"""
    
    def __init__(self, state_dim, num_partition_points, 
                 compression_min=0.1, compression_max=1.0,
                 lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, entropy_coef=0.01,
                 value_loss_coef=0.5, max_grad_norm=0.5,
                 device='cpu'):
        """
        Args:
            state_dim: 状态空间维度
            num_partition_points: 分割点数量
            compression_min: 最小压缩率
            compression_max: 最大压缩率
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_epsilon: PPO裁剪参数
            entropy_coef: 熵系数
            value_loss_coef: 价值损失系数
            max_grad_norm: 梯度裁剪阈值
            device: 设备
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # 创建网络
        self.actor = ActorNetwork(
            state_dim, num_partition_points,
            compression_min, compression_max
        ).to(device)
        
        self.critic = CriticNetwork(state_dim).to(device)
        
        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验缓冲区
        self.buffer = {
            'states': [],
            'partition_points': [],
            'compression_rates': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }
    
    def select_action(self, state, deterministic=False):
        """
        选择动作
        
        Args:
            state: 状态 (numpy array)
            deterministic: 是否确定性选择
        
        Returns:
            partition_point: 分割点
            compression_rate: 压缩率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                partition_logits, compression_mu, _ = self.actor(state_tensor)
                partition_point = torch.argmax(partition_logits, dim=-1)
                compression_rate = compression_mu.squeeze(-1)
            else:
                partition_point, compression_rate, log_prob = self.actor.sample_action(state_tensor)
                value = self.critic(state_tensor)
                
                # 存储到缓冲区
                self.buffer['states'].append(state)
                self.buffer['partition_points'].append(partition_point.item())
                self.buffer['compression_rates'].append(compression_rate.item())
                self.buffer['log_probs'].append(log_prob.item())
                self.buffer['values'].append(value.item())
        
        return partition_point.item(), compression_rate.item()
    
    def store_transition(self, reward, done):
        """
        存储转移
        
        Args:
            reward: 奖励
            done: 是否结束
        """
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
    
    def compute_gae(self, next_value):
        """
        计算GAE（Generalized Advantage Estimation）
        
        Args:
            next_value: 下一个状态的价值
        
        Returns:
            advantages: 优势函数
            returns: 回报
        """
        rewards = self.buffer['rewards']
        values = self.buffer['values']
        dones = self.buffer['dones']
        
        advantages = []
        gae = 0
        
        # 从后向前计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns
    
    def update(self, next_state, num_epochs=10, batch_size=64):
        """
        更新策略
        
        Args:
            next_state: 下一个状态
            num_epochs: 更新轮数
            batch_size: 批次大小
        
        Returns:
            metrics: 训练指标字典
        """
        if len(self.buffer['states']) == 0:
            return {}
        
        # 计算下一个状态的价值
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.critic(next_state_tensor).item()
        
        # 计算GAE
        advantages, returns = self.compute_gae(next_value)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        partition_points = torch.LongTensor(self.buffer['partition_points']).to(self.device)
        compression_rates = torch.FloatTensor(self.buffer['compression_rates']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        
        # 训练指标
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # 多轮更新
        for epoch in range(num_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_partition_points = partition_points[batch_indices]
                batch_compression_rates = compression_rates[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 评估动作
                log_probs, entropy = self.actor.evaluate_action(
                    batch_states, batch_partition_points, batch_compression_rates
                )
                
                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # 计算价值损失
                values = self.critic(batch_states).squeeze(-1)
                critic_loss = self.value_loss_coef * nn.MSELoss()(values, batch_returns)
                
                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
        
        # 清空缓冲区
        for key in self.buffer:
            self.buffer[key] = []
        
        num_updates = num_epochs * (len(states) // batch_size + 1)
        metrics = {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        return metrics
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


if __name__ == '__main__':
    print("测试混合动作空间PPO算法...")
    
    # 创建PPO智能体
    state_dim = 29
    num_partition_points = 7
    
    agent = HybridPPO(
        state_dim=state_dim,
        num_partition_points=num_partition_points,
        compression_min=0.1,
        compression_max=1.0,
        device='cpu'
    )
    
    print(f"Actor网络: {sum(p.numel() for p in agent.actor.parameters())} 参数")
    print(f"Critic网络: {sum(p.numel() for p in agent.critic.parameters())} 参数")
    
    # 测试动作选择
    state = np.random.randn(state_dim)
    partition_point, compression_rate = agent.select_action(state)
    print(f"\n采样动作:")
    print(f"  分割点: {partition_point}")
    print(f"  压缩率: {compression_rate:.3f}")
    
    # 模拟一个episode
    for step in range(10):
        state = np.random.randn(state_dim)
        partition_point, compression_rate = agent.select_action(state)
        reward = np.random.randn()
        done = (step == 9)
        agent.store_transition(reward, done)
    
    # 更新策略
    next_state = np.random.randn(state_dim)
    metrics = agent.update(next_state, num_epochs=5, batch_size=32)
    
    print(f"\n训练指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


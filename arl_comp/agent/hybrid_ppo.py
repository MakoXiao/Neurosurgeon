"""
Hybrid PPO Agent: 处理混合动作空间 (离散 + 连续)
- 离散动作: 分区点选择 (Categorical分布)
- 连续动作: 压缩率 (Beta分布, 输出在[0,1]范围, 再缩放到[0.1,1.0])
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
import numpy as np


class HybridActor(nn.Module):
    """混合动作空间的Actor网络"""

    def __init__(self, state_dim, num_discrete_actions, hidden_dim=128):
        super(HybridActor, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 离散动作头: 分区点选择
        self.discrete_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_discrete_actions),
        )

        # 连续动作头: 压缩率 (Beta分布的alpha和beta参数)
        self.continuous_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(hidden_dim // 2, 1)
        self.beta_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state):
        shared_features = self.shared(state)

        # 离散分布
        discrete_logits = self.discrete_head(shared_features)
        discrete_probs = F.softmax(discrete_logits, dim=-1)

        # 连续分布 (Beta分布)
        cont_features = self.continuous_head(shared_features)
        alpha = F.softplus(self.alpha_head(cont_features)) + 1.0  # > 1 确保unimodal
        beta = F.softplus(self.beta_head(cont_features)) + 1.0

        return discrete_probs, alpha.squeeze(-1), beta.squeeze(-1)


class Critic(nn.Module):
    """价值网络"""

    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


class HybridPPOBuffer:
    """经验回放缓冲区"""

    def __init__(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.discrete_logprobs = []
        self.continuous_logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, discrete_action, continuous_action,
              discrete_logprob, continuous_logprob, reward, done, value):
        self.states.append(state)
        self.discrete_actions.append(discrete_action)
        self.continuous_actions.append(continuous_action)
        self.discrete_logprobs.append(discrete_logprob)
        self.continuous_logprobs.append(continuous_logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.discrete_actions.clear()
        self.continuous_actions.clear()
        self.discrete_logprobs.clear()
        self.continuous_logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class HybridPPO:
    """
    Hybrid PPO: 支持混合动作空间的PPO算法

    离散动作使用Categorical分布
    连续动作使用Beta分布 (自然范围[0,1], 缩放到[action_low, action_high])
    """

    def __init__(self, state_dim, num_discrete_actions,
                 continuous_action_low=0.1, continuous_action_high=1.0,
                 hidden_dim=128, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, gae_lambda=0.95, eps_clip=0.2,
                 entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, update_epochs=10,
                 batch_size=64, device="cpu"):

        self.device = torch.device(device)
        self.state_dim = state_dim
        self.num_discrete_actions = num_discrete_actions
        self.continuous_action_low = continuous_action_low
        self.continuous_action_high = continuous_action_high

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        self.actor = HybridActor(state_dim, num_discrete_actions, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = HybridPPOBuffer()

        # 训练统计
        self.train_info = {}

    def select_action(self, state, deterministic=False):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            discrete_probs, alpha, beta_param = self.actor(state_tensor)
            value = self.critic(state_tensor)

        # 离散动作
        dist_discrete = Categorical(discrete_probs)
        if deterministic:
            discrete_action = torch.argmax(discrete_probs, dim=-1)
        else:
            discrete_action = dist_discrete.sample()

        # 连续动作 (Beta分布)
        dist_continuous = Beta(alpha, beta_param)
        if deterministic:
            raw_continuous = alpha / (alpha + beta_param)  # Beta分布的众数
        else:
            raw_continuous = dist_continuous.sample()

        # 缩放到[action_low, action_high]
        continuous_action = (self.continuous_action_low +
                             raw_continuous * (self.continuous_action_high - self.continuous_action_low))

        # 计算log概率
        discrete_logprob = dist_discrete.log_prob(discrete_action)
        continuous_logprob = dist_continuous.log_prob(raw_continuous)

        return (discrete_action.item(), continuous_action.item(),
                discrete_logprob.item(), continuous_logprob.item(),
                value.item())

    def store_transition(self, state, discrete_action, continuous_action,
                         discrete_logprob, continuous_logprob, reward, done, value):
        self.buffer.store(state, discrete_action, continuous_action,
                          discrete_logprob, continuous_logprob, reward, done, value)

    def _compute_gae(self, rewards, values, dones, next_value):
        """计算GAE (Generalized Advantage Estimation)"""
        gae = 0
        advantages = []
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values[:-1]).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, next_value=0.0):
        """PPO更新"""
        if len(self.buffer) == 0:
            return

        # 计算GAE
        advantages, returns = self._compute_gae(
            self.buffer.rewards, self.buffer.values,
            self.buffer.dones, next_value
        )

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        old_discrete_actions = torch.LongTensor(self.buffer.discrete_actions).to(self.device)
        old_continuous_actions = torch.FloatTensor(self.buffer.continuous_actions).to(self.device)
        old_discrete_logprobs = torch.FloatTensor(self.buffer.discrete_logprobs).to(self.device)
        old_continuous_logprobs = torch.FloatTensor(self.buffer.continuous_logprobs).to(self.device)

        # 将连续动作反缩放到[0,1]
        old_raw_continuous = ((old_continuous_actions - self.continuous_action_low) /
                              (self.continuous_action_high - self.continuous_action_low))
        old_raw_continuous = torch.clamp(old_raw_continuous, 1e-6, 1 - 1e-6)

        total_loss_actor = 0.0
        total_loss_critic = 0.0
        total_entropy = 0.0

        for _ in range(self.update_epochs):
            # Mini-batch
            indices = np.arange(len(self.buffer))
            np.random.shuffle(indices)

            for start in range(0, len(self.buffer), self.batch_size):
                end = min(start + self.batch_size, len(self.buffer))
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Actor forward
                discrete_probs, alpha, beta_param = self.actor(batch_states)

                # 离散动作的新log概率
                dist_discrete = Categorical(discrete_probs)
                new_discrete_logprobs = dist_discrete.log_prob(old_discrete_actions[batch_idx])

                # 连续动作的新log概率
                dist_continuous = Beta(alpha, beta_param)
                new_continuous_logprobs = dist_continuous.log_prob(old_raw_continuous[batch_idx])

                # 计算ratio
                discrete_ratio = torch.exp(new_discrete_logprobs - old_discrete_logprobs[batch_idx])
                continuous_ratio = torch.exp(new_continuous_logprobs - old_continuous_logprobs[batch_idx])
                ratio = discrete_ratio * continuous_ratio

                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 熵
                entropy = (dist_discrete.entropy().mean() +
                           dist_continuous.entropy().mean())

                # Actor loss
                actor_loss = policy_loss - self.entropy_coef * entropy

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()

                # Critic loss
                new_values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(new_values, batch_returns)

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

                total_loss_actor += actor_loss.item()
                total_loss_critic += critic_loss.item()
                total_entropy += entropy.item()

        num_updates = self.update_epochs * max(1, len(self.buffer) // self.batch_size)
        self.train_info = {
            "actor_loss": total_loss_actor / num_updates,
            "critic_loss": total_loss_critic / num_updates,
            "entropy": total_entropy / num_updates,
        }

        self.buffer.clear()
        return self.train_info

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
            "optimizer_critic": self.optimizer_critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor"])
        self.optimizer_critic.load_state_dict(checkpoint["optimizer_critic"])

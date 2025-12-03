"""
Actor-Critic networks for hybrid action space
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class Actor(nn.Module):
    """Actor network for hybrid action space"""
    
    def __init__(self, state_dim, num_partition_points, compression_min=0.1, compression_max=1.0):
        """
        :param state_dim: state dimension
        :param num_partition_points: number of partition points
        :param compression_min: minimum compression rate
        :param compression_max: maximum compression rate
        """
        super(Actor, self).__init__()
        
        self.compression_min = compression_min
        self.compression_max = compression_max
        
        # Shared base network
        self.base = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Partition point output (discrete)
        self.partition_header = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_partition_points),
            nn.Softmax(dim=-1)
        )
        
        # Compression rate output (continuous)
        self.compression_mu_header = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.compression_sigma_header = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    
    def forward(self, state):
        """
        Forward pass
        :param state: state tensor [batch_size, state_dim]
        :return: partition probabilities, (compression_mu, compression_sigma)
        """
        code = self.base(state)
        
        # Partition point probability distribution
        prob_partition = self.partition_header(code)
        
        # Compression rate (Gaussian distribution parameters)
        compression_mu = self.compression_mu_header(code) * (self.compression_max - self.compression_min) + self.compression_min
        compression_sigma = self.compression_sigma_header(code) + 1e-6  # Add small value for stability
        
        return prob_partition, (compression_mu, compression_sigma)
    
    def select_action(self, state, deterministic=False):
        """
        Select action from state
        :param state: state tensor
        :param deterministic: whether to use deterministic action
        :return: action dict
        """
        with torch.no_grad():
            prob_partition, (compression_mu, compression_sigma) = self.forward(state)
            
            # Sample partition point
            dist_partition = Categorical(prob_partition)
            if deterministic:
                partition_point = torch.argmax(prob_partition, dim=-1)
            else:
                partition_point = dist_partition.sample()
            
            # Sample compression rate
            dist_compression = Normal(compression_mu, compression_sigma)
            if deterministic:
                compression_rate = compression_mu
            else:
                compression_rate = dist_compression.sample()
            
            compression_rate = torch.clamp(compression_rate, self.compression_min, self.compression_max)
            
            # Convert to Python scalars
            if partition_point.dim() == 0:
                partition_point_val = partition_point.item()
            else:
                partition_point_val = partition_point[0].item() if len(partition_point) > 0 else 0
            
            if compression_rate.dim() == 0:
                compression_rate_val = compression_rate.item()
            else:
                compression_rate_val = compression_rate[0].item() if len(compression_rate) > 0 else 0.5
            
            return {
                'partition_point': partition_point_val,
                'compression_rate': compression_rate_val
            }
    
    def evaluate_action(self, state, action):
        """
        Evaluate action probability and entropy
        :param state: state tensor
        :param action: action dict
        :return: log_prob, entropy
        """
        prob_partition, (compression_mu, compression_sigma) = self.forward(state)

        # 将 Python 标量动作转换为张量，保证与分布接口兼容
        if isinstance(action["partition_point"], torch.Tensor):
            partition_action = action["partition_point"]
        else:
            partition_action = torch.tensor(
                action["partition_point"],
                dtype=torch.long,
                device=prob_partition.device,
            )
        if isinstance(action["compression_rate"], torch.Tensor):
            compression_action = action["compression_rate"]
        else:
            compression_action = torch.tensor(
                action["compression_rate"],
                dtype=torch.float32,
                device=prob_partition.device,
            )

        # Partition point
        dist_partition = Categorical(prob_partition)
        partition_log_prob = dist_partition.log_prob(partition_action)
        partition_entropy = dist_partition.entropy()
        
        # Compression rate
        dist_compression = Normal(compression_mu, compression_sigma)
        compression_log_prob = dist_compression.log_prob(compression_action)
        compression_entropy = dist_compression.entropy()
        
        # Total log probability
        total_log_prob = partition_log_prob + compression_log_prob
        total_entropy = partition_entropy + compression_entropy
        
        return total_log_prob, total_entropy


class Critic(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_dim):
        """
        :param state_dim: state dimension
        """
        super(Critic, self).__init__()
        self.net = nn.Sequential(
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
        Forward pass
        :param state: state tensor [batch_size, state_dim]
        :return: value [batch_size, 1]
        """
        return self.net(state)


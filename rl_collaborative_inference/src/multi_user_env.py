"""
Multi-user reinforcement learning environment for collaborative inference.

基于单用户的 CollaborativeInferenceEnv，扩展为支持 N 个用户共享带宽和边缘/云资源。
主要用于生成论文中多用户累计奖励曲线等实验数据。
"""

from typing import Dict, Tuple, Any, List

import numpy as np
import torch

from src.env import CollaborativeInferenceEnv


class MultiUserEnv:
    """
    Multi-user wrapper over CollaborativeInferenceEnv.

    设计目标（论文实验友好而非完美系统建模）：
    - 支持可配置用户数 num_users；
    - 每个 step 为所有用户各执行一次动作；
    - reward 为所有用户 reward 的加权和（默认平均），便于画一条全局累计奖励曲线；
    - 复用单用户 env 的实现，保证与现有代码兼容。
    """

    def __init__(
        self,
        base_env: CollaborativeInferenceEnv,
        num_users: int = 3,
        reward_aggregation: str = "mean",
    ):
        """
        :param base_env: 已初始化好的单用户 CollaborativeInferenceEnv，
                         其 dataset / model / bandwidth 等作为所有用户的公共环境。
        :param num_users: 用户数量（智能体数量）
        :param reward_aggregation: 'mean' 或 'sum'，全局 reward 的聚合方式
        """
        assert num_users >= 1, "num_users must be >= 1"
        assert reward_aggregation in ("mean", "sum")

        self.base_env = base_env
        self.num_users = num_users
        self.reward_aggregation = reward_aggregation

        # 为每个用户维护一个独立的 env 副本（共享模型与数据迭代器引用即可）
        self.user_envs: List[CollaborativeInferenceEnv] = []
        for _ in range(num_users):
            # 这里简单地复用同一模型与数据集引用，并复制关键标量参数
            env = CollaborativeInferenceEnv(
                model=base_env.model,
                dataset=base_env.dataset,
                edge_device=base_env.edge_device,
                cloud_device=base_env.cloud_device,
                network_bandwidth=base_env.network_bandwidth / num_users,
                pruning_type=base_env.pruning_manager.pruning_type,
                target_accuracy=base_env.target_accuracy,
                max_latency=base_env.max_latency,
                alpha=base_env.alpha,
                beta=base_env.beta,
            )
            self.user_envs.append(env)

        # 对外暴露的 action 维度 = 每个用户动作的字典列表
        self.num_partition_points = self.user_envs[0].num_partition_points
        self.state_dim = base_env.state_space.state_dim

    def reset(self) -> List[np.ndarray]:
        """Reset all user environments and返回所有用户的 state 列表。"""
        states: List[np.ndarray] = []
        for env in self.user_envs:
            states.append(env.reset())
        return states

    def step(
        self, actions: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], float, bool, Dict[str, Any]]:
        """
        对所有用户执行一步动作。

        :param actions: 长度为 num_users 的动作列表，
                        每个元素是 {'partition_point': int, 'compression_rate': float}
        :return: next_states(list), global_reward(float), done(bool), info(dict)
        """
        assert len(actions) == self.num_users, "actions length must equal num_users"

        next_states: List[np.ndarray] = []
        rewards: List[float] = []
        infos: List[Dict[str, Any]] = []

        for env, action in zip(self.user_envs, actions):
            n_state, r, done, info = env.step(action)
            next_states.append(n_state)
            rewards.append(r)
            infos.append(info)

        # 全局 reward 聚合
        if self.reward_aggregation == "mean":
            global_reward = float(np.mean(rewards))
        else:
            global_reward = float(np.sum(rewards))

        # Multi-user 环境视作持续任务，这里统一返回 done=False，
        # 若后续需要 episode 结束逻辑，可在外部按 step 数自行截断。
        done = False

        # 额外统计信息：各用户平均 latency / accuracy 等
        avg_latency = float(np.mean([inf["latency"] for inf in infos]))
        avg_accuracy = float(np.mean([inf["accuracy"] for inf in infos]))

        info = {
            "per_user_rewards": rewards,
            "per_user_infos": infos,
            "avg_latency": avg_latency,
            "avg_accuracy": avg_accuracy,
        }

        return next_states, global_reward, done, info



"""
运行一系列消融 & 基线实验，自动生成用于画论文曲线的 JSON 数据。

主要输出三类结果（保存在 experiments/ 目录下）：
1) baseline_vs_rl.json       —— Local / JALAD / RL 的累计奖励对比（单用户）
2) ablation_hyperparams.json —— 学习率 / update_freq / buffer_size 等超参消融
3) multi_user_scaling.json   —— 不同用户数量下的累计奖励（基于 MultiUserEnv）
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from src.actor_critic import Actor, Critic
from src.env import CollaborativeInferenceEnv
from src.multi_user_env import MultiUserEnv
from src.ppo import PPO, PPOBuffer
from src.dataset_loader import get_caltech101_dataloader
from models.AlexNet import AlexNet


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_single_env(args) -> CollaborativeInferenceEnv:
    dataloader, dataset = get_caltech101_dataloader(
        args.data_dir, batch_size=1, split="train", num_workers=0
    )
    model = AlexNet(input_channels=3, num_classes=101)
    model.eval()
    env = CollaborativeInferenceEnv(
        model=model,
        dataset=dataset,
        edge_device=args.device,
        cloud_device=args.device,
        network_bandwidth=args.network_bandwidth,
        pruning_type=args.pruning_type,
        target_accuracy=args.target_accuracy,
        max_latency=args.max_latency,
        alpha=args.alpha,
        beta=args.beta,
    )
    return env


def build_agent(env: CollaborativeInferenceEnv, args, lr_actor=None, lr_critic=None, buffer_size=None) -> PPO:
    # 动态从环境 reset 得到 state 维度，避免与 StateSpace 中手动配置不一致
    sample_state = env.reset()
    state_dim = len(sample_state)
    num_partition_points = env.num_partition_points

    actor = Actor(state_dim, num_partition_points).to(args.device)
    critic = Critic(state_dim).to(args.device)

    agent = PPO(
        actor=actor,
        critic=critic,
        lr_actor=lr_actor if lr_actor is not None else args.lr_actor,
        lr_critic=lr_critic if lr_critic is not None else args.lr_critic,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        entropy_coef=args.entropy_coef,
    )
    if buffer_size is not None:
        agent.buffer = PPOBuffer(capacity=buffer_size)
    return agent


def run_rl_training(env: CollaborativeInferenceEnv, agent: PPO, args, max_steps: int) -> Dict[str, List[float]]:
    """
    运行一次 RL 训练，返回用于画图的时间序列数据：
    - rewards: 每 step 累积 reward（移动平均）
    - value_loss: 每个 update epoch 的 critic loss 近似（这里只保存 returns 与 values 的 MSE 均值）
    """
    state = env.reset()
    episode_reward = 0.0
    episode_rewards = []
    cumulative_rewards = []
    value_losses = []

    reward_ma = 0.0
    ma_alpha = 0.01  # 用于平滑累计奖励

    for step in tqdm(range(max_steps), desc="RL training"):
        action, log_prob, value, entropy = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        agent.buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
        )

        state = next_state
        episode_reward += reward

        # 积累 reward（用滑动平均代替单 episode 和）
        reward_ma = (1 - ma_alpha) * reward_ma + ma_alpha * reward
        cumulative_rewards.append(reward_ma)

        # 定期更新
        if len(agent.buffer) >= args.batch_size and step % args.update_freq == 0:
            # 为了记录 value loss，手动计算一次 returns / values 的 MSE
            batch = agent.buffer.sample(args.batch_size)
            values = torch.FloatTensor([b["value"] for b in batch])
            rewards = torch.FloatTensor([b["reward"] for b in batch])
            dones = torch.FloatTensor([b["done"] for b in batch])
            returns, _ = agent._compute_gae(rewards, dones, values)
            critic_loss = torch.mean((returns - values) ** 2).item()
            value_losses.append(critic_loss)

            agent.update(batch_size=args.batch_size)

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            state = env.reset()

    return {
        "cumulative_rewards": cumulative_rewards,
        "value_losses": value_losses,
        "agent": agent,  # 返回agent以便保存模型
    }


def run_fixed_policy(env: CollaborativeInferenceEnv, policy_name: str, max_steps: int) -> List[float]:
    """
    在不使用学习的前提下运行固定策略（Local / JALAD），仅用于基线累计奖励曲线。
    """
    state = env.reset()
    reward_ma = 0.0
    ma_alpha = 0.01
    cumulative_rewards = []

    # 简单策略：根据策略名固定 partition_point 与 compression_rate
    if policy_name.lower() == "local":
        # 全部在本地推理：partition_point 设为最后一个，compression_rate=1.0
        partition_idx = env.num_partition_points - 1
        compression_rate = 1.0
    else:  # "jalad" 近似：中间分割+较高压缩率
        partition_idx = max(0, env.num_partition_points // 2)
        compression_rate = 0.5

    for _ in tqdm(range(max_steps), desc=f"{policy_name} policy"):
        action = {
            "partition_point": partition_idx,
            "compression_rate": compression_rate,
        }
        next_state, reward, done, info = env.step(action)
        reward_ma = (1 - ma_alpha) * reward_ma + ma_alpha * reward
        cumulative_rewards.append(reward_ma)
        state = next_state

    return cumulative_rewards


def experiment_baseline_vs_rl(args, output_dir: str):
    env = build_single_env(args)

    # Local / JALAD 基线
    local_curve = run_fixed_policy(env, "Local", args.max_steps_baseline)
    jalad_curve = run_fixed_policy(env, "JALAD", args.max_steps_baseline)

    # RL 方法
    agent = build_agent(env, args)
    rl_result = run_rl_training(env, agent, args, args.max_steps_rl)

    result = {
        "time_frames": list(range(len(rl_result["cumulative_rewards"]))),
        "local": local_curve,
        "jalad": jalad_curve,
        "rl": rl_result["cumulative_rewards"],
    }

    with open(os.path.join(output_dir, "baseline_vs_rl.json"), "w") as f:
        json.dump(result, f, indent=2)
    
    # 保存训练好的模型
    if "agent" in rl_result:
        model_path = os.path.join(output_dir, "final_model.pt")
        rl_result["agent"].save(model_path)
        print(f"Saved trained model to {model_path}")


def experiment_ablation(args, output_dir: str):
    """
    学习率 / update_freq / buffer_size 消融。
    """
    env = build_single_env(args)

    ablation_result = {
        "learning_rate": {},
        "update_freq": {},
        "buffer_size": {},
        "value_loss_buffer_size": {},
    }

    # 1) learning rate
    for lr in args.lr_list:
        set_seed(args.seed)
        agent = build_agent(env, args, lr_actor=lr, lr_critic=lr)
        curves = run_rl_training(env, agent, args, args.max_steps_rl)
        ablation_result["learning_rate"][str(lr)] = curves["cumulative_rewards"]

    # 2) update_freq
    for uf in args.update_freq_list:
        set_seed(args.seed)
        tmp_args = argparse.Namespace(**vars(args))
        tmp_args.update_freq = uf
        agent = build_agent(env, tmp_args)
        curves = run_rl_training(env, agent, tmp_args, tmp_args.max_steps_rl)
        ablation_result["update_freq"][str(uf)] = curves["cumulative_rewards"]

    # 3) buffer_size & value loss
    for buf in args.buffer_size_list:
        set_seed(args.seed)
        agent = build_agent(env, args, buffer_size=buf)
        curves = run_rl_training(env, agent, args, args.max_steps_rl)
        ablation_result["buffer_size"][str(buf)] = curves["cumulative_rewards"]
        ablation_result["value_loss_buffer_size"][str(buf)] = curves["value_losses"]

    with open(os.path.join(output_dir, "ablation_hyperparams.json"), "w") as f:
        json.dump(ablation_result, f, indent=2)


def experiment_multi_user_scaling(args, output_dir: str):
    """
    基于 MultiUserEnv，不同用户数量 (3~10) 的累计奖励曲线。
    """
    base_env = build_single_env(args)
    scaling_result: Dict[str, List[float]] = {}

    for num_users in args.num_users_list:
        set_seed(args.seed)
        mu_env = MultiUserEnv(base_env, num_users=num_users, reward_aggregation="mean")

        # 对多用户场景，这里使用一个共享参数的集中式 agent：
        # 从任意一个用户的 state 动态推断 state_dim，保证与环境一致
        sample_states = mu_env.reset()
        state_dim = len(sample_states[0])
        num_partition_points = mu_env.num_partition_points

        actor = Actor(state_dim, num_partition_points).to(args.device)
        critic = Critic(state_dim).to(args.device)
        agent = PPO(
            actor=actor,
            critic=critic,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            eps_clip=args.eps_clip,
            k_epochs=args.k_epochs,
            entropy_coef=args.entropy_coef,
        )

        # 初始化每个用户的 state
        states = mu_env.reset()
        reward_ma = 0.0
        ma_alpha = 0.01
        cumulative_rewards: List[float] = []

        for step in tqdm(range(args.max_steps_multi_user), desc=f"{num_users} users"):
            actions = []
            for s in states:
                action, log_prob, value, entropy = agent.select_action(s)
                actions.append(action)

            next_states, global_reward, done, info = mu_env.step(actions)

            # 这里只训练一个集中式 agent：用所有用户状态的均值近似
            mean_state = np.mean(np.stack(states, axis=0), axis=0)
            mean_next_state = np.mean(np.stack(next_states, axis=0), axis=0)

            agent.buffer.push(
                state=mean_state,
                action=actions[0],  # 动作结构相同，取第一个即可
                reward=global_reward,
                next_state=mean_next_state,
                done=False,
                log_prob=0.0,  # 仅用于 advantage 估计，这里简化处理
                value=0.0,
            )

            states = next_states

            reward_ma = (1 - ma_alpha) * reward_ma + ma_alpha * global_reward
            cumulative_rewards.append(reward_ma)

            if len(agent.buffer) >= args.batch_size and step % args.update_freq == 0:
                agent.update(batch_size=args.batch_size)

        scaling_result[str(num_users)] = cumulative_rewards

    with open(os.path.join(output_dir, "multi_user_scaling.json"), "w") as f:
        json.dump(scaling_result, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/caltech-101")
    parser.add_argument("--output_root", type=str, default="./experiments")
    parser.add_argument("--network_bandwidth", type=float, default=10.0)
    parser.add_argument("--pruning_type", type=str, default="structured")
    parser.add_argument("--target_accuracy", type=float, default=0.95)
    parser.add_argument("--max_latency", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)

    # PPO 超参（默认同 train.py）
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--k_epochs", type=int, default=10)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_freq", type=int, default=10)

    # 实验步数
    parser.add_argument("--max_steps_baseline", type=int, default=500000)
    parser.add_argument("--max_steps_rl", type=int, default=500000)
    parser.add_argument("--max_steps_multi_user", type=int, default=500000)

    # 消融范围
    parser.add_argument(
        "--lr_list",
        type=float,
        nargs="+",
        default=[1e-5, 1e-4, 3e-4, 1e-3, 1e-2],
    )
    parser.add_argument(
        "--update_freq_list",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40, 80],
    )
    parser.add_argument(
        "--buffer_size_list",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096],
    )
    parser.add_argument(
        "--num_users_list",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8, 9, 10],
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    # 为了避免现有 PPO/Actor-Critic 中的 device 不一致问题，这里统一使用 CPU 运行实验，
    # 也便于在不同环境下复现实验结果。如需使用 GPU，可后续统一在 PPO 与更新逻辑中加入 .to(device)。
    args.device = "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, f"ablation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 运行三类实验
    experiment_baseline_vs_rl(args, output_dir)
    experiment_ablation(args, output_dir)
    experiment_multi_user_scaling(args, output_dir)

    print(f"All ablation experiments finished. Results saved to {output_dir}")


if __name__ == "__main__":
    main()



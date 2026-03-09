"""
ARL-Comp 训练脚本
训练Hybrid PPO智能体在边云协同推理环境中学习最优的(分区点, 压缩率)策略
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import time
from collections import defaultdict

from utils.inference_utils import get_dnn_model
from arl_comp.model_profiler import profile_model
from arl_comp.partition_filter import filter_partition_points
from arl_comp.env.arl_env import ARLCompEnv
from arl_comp.agent.hybrid_ppo import HybridPPO


def train_arl_comp(model_type="alex_net",
                   num_episodes=300,
                   max_steps_per_episode=50,
                   bandwidth_range=(1, 50),
                   base_accuracy=0.85,
                   weight_latency=0.5,
                   weight_accuracy=0.5,
                   use_filter=True,
                   hidden_dim=128,
                   lr_actor=3e-4,
                   lr_critic=1e-3,
                   gamma=0.99,
                   gae_lambda=0.95,
                   update_interval=2048,
                   device="cpu",
                   seed=42,
                   save_dir="arl_comp/results",
                   verbose=True):
    """
    训练ARL-Comp智能体

    Args:
        model_type: 模型类型 (alex_net, vgg_net, le_net, mobile_net)
        num_episodes: 训练episode数
        max_steps_per_episode: 每个episode最大步数
        bandwidth_range: 带宽范围 (Mbps)
        base_accuracy: 基准准确率
        weight_latency: 延迟权重
        weight_accuracy: 准确率权重
        use_filter: 是否使用分区点筛选算法
        device: 运行设备

    Returns:
        agent: 训练后的智能体
        train_log: 训练日志
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    # 1. 加载模型
    if verbose:
        print(f"[ARL-Comp] 加载模型: {model_type}")
    model = get_dnn_model(model_type)

    # 2. 模型profiling
    if verbose:
        print(f"[ARL-Comp] 模型profiling中...")
    layer_profiles = profile_model(model, device=device)
    num_layers = len(layer_profiles)
    if verbose:
        print(f"[ARL-Comp] 模型共 {num_layers} 层")

    # 3. 分区点筛选
    feasible_points = None
    filter_info = None
    if use_filter:
        if verbose:
            print(f"\n[ARL-Comp] 执行分区点预筛选算法...")
        avg_bw = np.mean(bandwidth_range)
        feasible_points, filter_info = filter_partition_points(
            model, layer_profiles, avg_bw,
            base_accuracy=base_accuracy,
            verbose=verbose
        )

    # 4. 创建环境
    env = ARLCompEnv(
        model=model,
        layer_profiles=layer_profiles,
        feasible_points=feasible_points,
        bandwidth_range=bandwidth_range,
        base_accuracy=base_accuracy,
        weight_latency=weight_latency,
        weight_accuracy=weight_accuracy,
        max_steps=max_steps_per_episode,
    )

    if verbose:
        print(f"\n[ARL-Comp] 环境创建完成")
        print(f"  状态空间维度: {env.state_dim}")
        print(f"  离散动作数 (分区点): {env.num_discrete_actions}")
        print(f"  连续动作范围 (压缩率): [{env.continuous_action_low}, {env.continuous_action_high}]")

    # 5. 创建智能体
    agent = HybridPPO(
        state_dim=env.state_dim,
        num_discrete_actions=env.num_discrete_actions,
        continuous_action_low=env.continuous_action_low,
        continuous_action_high=env.continuous_action_high,
        hidden_dim=hidden_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        device=device,
    )

    # 6. 训练
    if verbose:
        print(f"\n[ARL-Comp] 开始训练, 共 {num_episodes} episodes...")
        print("=" * 80)

    train_log = {
        "episode_rewards": [],
        "episode_latencies": [],
        "episode_accuracies": [],
        "episode_partition_points": [],
        "episode_compression_ratios": [],
        "actor_losses": [],
        "critic_losses": [],
        "entropies": [],
    }

    total_steps = 0
    best_reward = -float("inf")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_latencies = []
        episode_accuracies = []
        episode_pps = []
        episode_crs = []

        for step in range(max_steps_per_episode):
            # 选择动作
            (discrete_action, continuous_action,
             d_logprob, c_logprob, value) = agent.select_action(state)

            # 执行动作
            next_state, reward, done, info = env.step(discrete_action, continuous_action)

            # 存储经验
            agent.store_transition(
                state, discrete_action, continuous_action,
                d_logprob, c_logprob, reward, done, value
            )

            episode_reward += reward
            episode_latencies.append(info["total_latency_ms"])
            episode_accuracies.append(info["accuracy"])
            episode_pps.append(info["partition_point"])
            episode_crs.append(info["compression_ratio"])

            state = next_state
            total_steps += 1

            if done:
                break

        # 更新策略 (每个episode结束时)
        update_info = agent.update()

        # 记录日志
        avg_reward = episode_reward / max_steps_per_episode
        avg_latency = np.mean(episode_latencies)
        avg_accuracy = np.mean(episode_accuracies)
        most_common_pp = max(set(episode_pps), key=episode_pps.count)
        avg_cr = np.mean(episode_crs)

        train_log["episode_rewards"].append(avg_reward)
        train_log["episode_latencies"].append(avg_latency)
        train_log["episode_accuracies"].append(avg_accuracy)
        train_log["episode_partition_points"].append(most_common_pp)
        train_log["episode_compression_ratios"].append(avg_cr)

        if update_info:
            train_log["actor_losses"].append(update_info["actor_loss"])
            train_log["critic_losses"].append(update_info["critic_loss"])
            train_log["entropies"].append(update_info["entropy"])

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(os.path.join(save_dir, f"best_model_{model_type}.pth"))

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:4d}/{num_episodes} | "
                  f"Reward: {avg_reward:.4f} | "
                  f"Latency: {avg_latency:.2f}ms | "
                  f"Accuracy: {avg_accuracy:.4f} | "
                  f"PP: {most_common_pp} | "
                  f"CR: {avg_cr:.3f} | "
                  f"A_Loss: {update_info['actor_loss']:.4f} | "
                  f"C_Loss: {update_info['critic_loss']:.4f}")

    # 保存最终模型和训练日志
    agent.save(os.path.join(save_dir, f"final_model_{model_type}.pth"))

    # 保存训练日志 (转换numpy类型以便JSON序列化)
    log_serializable = {}
    for k, v in train_log.items():
        log_serializable[k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]

    with open(os.path.join(save_dir, f"train_log_{model_type}.json"), "w") as f:
        json.dump(log_serializable, f, indent=2)

    if filter_info:
        with open(os.path.join(save_dir, f"filter_info_{model_type}.json"), "w") as f:
            # 只保存可序列化的部分
            serializable_info = {
                "total_candidates": filter_info["total_candidates"],
                "median_latency_ms": float(filter_info["median_latency_ms"]),
                "feasible_points": filter_info["feasible_points"],
            }
            json.dump(serializable_info, f, indent=2)

    if verbose:
        print("=" * 80)
        print(f"[ARL-Comp] 训练完成!")
        print(f"  最佳平均奖励: {best_reward:.4f}")
        print(f"  模型保存至: {save_dir}")

    return agent, train_log, filter_info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARL-Comp Training")
    parser.add_argument("--model", type=str, default="alex_net",
                        choices=["alex_net", "vgg_net", "le_net", "mobile_net"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-filter", action="store_true")
    args = parser.parse_args()

    train_arl_comp(
        model_type=args.model,
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        use_filter=not args.no_filter,
        device=args.device,
        seed=args.seed,
    )

"""
基于 run_ablation_experiments.py 生成的 JSON，一键画出类似论文的图。

示例：
    python plot_paper_style.py --exp_dir experiments/ablation_20251126_120000
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def _smooth_curve(y, window=50):
    """
    简单滑动平均平滑，保持输入与输出长度一致，避免 x/y 维度不匹配。
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    if len(y) < window or window <= 1:
        return y
    kernel = np.ones(window) / float(window)
    # 使用 'same' 模式保持长度一致
    return np.convolve(y, kernel, mode="same")


def plot_baseline_vs_rl(exp_dir: str, save_path: str):
    with open(os.path.join(exp_dir, "baseline_vs_rl.json"), "r") as f:
        data = json.load(f)

    plt.figure(figsize=(5, 3.5))
    plt.grid(ls="--", alpha=0.5)

    local_y = _smooth_curve(data["local"])
    jalad_y = _smooth_curve(data["jalad"])
    rl_y = _smooth_curve(data["rl"])

    plt.plot(np.arange(len(local_y)), local_y, label="Local", color="gray")
    plt.plot(np.arange(len(jalad_y)), jalad_y, label="JALAD", color="C0")
    plt.plot(np.arange(len(rl_y)), rl_y, label="MAHPPO", color="C2")

    plt.xlabel("Time Frame")
    plt.ylabel("Cumulative Reward")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ablation(exp_dir: str, save_prefix: str):
    with open(os.path.join(exp_dir, "ablation_hyperparams.json"), "r") as f:
        data = json.load(f)

    # (a) 学习率
    plt.figure(figsize=(5, 3.5))
    plt.grid(ls="--", alpha=0.5)
    for lr, curve in data["learning_rate"].items():
        y = _smooth_curve(curve)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"LR={lr}")
    plt.xlabel("Time Frame")
    plt.ylabel("Cumulative Reward")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_prefix + "_ablation_lr.png", dpi=300)
    plt.close()

    # (b) update_freq 近似 reuse time
    plt.figure(figsize=(5, 3.5))
    plt.grid(ls="--", alpha=0.5)
    for uf, curve in data["update_freq"].items():
        y = _smooth_curve(curve)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"Reuse Time={uf}")
    plt.xlabel("Time Frame")
    plt.ylabel("Cumulative Reward")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_prefix + "_ablation_reuse.png", dpi=300)
    plt.close()

    # (c) buffer_size
    plt.figure(figsize=(5, 3.5))
    plt.grid(ls="--", alpha=0.5)
    for buf, curve in data["buffer_size"].items():
        y = _smooth_curve(curve)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"Memory Size={buf}")
    plt.xlabel("Time Frame")
    plt.ylabel("Cumulative Reward")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_prefix + "_ablation_mem.png", dpi=300)
    plt.close()

    # (d) value loss vs time for different buffer_size
    plt.figure(figsize=(5, 3.5))
    plt.grid(ls="--", alpha=0.5)
    for buf, values in data["value_loss_buffer_size"].items():
        y = _smooth_curve(values, window=10)
        x = np.arange(len(y))
        plt.plot(x, y, label=f"Memory Size={buf}")
    plt.xlabel("Time Frame")
    plt.ylabel("Value Loss")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_prefix + "_value_loss.png", dpi=300)
    plt.close()


def plot_multi_user(exp_dir: str, save_path: str):
    with open(os.path.join(exp_dir, "multi_user_scaling.json"), "r") as f:
        data = json.load(f)

    plt.figure(figsize=(5, 3.5))
    plt.grid(ls="--", alpha=0.5)

    # 为了和论文配色接近，按照用户数排序
    for idx, (num_users, curve) in enumerate(sorted(data.items(), key=lambda x: int(x[0]))):
        y = _smooth_curve(curve)
        x = np.arange(len(y))
        plt.plot(
            x,
            y,
            label=f"{num_users} Users",
        )

    plt.xlabel("Time Frame")
    plt.ylabel("Cumulative Reward")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="ablation 结果目录")
    args = parser.parse_args()

    # 检查文件是否存在，只生成可用的图
    baseline_file = os.path.join(args.exp_dir, "baseline_vs_rl.json")
    ablation_file = os.path.join(args.exp_dir, "ablation_hyperparams.json")
    multi_user_file = os.path.join(args.exp_dir, "multi_user_scaling.json")
    
    if os.path.exists(baseline_file):
        print(f"Generating baseline vs RL figure...")
        plot_baseline_vs_rl(args.exp_dir, os.path.join(args.exp_dir, "fig_baseline_vs_rl.png"))
    else:
        print(f"Warning: {baseline_file} not found, skipping baseline vs RL figure")
    
    if os.path.exists(ablation_file):
        print(f"Generating ablation figures...")
        plot_ablation(args.exp_dir, os.path.join(args.exp_dir, "fig_ablation"))
    else:
        print(f"Warning: {ablation_file} not found, skipping ablation figures")
    
    if os.path.exists(multi_user_file):
        print(f"Generating multi-user scaling figure...")
        plot_multi_user(args.exp_dir, os.path.join(args.exp_dir, "fig_multi_user_scaling.png"))
    else:
        print(f"Warning: {multi_user_file} not found, skipping multi-user scaling figure")

    print("All available figures saved to", args.exp_dir)


if __name__ == "__main__":
    main()



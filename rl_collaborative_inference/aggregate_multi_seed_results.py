"""
汇总多个随机种子的训练结果，计算均值±标准差，生成最终论文图
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pathlib import Path


def load_json_safe(filepath: str) -> dict:
    """安全加载JSON文件"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_baseline_vs_rl(seed_dirs: List[str], output_dir: str):
    """汇总 baseline vs RL 结果"""
    all_data = {"local": [], "jalad": [], "rl": [], "time_frames": None}
    
    for seed_dir in seed_dirs:
        json_file = os.path.join(seed_dir, "baseline_vs_rl.json")
        data = load_json_safe(json_file)
        if data is None:
            continue
        
        if all_data["time_frames"] is None:
            all_data["time_frames"] = data.get("time_frames", list(range(len(data.get("rl", [])))))
        
        all_data["local"].append(data.get("local", []))
        all_data["jalad"].append(data.get("jalad", []))
        all_data["rl"].append(data.get("rl", []))
    
    if not all_data["local"]:
        print("Warning: No baseline_vs_rl.json files found")
        return None
    
    # 对齐长度
    min_len = min(len(seq) for seq in all_data["local"] + all_data["jalad"] + all_data["rl"])
    all_data["local"] = [seq[:min_len] for seq in all_data["local"]]
    all_data["jalad"] = [seq[:min_len] for seq in all_data["jalad"]]
    all_data["rl"] = [seq[:min_len] for seq in all_data["rl"]]
    all_data["time_frames"] = all_data["time_frames"][:min_len]
    
    # 计算均值和标准差
    result = {
        "time_frames": all_data["time_frames"],
        "local": {
            "mean": np.mean(all_data["local"], axis=0).tolist(),
            "std": np.std(all_data["local"], axis=0).tolist()
        },
        "jalad": {
            "mean": np.mean(all_data["jalad"], axis=0).tolist(),
            "std": np.std(all_data["jalad"], axis=0).tolist()
        },
        "rl": {
            "mean": np.mean(all_data["rl"], axis=0).tolist(),
            "std": np.std(all_data["rl"], axis=0).tolist()
        }
    }
    
    # 保存汇总结果
    output_file = os.path.join(output_dir, "aggregated_baseline_vs_rl.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # 生成带误差带的图
    plot_aggregated_baseline_vs_rl(result, os.path.join(output_dir, "fig_baseline_vs_rl_aggregated.png"))
    
    return result


def plot_aggregated_baseline_vs_rl(data: dict, save_path: str):
    """绘制带误差带的 baseline vs RL 图"""
    x = np.array(data["time_frames"])
    
    plt.figure(figsize=(8, 5))
    plt.grid(ls="--", alpha=0.5)
    
    # Local
    mean = np.array(data["local"]["mean"])
    std = np.array(data["local"]["std"])
    plt.plot(x, mean, label="Local", color="gray", linewidth=2)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, color="gray")
    
    # JALAD
    mean = np.array(data["jalad"]["mean"])
    std = np.array(data["jalad"]["std"])
    plt.plot(x, mean, label="JALAD", color="C0", linewidth=2)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, color="C0")
    
    # RL
    mean = np.array(data["rl"]["mean"])
    std = np.array(data["rl"]["std"])
    plt.plot(x, mean, label="MAHPPO", color="C2", linewidth=2)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2, color="C2")
    
    plt.xlabel("Time Frame", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregated baseline vs RL figure to {save_path}")


def aggregate_multi_user_scaling(seed_dirs: List[str], output_dir: str):
    """汇总多用户扩展性结果"""
    all_data: Dict[str, List[List[float]]] = {}
    
    for seed_dir in seed_dirs:
        json_file = os.path.join(seed_dir, "multi_user_scaling.json")
        data = load_json_safe(json_file)
        if data is None:
            continue
        
        for num_users, curve in data.items():
            if num_users not in all_data:
                all_data[num_users] = []
            all_data[num_users].append(curve)
    
    if not all_data:
        print("Warning: No multi_user_scaling.json files found")
        return None
    
    # 对齐长度并计算统计量
    result = {}
    for num_users, curves in all_data.items():
        min_len = min(len(c) for c in curves)
        curves_aligned = [c[:min_len] for c in curves]
        result[num_users] = {
            "mean": np.mean(curves_aligned, axis=0).tolist(),
            "std": np.std(curves_aligned, axis=0).tolist()
        }
    
    # 保存汇总结果
    output_file = os.path.join(output_dir, "aggregated_multi_user_scaling.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # 生成图
    plot_aggregated_multi_user_scaling(result, os.path.join(output_dir, "fig_multi_user_scaling_aggregated.png"))
    
    return result


def plot_aggregated_multi_user_scaling(data: dict, save_path: str):
    """绘制带误差带的多用户扩展性图"""
    plt.figure(figsize=(8, 5))
    plt.grid(ls="--", alpha=0.5)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    user_counts = sorted([int(k) for k in data.keys()])
    
    for i, num_users in enumerate(user_counts):
        key = str(num_users)
        mean = np.array(data[key]["mean"])
        std = np.array(data[key]["std"])
        x = np.arange(len(mean))
        
        plt.plot(x, mean, label=f"{num_users} Users", color=colors[i], linewidth=2)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[i])
    
    plt.xlabel("Time Frame", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend(frameon=True, fontsize=10, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregated multi-user scaling figure to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="汇总多个随机种子的实验结果")
    parser.add_argument("--seed_dirs", type=str, nargs="+", required=True,
                       help="各个seed的实验结果目录（包含 ablation_* 子目录）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="汇总结果输出目录")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 找到每个seed的ablation目录
    ablation_dirs = []
    for seed_dir in args.seed_dirs:
        # 查找 ablation_* 子目录
        for item in os.listdir(seed_dir):
            item_path = os.path.join(seed_dir, item)
            if os.path.isdir(item_path) and item.startswith("ablation_"):
                ablation_dirs.append(item_path)
                break
    
    if not ablation_dirs:
        print("Error: No ablation directories found in seed directories")
        return
    
    print(f"Found {len(ablation_dirs)} experiment directories to aggregate")
    
    # 汇总各类结果
    print("\n1. Aggregating baseline vs RL results...")
    aggregate_baseline_vs_rl(ablation_dirs, args.output_dir)
    
    print("\n2. Aggregating multi-user scaling results...")
    aggregate_multi_user_scaling(ablation_dirs, args.output_dir)
    
    print(f"\nAll aggregated results saved to {args.output_dir}")


if __name__ == "__main__":
    main()


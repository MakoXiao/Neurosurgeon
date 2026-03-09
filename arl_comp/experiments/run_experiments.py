"""
ARL-Comp 实验运行器
生成所有实验对比图和数据表, 用于论文撰写

实验内容:
1. RL收敛曲线图 (Reward, Latency, Accuracy vs Episode)
2. 不同方法的延迟对比 (ARL-Comp vs Neurosurgeon vs 纯本地 vs 纯云端)
3. 不同方法的准确率对比
4. 不同带宽下的性能对比
5. 分区点筛选算法效果分析
6. 压缩率与性能的关系
7. 不同模型上的对比实验
8. 消融实验 (有/无筛选, 有/无压缩)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from utils.inference_utils import get_dnn_model
from arl_comp.model_profiler import profile_model, compute_partition_latency
from arl_comp.partition_filter import filter_partition_points
from arl_comp.pruning.pruner import estimate_accuracy_after_compression
from arl_comp.env.arl_env import ARLCompEnv
from arl_comp.agent.hybrid_ppo import HybridPPO
from arl_comp.train import train_arl_comp

# 设置中文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.pad_inches'] = 0.1

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def smooth_curve(data, weight=0.9):
    """指数加权平滑"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# =============================================================================
# 实验1: RL收敛曲线
# =============================================================================
def plot_convergence_curves(train_log, model_type, save_dir=FIGURES_DIR):
    """绘制RL训练收敛曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'ARL-Comp Training Convergence ({model_type})', fontsize=16, fontweight='bold')

    episodes = list(range(1, len(train_log["episode_rewards"]) + 1))

    # Reward曲线
    ax = axes[0, 0]
    raw = train_log["episode_rewards"]
    smoothed = smooth_curve(raw)
    ax.plot(episodes, raw, alpha=0.3, color='blue', label='Raw')
    ax.plot(episodes, smoothed, color='blue', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('(a) Reward Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Latency曲线
    ax = axes[0, 1]
    raw = train_log["episode_latencies"]
    smoothed = smooth_curve(raw)
    ax.plot(episodes, raw, alpha=0.3, color='red', label='Raw')
    ax.plot(episodes, smoothed, color='red', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Latency (ms)')
    ax.set_title('(b) Latency Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy曲线
    ax = axes[0, 2]
    raw = train_log["episode_accuracies"]
    smoothed = smooth_curve(raw)
    ax.plot(episodes, raw, alpha=0.3, color='green', label='Raw')
    ax.plot(episodes, smoothed, color='green', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Accuracy')
    ax.set_title('(c) Accuracy Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Actor Loss
    ax = axes[1, 0]
    if train_log["actor_losses"]:
        ax.plot(train_log["actor_losses"], color='purple', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Actor Loss')
    ax.set_title('(d) Actor Loss')
    ax.grid(True, alpha=0.3)

    # Critic Loss
    ax = axes[1, 1]
    if train_log["critic_losses"]:
        ax.plot(train_log["critic_losses"], color='orange', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Critic Loss')
    ax.set_title('(e) Critic Loss')
    ax.grid(True, alpha=0.3)

    # Compression Ratio变化
    ax = axes[1, 2]
    raw = train_log["episode_compression_ratios"]
    smoothed = smooth_curve(raw)
    ax.plot(episodes, raw, alpha=0.3, color='brown', label='Raw')
    ax.plot(episodes, smoothed, color='brown', linewidth=2, label='Smoothed')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Compression Ratio')
    ax.set_title('(f) Compression Ratio Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'convergence_{model_type}.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")
    return path


# =============================================================================
# 实验2: 方法对比 (延迟和准确率)
# =============================================================================
def compute_baseline_results(model, layer_profiles, bandwidth_mbps, base_accuracy=0.85):
    """计算各baseline方法的结果"""
    num_layers = len(layer_profiles)

    # 1. 纯本地推理
    local_lat, _, _, _ = compute_partition_latency(layer_profiles, num_layers, 1.0, bandwidth_mbps)
    local_acc = base_accuracy

    # 2. 纯云端推理
    cloud_lat, _, _, _ = compute_partition_latency(layer_profiles, 0, 1.0, bandwidth_mbps)
    cloud_acc = base_accuracy

    # 3. Neurosurgeon (最优分区点, 不压缩)
    best_ns_lat = float("inf")
    best_ns_pp = 0
    for pp in range(num_layers + 1):
        lat, _, _, _ = compute_partition_latency(layer_profiles, pp, 1.0, bandwidth_mbps)
        if lat < best_ns_lat:
            best_ns_lat = lat
            best_ns_pp = pp
    ns_acc = base_accuracy  # Neurosurgeon不压缩, 准确率不变

    return {
        "Local Only": {"latency": local_lat, "accuracy": local_acc, "pp": num_layers, "cr": 1.0},
        "Cloud Only": {"latency": cloud_lat, "accuracy": cloud_acc, "pp": 0, "cr": 1.0},
        "Neurosurgeon": {"latency": best_ns_lat, "accuracy": ns_acc, "pp": best_ns_pp, "cr": 1.0},
    }


def evaluate_arl_comp(agent, env, num_eval_episodes=20):
    """评估ARL-Comp智能体"""
    latencies = []
    accuracies = []
    pps = []
    crs = []

    for _ in range(num_eval_episodes):
        state = env.reset()
        ep_lat = []
        ep_acc = []
        ep_pp = []
        ep_cr = []

        for _ in range(env.max_steps):
            d_action, c_action, _, _, _ = agent.select_action(state, deterministic=True)
            state, _, done, info = env.step(d_action, c_action)
            ep_lat.append(info["total_latency_ms"])
            ep_acc.append(info["accuracy"])
            ep_pp.append(info["partition_point"])
            ep_cr.append(info["compression_ratio"])
            if done:
                break

        latencies.append(np.mean(ep_lat))
        accuracies.append(np.mean(ep_acc))
        pps.extend(ep_pp)
        crs.extend(ep_cr)

    return {
        "latency": np.mean(latencies),
        "latency_std": np.std(latencies),
        "accuracy": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "avg_pp": np.mean(pps),
        "avg_cr": np.mean(crs),
    }


def plot_method_comparison(results_dict, model_type, save_dir=FIGURES_DIR):
    """绘制方法对比柱状图"""
    methods = list(results_dict.keys())
    latencies = [results_dict[m]["latency"] for m in methods]
    accuracies = [results_dict[m]["accuracy"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Method Comparison on {model_type}', fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    x = np.arange(len(methods))
    width = 0.5

    # 延迟对比
    bars1 = ax1.bar(x, latencies, width, color=colors[:len(methods)], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Inference Latency (ms)', fontsize=12)
    ax1.set_title('(a) End-to-End Latency Comparison', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, latencies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 准确率对比
    bars2 = ax2.bar(x, [a * 100 for a in accuracies], width, color=colors[:len(methods)],
                    edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Inference Accuracy Comparison', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(a * 100 for a in accuracies) - 5, 100])
    for bar, val in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(save_dir, f'method_comparison_{model_type}.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")
    return path


# =============================================================================
# 实验3: 不同带宽下的性能对比
# =============================================================================
def plot_bandwidth_comparison(model, layer_profiles, agent, env,
                              bandwidth_list, model_type, base_accuracy=0.85,
                              save_dir=FIGURES_DIR):
    """绘制不同带宽下各方法的性能对比"""
    results = {
        "bandwidth": bandwidth_list,
        "Local Only": {"latency": [], "accuracy": []},
        "Cloud Only": {"latency": [], "accuracy": []},
        "Neurosurgeon": {"latency": [], "accuracy": []},
        "ARL-Comp": {"latency": [], "accuracy": []},
    }

    num_layers = len(layer_profiles)

    for bw in bandwidth_list:
        # Baselines
        baselines = compute_baseline_results(model, layer_profiles, bw, base_accuracy)
        for method in ["Local Only", "Cloud Only", "Neurosurgeon"]:
            results[method]["latency"].append(baselines[method]["latency"])
            results[method]["accuracy"].append(baselines[method]["accuracy"])

        # ARL-Comp评估
        env.bandwidth_range = (bw * 0.9, bw * 1.1)
        env.bandwidth_variation = False
        env.current_bandwidth = bw
        arl_result = evaluate_arl_comp(agent, env, num_eval_episodes=10)
        results["ARL-Comp"]["latency"].append(arl_result["latency"])
        results["ARL-Comp"]["accuracy"].append(arl_result["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Performance under Different Bandwidths ({model_type})', fontsize=14, fontweight='bold')

    markers = ['o', 's', '^', 'D']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    methods = ["Local Only", "Cloud Only", "Neurosurgeon", "ARL-Comp"]

    for i, method in enumerate(methods):
        ax1.plot(bandwidth_list, results[method]["latency"],
                marker=markers[i], color=colors[i], linewidth=2,
                markersize=8, label=method)

    ax1.set_xlabel('Bandwidth (Mbps)', fontsize=12)
    ax1.set_ylabel('Inference Latency (ms)', fontsize=12)
    ax1.set_title('(a) Latency vs Bandwidth', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for i, method in enumerate(methods):
        ax2.plot(bandwidth_list, [a * 100 for a in results[method]["accuracy"]],
                marker=markers[i], color=colors[i], linewidth=2,
                markersize=8, label=method)

    ax2.set_xlabel('Bandwidth (Mbps)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Accuracy vs Bandwidth', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'bandwidth_comparison_{model_type}.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")
    return results


# =============================================================================
# 实验4: 压缩率分析
# =============================================================================
def plot_compression_analysis(model, layer_profiles, model_type,
                              bandwidth_mbps=10, base_accuracy=0.85,
                              save_dir=FIGURES_DIR):
    """分析不同压缩率下的延迟和准确率"""
    compression_ratios = np.arange(0.1, 1.05, 0.05)
    num_layers = len(layer_profiles)

    # 选取几个代表性的分区点
    candidate_points = [1, num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    candidate_points = [p for p in candidate_points if p < num_layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Compression Ratio Analysis ({model_type})', fontsize=14, fontweight='bold')

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(candidate_points)))

    for idx, pp in enumerate(candidate_points):
        latencies = []
        accuracies = []
        for cr in compression_ratios:
            lat, _, _, _ = compute_partition_latency(layer_profiles, pp, cr, bandwidth_mbps)
            acc = estimate_accuracy_after_compression(model, layer_profiles, pp, cr, base_accuracy)
            latencies.append(lat)
            accuracies.append(acc)

        ax1.plot(compression_ratios, latencies, color=colors[idx], linewidth=2,
                marker='o', markersize=4, label=f'Partition Point {pp}')
        ax2.plot(compression_ratios, [a*100 for a in accuracies], color=colors[idx],
                linewidth=2, marker='o', markersize=4, label=f'Partition Point {pp}')

    ax1.set_xlabel('Compression Ratio', fontsize=12)
    ax1.set_ylabel('Inference Latency (ms)', fontsize=12)
    ax1.set_title('(a) Latency vs Compression Ratio', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Compression Ratio', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Accuracy vs Compression Ratio', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'compression_analysis_{model_type}.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 实验5: 分区点筛选效果分析
# =============================================================================
def plot_filter_analysis(model, layer_profiles, model_type,
                         bandwidth_mbps=10, base_accuracy=0.85,
                         save_dir=FIGURES_DIR):
    """分析分区点筛选算法效果"""
    num_layers = len(layer_profiles)

    # 所有分区点的延迟和准确率
    all_latencies = []
    all_accuracies = []
    for pp in range(num_layers + 1):
        lat, _, _, _ = compute_partition_latency(layer_profiles, pp, 0.8, bandwidth_mbps)
        acc = estimate_accuracy_after_compression(model, layer_profiles, pp, 0.8, base_accuracy)
        all_latencies.append(lat)
        all_accuracies.append(acc)

    # 执行筛选
    feasible_points, filter_info = filter_partition_points(
        model, layer_profiles, bandwidth_mbps,
        base_accuracy=base_accuracy, verbose=False
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Partition Point Filtering Analysis ({model_type})', fontsize=14, fontweight='bold')

    # 延迟分析
    x = list(range(num_layers + 1))
    colors = ['green' if pp in feasible_points else 'lightcoral' for pp in x]

    ax1.bar(x, all_latencies, color=colors, edgecolor='black', linewidth=0.3)
    ax1.axhline(y=np.median(all_latencies), color='blue', linestyle='--',
                linewidth=1.5, label=f'Median={np.median(all_latencies):.1f}ms')
    ax1.set_xlabel('Partition Point', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('(a) Per-Partition-Point Latency\n(Green=Feasible, Red=Filtered)', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 准确率分析
    ax2.bar(x, [a*100 for a in all_accuracies], color=colors, edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('Partition Point', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Per-Partition-Point Accuracy\n(Green=Feasible, Red=Filtered)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(save_dir, f'filter_analysis_{model_type}.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")

    return feasible_points


# =============================================================================
# 实验6: 消融实验
# =============================================================================
def run_ablation_study(model_type, layer_profiles, model, base_accuracy=0.85,
                       bandwidth_mbps=10, save_dir=FIGURES_DIR):
    """消融实验: 有/无筛选, 有/无压缩"""

    num_layers = len(layer_profiles)

    # A: ARL-Comp (有筛选 + 有压缩) - 完整版
    # B: 无筛选 + 有压缩
    # C: 有筛选 + 无压缩 (固定压缩率=1.0)
    # D: 无筛选 + 无压缩 (等价于Neurosurgeon+RL)

    variants = {
        "ARL-Comp\n(Filter+Compress)": {"filter": True, "compress": True},
        "No Filter\n(Compress Only)": {"filter": False, "compress": True},
        "No Compress\n(Filter Only)": {"filter": True, "compress": False},
        "No Filter\nNo Compress": {"filter": False, "compress": False},
    }

    results = {}
    for name, config in variants.items():
        if config["filter"]:
            fps, _ = filter_partition_points(model, layer_profiles, bandwidth_mbps,
                                             base_accuracy=base_accuracy, verbose=False)
        else:
            fps = list(range(num_layers + 1))

        # 模拟: 遍历所有可行分区点, 找最优
        best_reward = -float("inf")
        best_lat = 0
        best_acc = 0
        best_pp = 0
        best_cr = 1.0

        cr_range = np.arange(0.1, 1.05, 0.1) if config["compress"] else [1.0]

        for pp in fps:
            for cr in cr_range:
                lat, _, _, _ = compute_partition_latency(layer_profiles, pp, cr, bandwidth_mbps)
                acc = estimate_accuracy_after_compression(model, layer_profiles, pp, cr, base_accuracy)
                reward = -0.5 * (lat / 100) + 0.5 * acc
                if reward > best_reward:
                    best_reward = reward
                    best_lat = lat
                    best_acc = acc
                    best_pp = pp
                    best_cr = cr

        results[name] = {
            "latency": best_lat,
            "accuracy": best_acc,
            "pp": best_pp,
            "cr": best_cr,
            "reward": best_reward,
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Ablation Study ({model_type})', fontsize=14, fontweight='bold')

    names = list(results.keys())
    lats = [results[n]["latency"] for n in names]
    accs = [results[n]["accuracy"] for n in names]
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3']

    x = np.arange(len(names))
    bars1 = ax1.bar(x, lats, 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('(a) Latency Ablation', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, lats):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, [a*100 for a in accs], 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Accuracy Ablation', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(a*100 for a in accs) - 5, 100])
    for bar, val in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f'ablation_{model_type}.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")

    return results


# =============================================================================
# 实验7: 多模型对比表格
# =============================================================================
def generate_comparison_table(all_model_results, save_dir=TABLES_DIR):
    """生成各模型上的对比结果表格"""
    rows = []
    for model_type, methods_results in all_model_results.items():
        for method, res in methods_results.items():
            rows.append({
                "Model": model_type,
                "Method": method,
                "Latency (ms)": round(res["latency"], 2),
                "Accuracy (%)": round(res["accuracy"] * 100, 2),
                "Partition Point": res.get("pp", "-"),
                "Compression Ratio": res.get("cr", "-"),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "method_comparison_all_models.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [Saved] {csv_path}")

    # 生成LaTeX表格
    latex_path = os.path.join(save_dir, "method_comparison_all_models.tex")
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Comparison of Different Methods on Various DNN Models}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{l|l|c|c|c|c}\n")
        f.write("\\hline\n")
        f.write("Model & Method & Latency (ms) & Accuracy (\\%) & Partition Pt & Comp. Ratio \\\\\n")
        f.write("\\hline\\hline\n")
        for _, row in df.iterrows():
            cr = f"{row['Compression Ratio']:.2f}" if isinstance(row['Compression Ratio'], float) else str(row['Compression Ratio'])
            f.write(f"{row['Model']} & {row['Method']} & {row['Latency (ms)']} & "
                    f"{row['Accuracy (%)']} & {row['Partition Point']} & {cr} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  [Saved] {latex_path}")

    return df


# =============================================================================
# 实验8: 多模型延迟/准确率对比折线图
# =============================================================================
def plot_multi_model_comparison(all_model_results, save_dir=FIGURES_DIR):
    """绘制多模型对比图"""
    models = list(all_model_results.keys())
    methods = ["Local Only", "Cloud Only", "Neurosurgeon", "ARL-Comp"]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    hatches = ['/', '\\', '|', '-']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Multi-Model Performance Comparison', fontsize=14, fontweight='bold')

    x = np.arange(len(models))
    width = 0.18

    for i, method in enumerate(methods):
        lats = []
        accs = []
        for m in models:
            if method in all_model_results[m]:
                lats.append(all_model_results[m][method]["latency"])
                accs.append(all_model_results[m][method]["accuracy"] * 100)
            else:
                lats.append(0)
                accs.append(0)

        ax1.bar(x + i * width, lats, width, label=method, color=colors[i],
                edgecolor='black', linewidth=0.5, hatch=hatches[i])
        ax2.bar(x + i * width, accs, width, label=method, color=colors[i],
                edgecolor='black', linewidth=0.5, hatch=hatches[i])

    ax1.set_xlabel('DNN Model', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('(a) Latency Comparison', fontsize=12)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.set_xlabel('DNN Model', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('(b) Accuracy Comparison', fontsize=12)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(save_dir, 'multi_model_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 实验9: 分区点决策分布热力图
# =============================================================================
def plot_decision_heatmap(train_log, model_type, num_layers, save_dir=FIGURES_DIR):
    """绘制分区点和压缩率的决策分布热力图"""
    pps = train_log["episode_partition_points"]
    crs = train_log["episode_compression_ratios"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Decision Distribution ({model_type})', fontsize=14, fontweight='bold')

    # 分区点分布
    episodes = list(range(len(pps)))
    ax1.scatter(episodes, pps, c=crs, cmap='coolwarm', alpha=0.6, s=15)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Partition Point', fontsize=12)
    ax1.set_title('(a) Partition Point Selection over Training', fontsize=11)
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Compression Ratio')
    ax1.grid(True, alpha=0.3)

    # 压缩率分布直方图
    ax2.hist(crs, bins=20, color='#4CAF50', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax2.set_xlabel('Compression Ratio', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('(b) Compression Ratio Distribution', fontsize=11)
    ax2.axvline(x=np.mean(crs), color='red', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(crs):.3f}')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'decision_distribution_{model_type}.png')
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")


# =============================================================================
# 主实验流程
# =============================================================================
def run_all_experiments(model_types=None, num_episodes=300, bandwidth_mbps=10,
                        base_accuracy=0.85, device="cpu"):
    """运行所有实验"""
    if model_types is None:
        model_types = ["alex_net", "vgg_net"]

    ensure_dirs()

    all_model_results = {}
    bandwidth_list = [1, 5, 10, 20, 30, 50]

    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"  Running experiments for: {model_type}")
        print(f"{'='*80}")

        # 训练ARL-Comp
        print(f"\n[1/7] Training ARL-Comp agent...")
        agent, train_log, filter_info = train_arl_comp(
            model_type=model_type,
            num_episodes=num_episodes,
            max_steps_per_episode=50,
            bandwidth_range=(1, 50),
            base_accuracy=base_accuracy,
            device=device,
            save_dir=RESULTS_DIR,
        )

        # 获取模型和profiles
        model = get_dnn_model(model_type)
        layer_profiles = profile_model(model, device=device)
        num_layers = len(layer_profiles)

        # 创建评估环境
        feasible_points = filter_info["feasible_points"] if filter_info else None
        eval_env = ARLCompEnv(
            model=model, layer_profiles=layer_profiles,
            feasible_points=feasible_points,
            bandwidth_range=(bandwidth_mbps * 0.9, bandwidth_mbps * 1.1),
            base_accuracy=base_accuracy,
            max_steps=50,
        )

        # 收敛曲线
        print(f"\n[2/7] Plotting convergence curves...")
        plot_convergence_curves(train_log, model_type)

        # 方法对比
        print(f"\n[3/7] Computing baseline comparisons...")
        baselines = compute_baseline_results(model, layer_profiles, bandwidth_mbps, base_accuracy)
        arl_result = evaluate_arl_comp(agent, eval_env)
        all_results = {**baselines}
        all_results["ARL-Comp"] = {
            "latency": arl_result["latency"],
            "accuracy": arl_result["accuracy"],
            "pp": int(arl_result["avg_pp"]),
            "cr": round(arl_result["avg_cr"], 3),
        }
        plot_method_comparison(all_results, model_type)
        all_model_results[model_type] = all_results

        # 带宽对比
        print(f"\n[4/7] Bandwidth comparison experiment...")
        plot_bandwidth_comparison(model, layer_profiles, agent, eval_env,
                                 bandwidth_list, model_type, base_accuracy)

        # 压缩率分析
        print(f"\n[5/7] Compression ratio analysis...")
        plot_compression_analysis(model, layer_profiles, model_type,
                                  bandwidth_mbps, base_accuracy)

        # 分区点筛选分析
        print(f"\n[6/7] Filter analysis...")
        plot_filter_analysis(model, layer_profiles, model_type,
                             bandwidth_mbps, base_accuracy)

        # 消融实验
        print(f"\n[7/7] Ablation study...")
        run_ablation_study(model_type, layer_profiles, model, base_accuracy,
                          bandwidth_mbps)

        # 决策分布图
        plot_decision_heatmap(train_log, model_type, num_layers)

    # 多模型对比
    if len(model_types) >= 2:
        print(f"\n{'='*80}")
        print(f"  Generating cross-model comparisons")
        print(f"{'='*80}")
        plot_multi_model_comparison(all_model_results)
        generate_comparison_table(all_model_results)

    print(f"\n{'='*80}")
    print(f"  All experiments completed!")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Tables saved to: {TABLES_DIR}")
    print(f"{'='*80}")

    return all_model_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARL-Comp Experiments")
    parser.add_argument("--models", nargs="+", default=["alex_net", "vgg_net"],
                        choices=["alex_net", "vgg_net", "le_net", "mobile_net"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--bandwidth", type=float, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_all_experiments(
        model_types=args.models,
        num_episodes=args.episodes,
        bandwidth_mbps=args.bandwidth,
        device=args.device,
    )

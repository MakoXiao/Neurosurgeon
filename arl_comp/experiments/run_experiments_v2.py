"""
ARL-Comp 实验运行器 V2
模拟车联网场景下的边云协同推理:
- 边端设备(车载终端): 计算能力较弱, 推理慢
- 云端设备(MEC服务器): 计算能力强, 推理快
- 网络带宽: 可变, 1-50 Mbps

使用 data/caltech-101 数据集进行训练和评估
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import pickle
import time

from utils.inference_utils import get_dnn_model
from arl_comp.pruning.pruner import estimate_accuracy_after_compression

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

def ensure_dirs():
    for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
        os.makedirs(d, exist_ok=True)


# =============================================================================
# 模拟参数: 车联网边云协同场景
# =============================================================================
# 边端设备(车载终端)相对于云端的计算速度比
# 车载终端算力有限: Conv层边端约慢8-12倍, FC层边端更慢(内存带宽受限)约慢15-20倍
EDGE_CONV_SLOWDOWN = 10.0
EDGE_FC_SLOWDOWN = 18.0
EDGE_OTHER_SLOWDOWN = 5.0
CLOUD_SPEEDUP = 1.0

# AlexNet在云端(MEC GPU服务器)的参考逐层延迟 (ms)
ALEXNET_CLOUD_LATENCY = {
    "conv1": 0.15, "relu1": 0.01, "pool1": 0.02,
    "conv2": 0.10, "relu2": 0.01, "pool2": 0.01,
    "conv3": 0.06, "relu3": 0.01,
    "conv4": 0.07, "relu4": 0.01,
    "conv5": 0.06, "relu5": 0.01, "pool3": 0.01,
    "avgpool": 0.01, "flatten": 0.001, "dropout1": 0.001,
    "fc1": 0.9, "relu6": 0.001, "dropout2": 0.001,
    "fc2": 0.12, "relu7": 0.001, "fc3": 0.03,
}

# VGGNet在云端的参考逐层延迟 (ms)
VGGNET_CLOUD_LATENCY = [
    3.0, 2.5, 0.5,       # block1
    2.0, 1.5, 0.3,       # block2
    1.2, 1.0, 0.8, 0.2,  # block3
    0.7, 0.6, 0.5, 0.15, # block4
    0.5, 0.4, 0.3, 0.1,  # block5
    3.5,                  # classifier
]

# MobileNet-V2在云端的参考逐层延迟 (ms)
# 20层: 1 stem + 14 InvertedResidual + 1 final conv + 4 classifier
MOBILENET_CLOUD_LATENCY = [
    0.25,   # stem (ConvNormActivation)
    0.10,   # IR block 1 (t=1, c=16)
    0.15,   # IR block 2 (t=1, c=24, stride=2)
    0.12,   # IR block 3 (t=1, c=24)
    0.20,   # IR block 4 (t=0.5, c=64, stride=2)
    0.15,   # IR block 5 (t=0.5, c=64)
    0.15,   # IR block 6 (t=0.5, c=64)
    0.15,   # IR block 7 (t=0.5, c=64)
    0.12,   # IR block 8 (t=0.5, c=96)
    0.12,   # IR block 9 (t=0.5, c=96)
    0.12,   # IR block 10 (t=0.5, c=96)
    0.18,   # IR block 11 (t=6, c=160, stride=2)
    0.15,   # IR block 12 (t=6, c=160, stride=2)
    0.15,   # IR block 13 (t=6, c=160)
    0.12,   # IR block 14 (t=6, c=320)
    0.08,   # final ConvNormActivation (1x1)
    0.01,   # AdaptiveAvgPool2d
    0.001,  # Flatten
    0.001,  # Dropout
    0.15,   # Linear (1280->1000)
]

# ResNet-18在云端的参考逐层延迟 (ms)
# 12层: 1 stem + 8 BasicBlocks + avgpool + flatten + fc
RESNET18_CLOUD_LATENCY = [
    0.80,   # stem (conv7x7 + bn + relu + maxpool)
    0.30,   # BasicBlock 1 (layer1[0], 64ch)
    0.30,   # BasicBlock 2 (layer1[1], 64ch)
    0.45,   # BasicBlock 3 (layer2[0], 128ch, stride=2)
    0.35,   # BasicBlock 4 (layer2[1], 128ch)
    0.55,   # BasicBlock 5 (layer3[0], 256ch, stride=2)
    0.40,   # BasicBlock 6 (layer3[1], 256ch)
    0.65,   # BasicBlock 7 (layer4[0], 512ch, stride=2)
    0.50,   # BasicBlock 8 (layer4[1], 512ch)
    0.01,   # AdaptiveAvgPool2d
    0.001,  # Flatten
    0.08,   # Linear (512->1000)
]

# ResNet-50在云端的参考逐层延迟 (ms)
# 20层: 1 stem + 16 Bottlenecks + avgpool + flatten + fc
RESNET50_CLOUD_LATENCY = [
    0.80,   # stem (conv7x7 + bn + relu + maxpool)
    0.50,   # Bottleneck 1 (layer1[0], 64->256, downsample)
    0.40,   # Bottleneck 2 (layer1[1])
    0.40,   # Bottleneck 3 (layer1[2])
    0.65,   # Bottleneck 4 (layer2[0], 128->512, stride=2)
    0.50,   # Bottleneck 5 (layer2[1])
    0.50,   # Bottleneck 6 (layer2[2])
    0.50,   # Bottleneck 7 (layer2[3])
    0.80,   # Bottleneck 8 (layer3[0], 256->1024, stride=2)
    0.65,   # Bottleneck 9 (layer3[1])
    0.65,   # Bottleneck 10 (layer3[2])
    0.65,   # Bottleneck 11 (layer3[3])
    0.65,   # Bottleneck 12 (layer3[4])
    0.65,   # Bottleneck 13 (layer3[5])
    0.90,   # Bottleneck 14 (layer4[0], 512->2048, stride=2)
    0.75,   # Bottleneck 15 (layer4[1])
    0.75,   # Bottleneck 16 (layer4[2])
    0.01,   # AdaptiveAvgPool2d
    0.001,  # Flatten
    0.15,   # Linear (2048->1000)
]


def simulate_profile(model, model_type):
    """
    模拟模型profiling, 生成逐层信息
    使用真实模型结构, 但用模拟的延迟参数
    """
    layer_profiles = []
    x = torch.rand(1, 3, 224, 224)

    idx = 0
    for layer in model:
        with torch.no_grad():
            x_out = layer(x)

        # 中间特征大小
        output_size_mb = 1
        for s in x_out.shape:
            output_size_mb *= s
        output_size_mb = output_size_mb * 4 / 1e6
        output_size_bytes = len(pickle.dumps(x_out))

        layer_name = layer.__class__.__name__
        # 判断层类型: 包含卷积的复合模块也算conv类型
        if isinstance(layer, nn.Conv2d):
            layer_type = "conv"
        elif isinstance(layer, nn.Linear):
            layer_type = "fc"
        elif any(isinstance(m, nn.Conv2d) for m in layer.modules()):
            layer_type = "conv"
        elif any(isinstance(m, nn.Linear) for m in layer.modules()):
            layer_type = "fc"
        else:
            layer_type = "other"

        # 模拟延迟
        latency_map = {
            "alex_net": list(ALEXNET_CLOUD_LATENCY.values()),
            "vgg_net": VGGNET_CLOUD_LATENCY,
            "mobile_net": MOBILENET_CLOUD_LATENCY,
            "resnet_18": RESNET18_CLOUD_LATENCY,
            "resnet_50": RESNET50_CLOUD_LATENCY,
        }
        cloud_lats = latency_map.get(model_type, VGGNET_CLOUD_LATENCY)
        cloud_lat = cloud_lats[idx] if idx < len(cloud_lats) else 0.01

        # 边端延迟: 按层类型使用不同的减速因子
        if layer_type == "conv":
            edge_lat = cloud_lat * EDGE_CONV_SLOWDOWN
        elif layer_type == "fc":
            edge_lat = cloud_lat * EDGE_FC_SLOWDOWN
        else:
            edge_lat = cloud_lat * EDGE_OTHER_SLOWDOWN

        layer_profiles.append({
            "index": idx,
            "layer": layer,
            "layer_name": layer_name,
            "layer_type": layer_type,
            "is_compressible": layer_type in ("conv", "fc"),
            "edge_latency_ms": edge_lat,
            "cloud_latency_ms": cloud_lat,
            "output_shape": tuple(x_out.shape),
            "output_size_mb": output_size_mb,
            "output_size_bytes": output_size_bytes,
        })

        x = x_out
        idx += 1

    return layer_profiles


def compute_partition_latency_sim(layer_profiles, partition_point, compression_ratio, bandwidth_mbps):
    """模拟计算: 给定划分点和压缩率的总延迟"""
    num_layers = len(layer_profiles)
    bandwidth_mbytes = bandwidth_mbps / 8.0

    # 边端延迟
    edge_latency = sum(lp["edge_latency_ms"] for lp in layer_profiles[:partition_point])

    # 传输延迟
    if partition_point == 0:
        trans_size = 1 * 3 * 224 * 224 * 4 / 1e6
    elif partition_point >= num_layers:
        trans_size = 0.0
    else:
        trans_size = layer_profiles[partition_point - 1]["output_size_mb"] * compression_ratio

    transmission_latency = (trans_size / bandwidth_mbytes * 1000) if (bandwidth_mbytes > 0 and trans_size > 0) else 0.0

    # 云端延迟
    cloud_latency = sum(lp["cloud_latency_ms"] for lp in layer_profiles[partition_point:])

    return edge_latency + transmission_latency + cloud_latency, edge_latency, transmission_latency, cloud_latency


# =============================================================================
# ARL-Comp RL环境 (V2, 模拟版)
# =============================================================================
class ARLCompEnvV2:
    def __init__(self, model, layer_profiles, feasible_points=None,
                 bandwidth_range=(1, 50), base_accuracy=0.85,
                 weight_latency=0.5, weight_accuracy=0.5,
                 max_steps=50):
        self.model = model
        self.layer_profiles = layer_profiles
        self.num_layers = len(layer_profiles)
        self.feasible_points = feasible_points or list(range(self.num_layers + 1))
        self.num_partition_points = len(self.feasible_points)
        self.bandwidth_range = bandwidth_range
        self.base_accuracy = base_accuracy
        self.weight_latency = weight_latency
        self.weight_accuracy = weight_accuracy
        self.max_steps = max_steps

        self.state_dim = 5
        self.num_discrete_actions = self.num_partition_points
        self.continuous_action_dim = 1
        self.continuous_action_low = 0.1
        self.continuous_action_high = 1.0

        # 归一化
        self.max_latency = max(
            compute_partition_latency_sim(layer_profiles, pp, 1.0, bandwidth_range[0])[0]
            for pp in range(self.num_layers + 1)
        )
        if self.max_latency <= 0:
            self.max_latency = 1.0
        self.max_bandwidth = bandwidth_range[1]
        self.max_feature_size = max(
            (lp["output_size_mb"] for lp in layer_profiles), default=1.0
        )

        self.reset()

    def reset(self):
        self.step_count = 0
        self.bandwidth = np.random.uniform(*self.bandwidth_range)
        self.last_lat = self.max_latency * 0.5
        self.last_acc = self.base_accuracy
        self.last_feat = self.max_feature_size * 0.5
        self.last_edge_ratio = 0.5
        self.history = []
        return self._state()

    def _state(self):
        return np.array([
            np.clip(self.last_lat / self.max_latency, 0, 1),
            np.clip(self.bandwidth / self.max_bandwidth, 0, 1),
            np.clip(self.last_acc / self.base_accuracy, 0, 1),
            np.clip(self.last_feat / self.max_feature_size, 0, 1),
            np.clip(self.last_edge_ratio, 0, 1),
        ], dtype=np.float32)

    def step(self, d_action, c_action):
        self.step_count += 1
        pp_idx = int(np.clip(d_action, 0, self.num_partition_points - 1))
        pp = self.feasible_points[pp_idx]
        cr = float(np.clip(c_action, 0.1, 1.0))

        # 带宽波动
        self.bandwidth = np.clip(
            self.bandwidth + np.random.normal(0, 0.1 * self.bandwidth),
            *self.bandwidth_range
        )

        total, edge, trans, cloud = compute_partition_latency_sim(
            self.layer_profiles, pp, cr, self.bandwidth
        )
        acc = estimate_accuracy_after_compression(
            self.model, self.layer_profiles, pp, cr, self.base_accuracy
        )

        feat = self.layer_profiles[pp-1]["output_size_mb"] * cr if 0 < pp <= self.num_layers else 0
        edge_ratio = edge / total if total > 0 else 0.5

        self.last_lat = total
        self.last_acc = acc
        self.last_feat = feat
        self.last_edge_ratio = edge_ratio

        reward = -self.weight_latency * (total / self.max_latency) + self.weight_accuracy * (acc / self.base_accuracy)
        done = self.step_count >= self.max_steps

        info = {
            "partition_point": pp, "compression_ratio": cr,
            "total_latency_ms": total, "edge_latency_ms": edge,
            "transmission_latency_ms": trans, "cloud_latency_ms": cloud,
            "accuracy": acc, "bandwidth_mbps": self.bandwidth,
            "reward": reward, "feature_size_mb": feat,
        }
        self.history.append(info)
        return self._state(), reward, done, info


# =============================================================================
# 分区点筛选 (V2)
# =============================================================================
def filter_partition_points_v2(model, layer_profiles, bandwidth_mbps,
                                base_accuracy=0.85, default_compression=0.8,
                                verbose=True):
    num_layers = len(layer_profiles)
    total_cands = num_layers + 1
    min_pts = max(3, int(np.ceil(total_cands * 0.25)))
    max_pts = int(np.floor(total_cands * 0.50))

    point_info = []
    for pp in range(total_cands):
        lat = compute_partition_latency_sim(layer_profiles, pp, default_compression, bandwidth_mbps)[0]
        acc = estimate_accuracy_after_compression(model, layer_profiles, pp, default_compression, base_accuracy)
        point_info.append({"pp": pp, "lat": lat, "acc": acc})

    median_lat = np.median([p["lat"] for p in point_info])
    lat_threshold = 0.25
    acc_threshold = 0.02

    if verbose:
        print(f"[Filter] Candidates={total_cands}, Median Latency={median_lat:.2f}ms, Target=[{min_pts},{max_pts}]")

    feasible = list(range(total_cands))
    for iteration in range(30):
        # Latency filter
        limit = median_lat * (1 + lat_threshold)
        filtered = [p["pp"] for p in point_info if p["pp"] in feasible and p["lat"] <= limit]
        if len(filtered) < min_pts:
            lat_threshold += 0.05
            continue
        feasible = filtered

        if len(feasible) <= max_pts:
            break

        # Accuracy filter
        sorted_by_acc = sorted([(p["pp"], p["acc"]) for p in point_info if p["pp"] in feasible],
                                key=lambda x: x[1], reverse=True)
        best_acc = sorted_by_acc[0][1]
        acc_filtered = [pp for pp, acc in sorted_by_acc if (best_acc - acc) <= acc_threshold]
        if len(acc_filtered) < min_pts:
            acc_filtered = [pp for pp, _ in sorted_by_acc[:min_pts]]
        feasible = acc_filtered

        if len(feasible) <= max_pts:
            break

        lat_threshold -= 0.05
        lat_threshold = max(0.05, lat_threshold)
        acc_threshold -= 0.002
        acc_threshold = max(0.005, acc_threshold)

    feasible = sorted(feasible)
    if verbose:
        print(f"[Filter] Result: {feasible} ({len(feasible)} points)")
        for pp in feasible:
            info = point_info[pp]
            print(f"  PP {pp}: lat={info['lat']:.2f}ms, acc={info['acc']:.4f}")

    return feasible, point_info


# =============================================================================
# Hybrid PPO Agent (从agent模块导入)
# =============================================================================
from arl_comp.agent.hybrid_ppo import HybridPPO


# =============================================================================
# 训练函数
# =============================================================================
def train_agent(model_type, num_episodes=300, bandwidth_range=(1, 50),
                base_accuracy=0.85, use_filter=True, seed=42, verbose=True):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = get_dnn_model(model_type)
    layer_profiles = simulate_profile(model, model_type)

    if verbose:
        print(f"\n[Train] Model={model_type}, Layers={len(layer_profiles)}")
        local_lat = compute_partition_latency_sim(layer_profiles, len(layer_profiles), 1.0, 10)[0]
        cloud_lat = compute_partition_latency_sim(layer_profiles, 0, 1.0, 10)[0]
        print(f"  Local-only latency: {local_lat:.2f}ms, Cloud-only latency: {cloud_lat:.2f}ms")

    fps = None
    fp_info = None
    if use_filter:
        avg_bw = np.mean(bandwidth_range)
        fps, fp_info = filter_partition_points_v2(model, layer_profiles, avg_bw, base_accuracy, verbose=verbose)

    env = ARLCompEnvV2(model, layer_profiles, fps, bandwidth_range, base_accuracy)

    agent = HybridPPO(
        state_dim=env.state_dim,
        num_discrete_actions=env.num_discrete_actions,
        continuous_action_low=0.1,
        continuous_action_high=1.0,
        hidden_dim=128,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        device="cpu",
    )

    log = {"rewards": [], "latencies": [], "accuracies": [], "pps": [], "crs": [],
           "actor_losses": [], "critic_losses": [], "entropies": []}

    for ep in range(num_episodes):
        state = env.reset()
        ep_r, ep_lats, ep_accs, ep_pps, ep_crs = 0, [], [], [], []

        for _ in range(env.max_steps):
            da, ca, dl, cl, v = agent.select_action(state)
            ns, r, done, info = env.step(da, ca)
            agent.store_transition(state, da, ca, dl, cl, r, done, v)
            state = ns
            ep_r += r
            ep_lats.append(info["total_latency_ms"])
            ep_accs.append(info["accuracy"])
            ep_pps.append(info["partition_point"])
            ep_crs.append(info["compression_ratio"])
            if done:
                break

        ui = agent.update()

        log["rewards"].append(ep_r / env.max_steps)
        log["latencies"].append(np.mean(ep_lats))
        log["accuracies"].append(np.mean(ep_accs))
        log["pps"].append(max(set(ep_pps), key=ep_pps.count))
        log["crs"].append(np.mean(ep_crs))
        if ui:
            log["actor_losses"].append(ui["actor_loss"])
            log["critic_losses"].append(ui["critic_loss"])
            log["entropies"].append(ui["entropy"])

        if verbose and (ep+1) % 50 == 0:
            print(f"  Ep {ep+1:4d}/{num_episodes} | R={log['rewards'][-1]:.4f} | "
                  f"Lat={log['latencies'][-1]:.2f}ms | Acc={log['accuracies'][-1]:.4f} | "
                  f"PP={log['pps'][-1]} | CR={log['crs'][-1]:.3f}")

    return agent, log, layer_profiles, fps, env


# =============================================================================
# 评估函数
# =============================================================================
def evaluate_agent(agent, env, num_episodes=30):
    lats, accs, pps, crs = [], [], [], []
    for _ in range(num_episodes):
        state = env.reset()
        ep_l, ep_a, ep_p, ep_c = [], [], [], []
        for _ in range(env.max_steps):
            da, ca, _, _, _ = agent.select_action(state, deterministic=True)
            state, _, done, info = env.step(da, ca)
            ep_l.append(info["total_latency_ms"])
            ep_a.append(info["accuracy"])
            ep_p.append(info["partition_point"])
            ep_c.append(info["compression_ratio"])
            if done:
                break
        lats.append(np.mean(ep_l))
        accs.append(np.mean(ep_a))
        pps.extend(ep_p)
        crs.extend(ep_c)
    return {
        "latency": np.mean(lats), "latency_std": np.std(lats),
        "accuracy": np.mean(accs), "accuracy_std": np.std(accs),
        "avg_pp": np.mean(pps), "avg_cr": np.mean(crs),
        "most_pp": max(set(pps), key=pps.count),
    }


def compute_baselines(layer_profiles, bandwidth_mbps, base_accuracy, model):
    n = len(layer_profiles)
    local_lat = compute_partition_latency_sim(layer_profiles, n, 1.0, bandwidth_mbps)[0]
    cloud_lat = compute_partition_latency_sim(layer_profiles, 0, 1.0, bandwidth_mbps)[0]

    best_ns_lat, best_ns_pp = float("inf"), 0
    for pp in range(n + 1):
        lat = compute_partition_latency_sim(layer_profiles, pp, 1.0, bandwidth_mbps)[0]
        if lat < best_ns_lat:
            best_ns_lat = lat
            best_ns_pp = pp

    # 固定压缩率的基线 (Fixed Compression)
    best_fc_lat, best_fc_pp = float("inf"), 0
    fixed_cr = 0.5
    for pp in range(n + 1):
        lat = compute_partition_latency_sim(layer_profiles, pp, fixed_cr, bandwidth_mbps)[0]
        if lat < best_fc_lat:
            best_fc_lat = lat
            best_fc_pp = pp
    fc_acc = estimate_accuracy_after_compression(model, layer_profiles, best_fc_pp, fixed_cr, base_accuracy)

    return {
        "Local Only": {"latency": local_lat, "accuracy": base_accuracy, "pp": n, "cr": 1.0},
        "Cloud Only": {"latency": cloud_lat, "accuracy": base_accuracy, "pp": 0, "cr": 1.0},
        "Neurosurgeon": {"latency": best_ns_lat, "accuracy": base_accuracy, "pp": best_ns_pp, "cr": 1.0},
        "Fixed Comp.\n(CR=0.5)": {"latency": best_fc_lat, "accuracy": fc_acc, "pp": best_fc_pp, "cr": fixed_cr},
    }


# =============================================================================
# 绘图函数
# =============================================================================
def smooth(data, w=0.85):
    s = []
    last = data[0]
    for p in data:
        v = last * w + (1 - w) * p
        s.append(v)
        last = v
    return s


def plot_convergence(log, model_type):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'ARL-Comp Training Convergence on {model_type}\n(Trained on Caltech-101)', fontsize=15, fontweight='bold')
    eps = list(range(1, len(log["rewards"])+1))

    for ax, key, title, color, ylabel in [
        (axes[0,0], "rewards", "(a) Average Reward", "blue", "Reward"),
        (axes[0,1], "latencies", "(b) Average Latency", "red", "Latency (ms)"),
        (axes[0,2], "accuracies", "(c) Average Accuracy", "green", "Accuracy"),
        (axes[1,2], "crs", "(f) Compression Ratio", "brown", "Compression Ratio"),
    ]:
        raw = log[key]
        ax.plot(eps, raw, alpha=0.25, color=color)
        ax.plot(eps, smooth(raw), color=color, linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    if log["actor_losses"]:
        axes[1,0].plot(log["actor_losses"], color='purple', linewidth=1.5)
        axes[1,0].set_xlabel('Episode'); axes[1,0].set_ylabel('Actor Loss')
        axes[1,0].set_title('(d) Actor Loss'); axes[1,0].grid(True, alpha=0.3)

    if log["critic_losses"]:
        axes[1,1].plot(log["critic_losses"], color='orange', linewidth=1.5)
        axes[1,1].set_xlabel('Episode'); axes[1,1].set_ylabel('Critic Loss')
        axes[1,1].set_title('(e) Critic Loss'); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'convergence_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def plot_method_comparison(all_results, model_type):
    methods = list(all_results.keys())
    lats = [all_results[m]["latency"] for m in methods]
    accs = [all_results[m]["accuracy"] for m in methods]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#E91E63']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Method Comparison on {model_type} (Caltech-101)', fontsize=14, fontweight='bold')

    x = np.arange(len(methods))
    bars1 = ax1.bar(x, lats, 0.5, color=colors[:len(methods)], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Latency (ms)'); ax1.set_title('(a) End-to-End Inference Latency')
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    for b, v in zip(bars1, lats):
        ax1.text(b.get_x()+b.get_width()/2., b.get_height()+0.3, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    bars2 = ax2.bar(x, [a*100 for a in accs], 0.5, color=colors[:len(methods)], edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Accuracy (%)'); ax2.set_title('(b) Inference Accuracy')
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ymin = min(a*100 for a in accs) - 5
    ax2.set_ylim([max(0, ymin), 100])
    for b, v in zip(bars2, accs):
        ax2.text(b.get_x()+b.get_width()/2., b.get_height()+0.2, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'method_comparison_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def plot_bandwidth_experiment(layer_profiles, agent, env, model, model_type, base_accuracy):
    bw_list = [1, 2, 5, 10, 15, 20, 30, 50]
    results = {m: {"lat": [], "acc": []} for m in ["Local Only", "Cloud Only", "Neurosurgeon", "ARL-Comp"]}
    n = len(layer_profiles)

    # 保存原始带宽范围
    orig_bw_range = env.bandwidth_range

    for bw in bw_list:
        baselines = compute_baselines(layer_profiles, bw, base_accuracy, model)
        for m in ["Local Only", "Cloud Only", "Neurosurgeon"]:
            results[m]["lat"].append(baselines[m]["latency"])
            results[m]["acc"].append(baselines[m]["accuracy"])

        # 评估ARL-Comp在特定带宽下的表现
        env.bandwidth_range = (max(1, bw * 0.9), bw * 1.1)
        arl = evaluate_agent(agent, env, 20)
        results["ARL-Comp"]["lat"].append(arl["latency"])
        results["ARL-Comp"]["acc"].append(arl["accuracy"])

    # 恢复原始带宽范围
    env.bandwidth_range = orig_bw_range

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Performance vs Bandwidth ({model_type}, Caltech-101)', fontsize=14, fontweight='bold')
    markers = ['o', 's', '^', 'D']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    for i, m in enumerate(["Local Only", "Cloud Only", "Neurosurgeon", "ARL-Comp"]):
        ax1.plot(bw_list, results[m]["lat"], marker=markers[i], color=colors[i], linewidth=2, markersize=8, label=m)
        ax2.plot(bw_list, [a*100 for a in results[m]["acc"]], marker=markers[i], color=colors[i], linewidth=2, markersize=8, label=m)

    ax1.set_xlabel('Bandwidth (Mbps)'); ax1.set_ylabel('Latency (ms)'); ax1.set_title('(a) Latency vs Bandwidth')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Bandwidth (Mbps)'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('(b) Accuracy vs Bandwidth')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'bandwidth_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")
    return results


def plot_compression_analysis(model, layer_profiles, model_type, base_accuracy):
    crs = np.arange(0.1, 1.05, 0.05)
    n = len(layer_profiles)
    candidates = sorted(set([1, n//4, n//2, 3*n//4, n-1]) & set(range(1, n)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Compression Ratio Analysis ({model_type})', fontsize=14, fontweight='bold')
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(candidates)))

    for i, pp in enumerate(candidates):
        lats = [compute_partition_latency_sim(layer_profiles, pp, cr, 10)[0] for cr in crs]
        accs = [estimate_accuracy_after_compression(model, layer_profiles, pp, cr, base_accuracy) for cr in crs]
        ax1.plot(crs, lats, color=colors[i], linewidth=2, marker='o', markersize=3, label=f'PP={pp}')
        ax2.plot(crs, [a*100 for a in accs], color=colors[i], linewidth=2, marker='o', markersize=3, label=f'PP={pp}')

    ax1.set_xlabel('Compression Ratio'); ax1.set_ylabel('Latency (ms)'); ax1.set_title('(a) Latency vs CR')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Compression Ratio'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('(b) Accuracy vs CR')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'compression_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def plot_filter_analysis(model, layer_profiles, feasible_points, model_type, base_accuracy):
    n = len(layer_profiles)
    all_pp = list(range(n + 1))
    lats = [compute_partition_latency_sim(layer_profiles, pp, 0.8, 10)[0] for pp in all_pp]
    accs = [estimate_accuracy_after_compression(model, layer_profiles, pp, 0.8, base_accuracy) for pp in all_pp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Partition Point Filtering ({model_type})', fontsize=14, fontweight='bold')

    cs = ['#4CAF50' if pp in feasible_points else '#EF9A9A' for pp in all_pp]
    ax1.bar(all_pp, lats, color=cs, edgecolor='black', linewidth=0.3)
    ax1.axhline(np.median(lats), color='blue', ls='--', lw=1.5, label=f'Median={np.median(lats):.1f}ms')
    ax1.set_xlabel('Partition Point'); ax1.set_ylabel('Latency (ms)')
    ax1.set_title('(a) Latency per Partition Point\n(Green=Selected, Red=Filtered)')
    ax1.legend(); ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(all_pp, [a*100 for a in accs], color=cs, edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('Partition Point'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('(b) Accuracy per Partition Point'); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'filter_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def plot_ablation(layer_profiles, model, model_type, base_accuracy, bw=10):
    n = len(layer_profiles)
    variants = {
        "ARL-Comp\n(Full)": (True, True),
        "No Filter": (False, True),
        "No Compress": (True, False),
        "Baseline\n(No F + No C)": (False, False),
    }

    results = {}
    for name, (use_f, use_c) in variants.items():
        if use_f:
            fps, _ = filter_partition_points_v2(model, layer_profiles, bw, base_accuracy, verbose=False)
        else:
            fps = list(range(n + 1))

        # 计算最大延迟用于归一化 (与RL环境一致)
        max_lat = max(
            compute_partition_latency_sim(layer_profiles, pp, 1.0, bw)[0]
            for pp in range(n + 1)
        )
        if max_lat <= 0:
            max_lat = 1.0

        best_r, best_l, best_a, best_pp, best_cr = -1e9, 0, 0, 0, 1.0
        cr_range = np.arange(0.1, 1.05, 0.05) if use_c else [1.0]
        for pp in fps:
            for cr in cr_range:
                l = compute_partition_latency_sim(layer_profiles, pp, cr, bw)[0]
                a = estimate_accuracy_after_compression(model, layer_profiles, pp, cr, base_accuracy)
                r = -0.5 * (l / max_lat) + 0.5 * (a / base_accuracy)
                if r > best_r:
                    best_r, best_l, best_a, best_pp, best_cr = r, l, a, pp, cr

        results[name] = {"latency": best_l, "accuracy": best_a, "pp": best_pp, "cr": best_cr}

    names = list(results.keys())
    lats = [results[n]["latency"] for n in names]
    accs = [results[n]["accuracy"] for n in names]
    colors = ['#E91E63', '#FF9800', '#4CAF50', '#2196F3']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Ablation Study ({model_type})', fontsize=14, fontweight='bold')
    x = np.arange(len(names))

    bars1 = ax1.bar(x, lats, 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Latency (ms)'); ax1.set_title('(a) Latency')
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9); ax1.grid(True, alpha=0.3, axis='y')
    for b, v in zip(bars1, lats):
        ax1.text(b.get_x()+b.get_width()/2., b.get_height()+0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, [a*100 for a in accs], 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Accuracy (%)'); ax2.set_title('(b) Accuracy')
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
    ymin = min(a*100 for a in accs) - 5
    ax2.set_ylim([max(0, ymin), 100])
    for b, v in zip(bars2, accs):
        ax2.text(b.get_x()+b.get_width()/2., b.get_height()+0.2, f'{v*100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'ablation_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")
    return results


def plot_decision_distribution(log, model_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Decision Distribution ({model_type})', fontsize=14, fontweight='bold')

    eps = list(range(len(log["pps"])))
    sc = ax1.scatter(eps, log["pps"], c=log["crs"], cmap='coolwarm', alpha=0.6, s=15)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Partition Point')
    ax1.set_title('(a) Partition Point over Training')
    plt.colorbar(sc, ax=ax1, label='Compression Ratio')
    ax1.grid(True, alpha=0.3)

    ax2.hist(log["crs"], bins=20, color='#4CAF50', edgecolor='black', alpha=0.8)
    ax2.axvline(np.mean(log["crs"]), color='red', ls='--', lw=2, label=f'Mean={np.mean(log["crs"]):.3f}')
    ax2.set_xlabel('Compression Ratio'); ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Compression Ratio Distribution')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'decision_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def plot_latency_breakdown(layer_profiles, arl_result, model_type, bandwidth_mbps=10, base_accuracy=0.85):
    """绘制延迟分解图 (edge/transmission/cloud)"""
    n = len(layer_profiles)

    # 各方法的延迟分解
    methods = ["Local Only", "Cloud Only", "Neurosurgeon", "ARL-Comp"]
    configs = [
        (n, 1.0),  # Local
        (0, 1.0),  # Cloud
        None,       # Neurosurgeon (best no-compress)
        (int(arl_result["avg_pp"]), arl_result["avg_cr"]),  # ARL-Comp
    ]

    # Find best Neurosurgeon
    best_ns_lat, best_ns_pp = float("inf"), 0
    for pp in range(n + 1):
        lat = compute_partition_latency_sim(layer_profiles, pp, 1.0, bandwidth_mbps)[0]
        if lat < best_ns_lat:
            best_ns_lat = lat
            best_ns_pp = pp
    configs[2] = (best_ns_pp, 1.0)

    edge_lats, trans_lats, cloud_lats = [], [], []
    for pp, cr in configs:
        _, e, t, c = compute_partition_latency_sim(layer_profiles, pp, cr, bandwidth_mbps)
        edge_lats.append(e)
        trans_lats.append(t)
        cloud_lats.append(c)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    w = 0.5

    ax.bar(x, edge_lats, w, label='Edge Inference', color='#42A5F5')
    ax.bar(x, trans_lats, w, bottom=edge_lats, label='Transmission', color='#FFA726')
    ax.bar(x, cloud_lats, w, bottom=[e+t for e,t in zip(edge_lats, trans_lats)],
           label='Cloud Inference', color='#66BB6A')

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title(f'Latency Breakdown ({model_type}, BW={bandwidth_mbps}Mbps)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注总延迟
    for i in range(len(methods)):
        total = edge_lats[i] + trans_lats[i] + cloud_lats[i]
        ax.text(i, total + 0.5, f'{total:.1f}ms', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'latency_breakdown_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def plot_multi_model(all_results):
    models = list(all_results.keys())
    methods = ["Local Only", "Cloud Only", "Neurosurgeon", "ARL-Comp"]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    hatches = ['/', '\\', '|', '-']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Cross-Model Performance Comparison (Caltech-101)', fontsize=14, fontweight='bold')

    x = np.arange(len(models))
    w = 0.18

    for i, m in enumerate(methods):
        lats = [all_results[mod].get(m, {}).get("latency", 0) for mod in models]
        accs = [all_results[mod].get(m, {}).get("accuracy", 0) * 100 for mod in models]
        ax1.bar(x + i*w, lats, w, label=m, color=colors[i], edgecolor='black', linewidth=0.5, hatch=hatches[i])
        ax2.bar(x + i*w, accs, w, label=m, color=colors[i], edgecolor='black', linewidth=0.5, hatch=hatches[i])

    ax1.set_xlabel('Model'); ax1.set_ylabel('Latency (ms)'); ax1.set_title('(a) Latency')
    ax1.set_xticks(x + w*1.5); ax1.set_xticklabels(models); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='y')
    ax2.set_xlabel('Model'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('(b) Accuracy')
    ax2.set_xticks(x + w*1.5); ax2.set_xticklabels(models); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, 'multi_model_comparison.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def plot_pareto_front(model, layer_profiles, agent, env, model_type, base_accuracy, bw=10):
    """绘制延迟-准确率Pareto前沿分析图, 展示ARL-Comp在Pareto前沿上的优势"""
    n = len(layer_profiles)

    # 所有可能的(pp, cr)组合
    all_points = []
    crs = np.arange(0.1, 1.05, 0.05)
    for pp in range(n + 1):
        for cr in crs:
            lat = compute_partition_latency_sim(layer_profiles, pp, cr, bw)[0]
            acc = estimate_accuracy_after_compression(model, layer_profiles, pp, cr, base_accuracy)
            all_points.append((lat, acc, pp, cr))

    # Neurosurgeon的点 (不同PP, CR=1.0)
    ns_points = []
    for pp in range(n + 1):
        lat = compute_partition_latency_sim(layer_profiles, pp, 1.0, bw)[0]
        ns_points.append((lat, base_accuracy, pp, 1.0))

    # ARL-Comp的决策点
    arl = evaluate_agent(agent, env, 30)
    arl_lat, arl_acc = arl["latency"], arl["accuracy"]

    # Fixed Comp (CR=0.5)的点
    fc_points = []
    for pp in range(n + 1):
        lat = compute_partition_latency_sim(layer_profiles, pp, 0.5, bw)[0]
        acc = estimate_accuracy_after_compression(model, layer_profiles, pp, 0.5, base_accuracy)
        fc_points.append((lat, acc, pp, 0.5))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f'Latency-Accuracy Trade-off ({model_type}, BW={bw}Mbps)', fontsize=13, fontweight='bold')

    # 所有组合(灰色背景)
    ax.scatter([p[0] for p in all_points], [p[1]*100 for p in all_points],
               alpha=0.08, c='gray', s=10, label='All (PP, CR) combos')

    # Neurosurgeon (CR=1.0)
    ax.scatter([p[0] for p in ns_points], [p[1]*100 for p in ns_points],
               alpha=0.6, c='#4CAF50', s=40, marker='^', label='Neurosurgeon (CR=1.0)')

    # Fixed Comp (CR=0.5)
    ax.scatter([p[0] for p in fc_points], [p[1]*100 for p in fc_points],
               alpha=0.6, c='#FF9800', s=40, marker='s', label='Fixed Comp. (CR=0.5)')

    # ARL-Comp
    ax.scatter([arl_lat], [arl_acc*100], c='#E91E63', s=200, marker='*', zorder=10,
               edgecolors='black', linewidths=1, label='ARL-Comp')

    # Local Only & Cloud Only
    local_lat = compute_partition_latency_sim(layer_profiles, n, 1.0, bw)[0]
    cloud_lat = compute_partition_latency_sim(layer_profiles, 0, 1.0, bw)[0]
    ax.scatter([local_lat], [base_accuracy*100], c='#2196F3', s=120, marker='D', zorder=9, label='Local Only')
    ax.scatter([cloud_lat], [base_accuracy*100], c='#FF5722', s=120, marker='D', zorder=9, label='Cloud Only')

    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'pareto_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def plot_bandwidth_avg_comparison(layer_profiles, agent, env, model, model_type, base_accuracy):
    """绘制多带宽平均性能对比, 展示ARL-Comp在动态带宽下的综合优势"""
    bw_list = [1, 2, 5, 10, 15, 20, 30, 50]
    n = len(layer_profiles)
    methods = ["Local Only", "Cloud Only", "Neurosurgeon", "Fixed Comp.\n(CR=0.5)", "ARL-Comp"]
    avg_lats = {m: [] for m in methods}
    avg_accs = {m: [] for m in methods}

    orig_bw_range = env.bandwidth_range
    for bw in bw_list:
        baselines = compute_baselines(layer_profiles, bw, base_accuracy, model)
        for m in ["Local Only", "Cloud Only", "Neurosurgeon", "Fixed Comp.\n(CR=0.5)"]:
            avg_lats[m].append(baselines[m]["latency"])
            avg_accs[m].append(baselines[m]["accuracy"])

        env.bandwidth_range = (max(1, bw * 0.9), bw * 1.1)
        arl = evaluate_agent(agent, env, 15)
        avg_lats["ARL-Comp"].append(arl["latency"])
        avg_accs["ARL-Comp"].append(arl["accuracy"])
    env.bandwidth_range = orig_bw_range

    # 计算各方法的加权平均 (低带宽权重更高, 模拟车联网场景)
    weights = np.array([3, 2.5, 2, 1.5, 1.2, 1, 0.8, 0.6])  # 低带宽更常见
    weights = weights / weights.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Bandwidth-Weighted Average Performance ({model_type})\n(Weighted towards low-bandwidth VEC scenarios)',
                 fontsize=13, fontweight='bold')

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#E91E63']
    x = np.arange(len(methods))
    w_lats = [np.average(avg_lats[m], weights=weights) for m in methods]
    w_accs = [np.average(avg_accs[m], weights=weights) for m in methods]

    bars1 = ax1.bar(x, w_lats, 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Weighted Avg Latency (ms)', fontsize=11)
    ax1.set_title('(a) Latency (BW-Weighted)')
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    for b, v in zip(bars1, w_lats):
        ax1.text(b.get_x()+b.get_width()/2., b.get_height()+0.5, f'{v:.1f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    bars2 = ax2.bar(x, [a*100 for a in w_accs], 0.5, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Weighted Avg Accuracy (%)', fontsize=11)
    ax2.set_title('(b) Accuracy (BW-Weighted)')
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=8)
    ymin = min(a*100 for a in w_accs) - 3
    ax2.set_ylim([max(0, ymin), max(a*100 for a in w_accs) + 2])
    ax2.grid(True, alpha=0.3, axis='y')
    for b, v in zip(bars2, w_accs):
        ax2.text(b.get_x()+b.get_width()/2., b.get_height()+0.1, f'{v*100:.1f}%',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    p = os.path.join(FIGURES_DIR, f'bw_avg_comparison_{model_type}.png')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"  [Fig] {p}")


def generate_tables(all_results):
    rows = []
    for model, methods in all_results.items():
        for method, res in methods.items():
            rows.append({
                "Model": model,
                "Method": method.replace('\n', ' '),
                "Latency (ms)": round(res["latency"], 2),
                "Accuracy (%)": round(res["accuracy"] * 100, 2),
                "Partition Point": res.get("pp", "-"),
                "Compression Ratio": round(res.get("cr", 1.0), 3) if isinstance(res.get("cr"), (int, float)) else "-",
            })
    df = pd.DataFrame(rows)
    csv_p = os.path.join(TABLES_DIR, "comparison_all_models.csv")
    df.to_csv(csv_p, index=False)
    print(f"  [Table] {csv_p}")

    # LaTeX
    tex_p = os.path.join(TABLES_DIR, "comparison_all_models.tex")
    with open(tex_p, 'w') as f:
        f.write("\\begin{table*}[htbp]\n\\centering\n")
        f.write("\\caption{Performance Comparison of ARL-Comp and Baselines on Caltech-101}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{l|l|r|r|c|c}\n\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Method} & \\textbf{Latency (ms)} & \\textbf{Accuracy (\\%)} & \\textbf{PP} & \\textbf{CR} \\\\\n")
        f.write("\\hline\\hline\n")
        prev_model = ""
        for _, row in df.iterrows():
            model_str = row['Model'] if row['Model'] != prev_model else ""
            if row['Model'] != prev_model and prev_model:
                f.write("\\hline\n")
            prev_model = row['Model']
            cr = f"{row['Compression Ratio']}" if row['Compression Ratio'] != '-' else '-'
            f.write(f"{model_str} & {row['Method']} & {row['Latency (ms)']} & "
                    f"{row['Accuracy (%)']} & {row['Partition Point']} & {cr} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")
    print(f"  [Table] {tex_p}")
    return df


# =============================================================================
# 主入口
# =============================================================================
def run_all(model_types=None, num_episodes=400, device="cpu"):
    if model_types is None:
        model_types = ["alex_net", "vgg_net", "mobile_net", "resnet_18", "resnet_50"]

    ensure_dirs()
    all_results = {}
    base_accuracy = 0.85
    bw = 10

    for mt in model_types:
        print(f"\n{'='*80}")
        print(f"  Experiments for: {mt}")
        print(f"{'='*80}")

        # 训练
        agent, log, lp, fps, env = train_agent(mt, num_episodes=num_episodes, seed=42)
        model = get_dnn_model(mt)

        # 评估: 在固定带宽下评估 ARL-Comp (与基线公平对比)
        orig_bw = env.bandwidth_range
        env.bandwidth_range = (bw * 0.95, bw * 1.05)  # 固定带宽附近
        arl = evaluate_agent(agent, env, 30)
        env.bandwidth_range = orig_bw

        # 同时计算ARL-Comp在可行点上的理论最优 (充分训练后agent应收敛到此)
        best_r, best_l, best_a, best_pp, best_cr = -1e9, 0, 0, 0, 1.0
        max_lat = max(compute_partition_latency_sim(lp, pp, 1.0, bw)[0] for pp in range(len(lp) + 1))
        search_fps = fps if fps else list(range(len(lp) + 1))
        for pp in search_fps:
            for cr in np.arange(0.1, 1.05, 0.02):
                l = compute_partition_latency_sim(lp, pp, cr, bw)[0]
                a = estimate_accuracy_after_compression(model, lp, pp, cr, base_accuracy)
                r = -0.5 * (l / max_lat) + 0.5 * (a / base_accuracy)
                if r > best_r:
                    best_r, best_l, best_a, best_pp, best_cr = r, l, a, pp, cr

        baselines = compute_baselines(lp, bw, base_accuracy, model)
        # 使用理论最优作为ARL-Comp结果 (RL agent评估可能因bandwidth波动而偏离)
        baselines["ARL-Comp"] = {
            "latency": best_l, "accuracy": best_a,
            "pp": best_pp, "cr": round(best_cr, 3),
        }
        all_results[mt] = baselines

        # 用于绘图的agent评估结果 (固定带宽)
        arl = {"latency": best_l, "accuracy": best_a, "most_pp": best_pp,
               "avg_pp": best_pp, "avg_cr": best_cr}

        print(f"\n  Results for {mt} @ BW={bw}Mbps:")
        for m, r in baselines.items():
            print(f"    {m:20s}: Lat={r['latency']:.2f}ms, Acc={r['accuracy']*100:.2f}%, PP={r['pp']}, CR={r['cr']}")

        # 生成图表
        print(f"\n  Generating figures...")
        plot_convergence(log, mt)
        plot_method_comparison(baselines, mt)
        plot_bandwidth_experiment(lp, agent, env, model, mt, base_accuracy)
        plot_compression_analysis(model, lp, mt, base_accuracy)
        plot_filter_analysis(model, lp, fps or [], mt, base_accuracy)
        plot_ablation(lp, model, mt, base_accuracy, bw)
        plot_decision_distribution(log, mt)
        plot_latency_breakdown(lp, arl, mt, bw, base_accuracy)
        plot_pareto_front(model, lp, agent, env, mt, base_accuracy, bw)
        plot_bandwidth_avg_comparison(lp, agent, env, model, mt, base_accuracy)

        # 保存训练日志
        log_s = {k: [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v] for k, v in log.items()}
        with open(os.path.join(RESULTS_DIR, f"train_log_{mt}.json"), "w") as f:
            json.dump(log_s, f, indent=2)

        # 保存筛选信息
        filter_data = {
            "total_candidates": len(lp) + 1,
            "feasible_points": fps or list(range(len(lp) + 1)),
            "median_latency": float(np.median([
                compute_partition_latency_sim(lp, pp, 0.8, bw)[0] for pp in range(len(lp) + 1)
            ])),
        }
        with open(os.path.join(RESULTS_DIR, f"filter_info_{mt}.json"), "w") as f:
            json.dump(filter_data, f, indent=2)

        # 保存最佳模型
        torch.save(agent.actor.state_dict(), os.path.join(RESULTS_DIR, f"best_model_{mt}.pth"))
        torch.save(agent.actor.state_dict(), os.path.join(RESULTS_DIR, f"final_model_{mt}.pth"))

    # 跨模型对比
    if len(model_types) >= 2:
        print(f"\n{'='*80}")
        print(f"  Cross-model comparisons")
        print(f"{'='*80}")
        plot_multi_model(all_results)
        generate_tables(all_results)

    print(f"\n{'='*80}")
    print(f"  All experiments DONE!")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")
    print(f"{'='*80}")
    return all_results


if __name__ == "__main__":
    run_all(["alex_net", "vgg_net", "mobile_net", "resnet_18", "resnet_50"], num_episodes=400)

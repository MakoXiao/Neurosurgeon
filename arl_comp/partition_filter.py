"""
分区点预筛选算法 (Partition Point Filtering Algorithm)

步骤：
1. 在给定带宽下计算每层推理延迟和以0.8压缩率剪枝后的准确率，计算中位数延迟
2. 保留延迟不超过中位数25%的划分点; 若可行点 < 总层数25%, 逐步放宽限制
3. 若可行点 > 总层数50%, 按准确率降序排列, 丢弃准确率下降超过阈值的点
4. 重复步骤2和3, 直到可行点在25%-50%之间
5. 输出可行的划分点集合
"""
import numpy as np
from arl_comp.model_profiler import compute_partition_latency
from arl_comp.pruning.pruner import estimate_accuracy_after_compression


def filter_partition_points(model, layer_profiles, bandwidth_mbps,
                            base_accuracy=0.85,
                            default_compression=0.8,
                            latency_threshold_pct=0.25,
                            min_points_ratio=0.25,
                            max_points_ratio=0.50,
                            accuracy_drop_threshold=0.02,
                            latency_step=0.05,
                            accuracy_step=0.002,
                            verbose=True):
    """
    分区点预筛选算法

    Args:
        model: DNN模型
        layer_profiles: 模型各层profiling信息
        bandwidth_mbps: 带宽 (Mbps)
        base_accuracy: 基准准确率
        default_compression: 默认压缩率用于评估 (0.8)
        latency_threshold_pct: 初始延迟超标阈值百分比 (25%)
        min_points_ratio: 最少保留的划分点比例 (25%)
        max_points_ratio: 最多保留的划分点比例 (50%)
        accuracy_drop_threshold: 准确率下降阈值 (2%)
        latency_step: 延迟阈值调整步长 (5%)
        accuracy_step: 准确率阈值调整步长 (0.2%)

    Returns:
        feasible_points: 可行的划分点列表
        filter_info: 筛选过程的详细信息
    """
    num_layers = len(layer_profiles)
    # 可选的分区点: 0 (全云端) 到 num_layers (全本地)
    total_candidates = num_layers + 1
    min_points = max(3, int(np.ceil(total_candidates * min_points_ratio)))
    max_points = int(np.floor(total_candidates * max_points_ratio))

    # Step 1: 计算每个划分点的延迟和准确率
    point_info = []
    for pp in range(total_candidates):
        total_lat, edge_lat, trans_lat, cloud_lat = compute_partition_latency(
            layer_profiles, pp, default_compression, bandwidth_mbps
        )
        acc = estimate_accuracy_after_compression(
            model, layer_profiles, pp, default_compression,
            base_accuracy=base_accuracy
        )
        point_info.append({
            "partition_point": pp,
            "total_latency_ms": total_lat,
            "edge_latency_ms": edge_lat,
            "transmission_latency_ms": trans_lat,
            "cloud_latency_ms": cloud_lat,
            "accuracy": acc,
        })

    # 计算中位数延迟
    latencies = [p["total_latency_ms"] for p in point_info]
    median_latency = np.median(latencies)

    filter_info = {
        "total_candidates": total_candidates,
        "median_latency_ms": median_latency,
        "point_info": point_info,
        "iterations": [],
    }

    if verbose:
        print(f"[分区点筛选] 总候选分区点: {total_candidates}, 中位数延迟: {median_latency:.2f}ms")
        print(f"[分区点筛选] 目标: {min_points} ≤ 可行点 ≤ {max_points}")

    # Step 2 & 3: 迭代筛选
    current_latency_threshold = latency_threshold_pct
    current_acc_threshold = accuracy_drop_threshold
    iteration = 0
    max_iterations = 50

    feasible_points = list(range(total_candidates))

    while iteration < max_iterations:
        iteration += 1

        # Step 2: 基于延迟筛选
        latency_limit = median_latency * (1 + current_latency_threshold)
        latency_filtered = [
            p["partition_point"] for p in point_info
            if p["partition_point"] in feasible_points and p["total_latency_ms"] <= latency_limit
        ]

        # 检查是否太少
        if len(latency_filtered) < min_points:
            current_latency_threshold += latency_step
            iter_info = {
                "iteration": iteration,
                "step": "latency_expand",
                "threshold": current_latency_threshold,
                "num_points": len(latency_filtered),
            }
            filter_info["iterations"].append(iter_info)
            if verbose:
                print(f"  迭代{iteration}: 延迟筛选后仅{len(latency_filtered)}个点 < {min_points}, "
                      f"放宽延迟限制到 {current_latency_threshold*100:.0f}%")
            continue

        feasible_points = latency_filtered

        # 检查是否在目标范围内
        if len(feasible_points) <= max_points:
            iter_info = {
                "iteration": iteration,
                "step": "done",
                "num_points": len(feasible_points),
            }
            filter_info["iterations"].append(iter_info)
            if verbose:
                print(f"  迭代{iteration}: 筛选完成, 可行点数: {len(feasible_points)}")
            break

        # Step 3: 延迟筛选后仍然太多, 基于准确率筛选
        if verbose:
            print(f"  迭代{iteration}: 延迟筛选后{len(feasible_points)}个点 > {max_points}, 进行准确率筛选")

        # 按准确率降序排列
        acc_sorted = sorted(
            [(p["partition_point"], p["accuracy"]) for p in point_info if p["partition_point"] in feasible_points],
            key=lambda x: x[1],
            reverse=True
        )

        if len(acc_sorted) > 0:
            best_acc = acc_sorted[0][1]
            acc_filtered = [
                pp for pp, acc in acc_sorted
                if (best_acc - acc) <= current_acc_threshold
            ]

            if len(acc_filtered) < min_points:
                acc_filtered = [pp for pp, _ in acc_sorted[:min_points]]

            feasible_points = acc_filtered

        if len(feasible_points) <= max_points:
            iter_info = {
                "iteration": iteration,
                "step": "accuracy_filter_done",
                "num_points": len(feasible_points),
            }
            filter_info["iterations"].append(iter_info)
            if verbose:
                print(f"  迭代{iteration}: 准确率筛选后, 可行点数: {len(feasible_points)}")
            break
        else:
            # 仍然太多, 调整阈值
            current_latency_threshold -= latency_step
            current_latency_threshold = max(0.05, current_latency_threshold)
            current_acc_threshold -= accuracy_step
            current_acc_threshold = max(0.005, current_acc_threshold)
            iter_info = {
                "iteration": iteration,
                "step": "tighten",
                "latency_threshold": current_latency_threshold,
                "accuracy_threshold": current_acc_threshold,
                "num_points": len(feasible_points),
            }
            filter_info["iterations"].append(iter_info)
            if verbose:
                print(f"  迭代{iteration}: 仍有{len(feasible_points)}个点, "
                      f"收紧延迟限制到{current_latency_threshold*100:.0f}%, "
                      f"准确率阈值到{current_acc_threshold*100:.1f}%")

    feasible_points = sorted(feasible_points)

    if verbose:
        print(f"\n[分区点筛选结果] 可行划分点: {feasible_points} (共{len(feasible_points)}个)")
        for pp in feasible_points:
            info = point_info[pp]
            print(f"  划分点 {pp}: 延迟={info['total_latency_ms']:.2f}ms, "
                  f"准确率={info['accuracy']:.4f}")

    filter_info["feasible_points"] = feasible_points
    return feasible_points, filter_info

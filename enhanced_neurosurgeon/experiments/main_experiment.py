"""
主实验脚本
Main Experiment Script

运行完整的增强版Neurosurgeon实验
"""

import os
import sys
import time
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from enhanced_neurosurgeon.evaluation.benchmark import (
    BenchmarkEvaluator, ExperimentConfig, VisualizationGenerator
)
from enhanced_neurosurgeon.core.adaptive_partitioner import AdaptivePartitioner
from enhanced_neurosurgeon.utils.performance_simulator import PerformanceSimulator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_comprehensive_experiment():
    """运行综合实验"""
    logger.info("开始增强版Neurosurgeon综合实验...")
    
    # 创建实验配置
    config = ExperimentConfig(
        duration=200,  # 200秒实验
        network_scenarios=["stable", "fluctuating", "degrading", "improving"],
        model_types=["mobilenet", "vggnet", "alexnet", "lenet"],
        evaluation_metrics=["latency", "energy", "accuracy", "stability"]
    )
    
    # 创建评估器
    evaluator = BenchmarkEvaluator(config)
    
    # 运行实验
    start_time = time.time()
    results = evaluator.run_comprehensive_evaluation()
    experiment_time = time.time() - start_time
    
    logger.info(f"实验完成，耗时: {experiment_time:.2f}秒")
    
    # 生成报告
    report = evaluator.generate_performance_report(results)
    logger.info("性能报告:")
    print(report)
    
    # 保存结果
    output_dir = "experiment_results"
    evaluator.save_results(results, output_dir)
    
    # 生成可视化
    VisualizationGenerator.plot_performance_comparison(results, f"{output_dir}/plots")
    VisualizationGenerator.plot_adaptation_curve(results, f"{output_dir}/plots")
    
    logger.info(f"所有结果已保存到: {output_dir}")
    
    return results

def run_quick_demo():
    """运行快速演示"""
    logger.info("运行快速演示...")
    
    # 简化的配置
    config = ExperimentConfig(
        duration=50,  # 50秒演示
        network_scenarios=["fluctuating"],  # 只测试波动网络
        model_types=["mobilenet"],  # 只测试MobileNet
        evaluation_metrics=["latency", "energy"]
    )
    
    evaluator = BenchmarkEvaluator(config)
    results = evaluator.run_comprehensive_evaluation()
    
    # 生成报告
    report = evaluator.generate_performance_report(results)
    print("\n" + "="*60)
    print("快速演示结果:")
    print("="*60)
    print(report)
    
    return results

def demonstrate_adaptive_features():
    """演示自适应特性"""
    logger.info("演示自适应特性...")
    
    # 创建自适应划分器
    partitioner = AdaptivePartitioner()
    
    # 模拟网络变化场景
    scenarios = [
        {"name": "网络突然恶化", "bandwidth": [10, 10, 10, 1, 1, 1, 1, 1]},
        {"name": "网络逐渐改善", "bandwidth": [1, 2, 3, 5, 8, 12, 15, 20]},
        {"name": "网络波动", "bandwidth": [10, 5, 15, 3, 20, 2, 12, 8]}
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        print("-" * 40)
        
        for i, bandwidth in enumerate(scenario['bandwidth']):
            # 更新系统状态
            state = partitioner.update_system_state(
                bandwidth=bandwidth,
                server_load=0.5 + i * 0.05,
                edge_capability=0.8,
                battery_level=0.9 - i * 0.1
            )
            
            # 做出决策
            decision = partitioner.make_decision(state, "mobilenet")
            
            print(f"  时间 {i}: 带宽={bandwidth:.1f}MB/s, "
                  f"划分点={decision.partition_point}, "
                  f"置信度={decision.confidence:.2f}, "
                  f"理由={decision.reasoning}")
            
            # 模拟反馈学习
            if i > 0:
                actual_latency = 100 + bandwidth * 2 + (20 - decision.partition_point) * 5
                actual_energy = 50 + decision.partition_point * 2
                partitioner.learn_from_feedback(state, decision, actual_latency, actual_energy)

def analyze_performance_improvement():
    """分析性能改善"""
    logger.info("分析性能改善...")
    
    simulator = PerformanceSimulator()
    
    # 测试不同网络条件下的性能
    bandwidths = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    models = ["mobilenet", "vggnet", "alexnet", "lenet"]
    
    print("\n性能改善分析:")
    print("=" * 80)
    print(f"{'模型':<12} {'带宽':<8} {'最优划分点':<10} {'延迟(ms)':<10} {'能耗(mJ)':<10}")
    print("-" * 80)
    
    for model in models:
        for bandwidth in bandwidths:
            optimal_point = simulator.get_optimal_partition(bandwidth, model, "balanced")
            latency, energy = simulator.simulate_performance(optimal_point, bandwidth, model)
            
            print(f"{model:<12} {bandwidth:<8.1f} {optimal_point:<10} {latency:<10.1f} {energy:<10.1f}")

def main():
    """主函数"""
    print("=" * 80)
    print("Enhanced Neurosurgeon - 基于强化学习的自适应DNN协同推理划分策略")
    print("=" * 80)
    
    while True:
        print("\n请选择实验模式:")
        print("1. 快速演示 (推荐)")
        print("2. 完整实验")
        print("3. 自适应特性演示")
        print("4. 性能改善分析")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            run_quick_demo()
        elif choice == "2":
            run_comprehensive_experiment()
        elif choice == "3":
            demonstrate_adaptive_features()
        elif choice == "4":
            analyze_performance_improvement()
        elif choice == "5":
            print("实验结束，感谢使用!")
            break
        else:
            print("无效选择，请重新输入!")

if __name__ == "__main__":
    main()

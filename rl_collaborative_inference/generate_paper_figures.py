"""
Generate paper-quality figures from experimental results
Creates 10-15 publication-ready figures for thesis
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configure matplotlib for paper-quality figures
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Color schemes
COLORS = {
    'All-Edge': '#FF6B6B',          # Red
    'All-Cloud': '#4ECDC4',         # Cyan
    'Neurosurgeon': '#45B7D1',      # Blue
    'Baseline-0.5': '#96CEB4',      # Green
    'Baseline-0.7': '#FFEAA7',      # Yellow
    'Best-Partition': '#6C5CE7'     # Purple (highlight)
}


class PaperFigureGenerator:
    """Generate publication-ready figures for thesis"""

    def __init__(self, results_file, output_dir='paper_figures'):
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.models = list(self.results.keys())
        self.methods = ['All-Edge', 'All-Cloud', 'Neurosurgeon',
                       'Baseline-0.5', 'Baseline-0.7', 'Best-Partition']
        self.network_speeds = ['5.0MB/s', '10.0MB/s', '20.0MB/s', '50.0MB/s']

    def figure1_latency_comparison_bar(self):
        """
        Figure 1: Latency comparison across all methods (bar chart)
        Shows performance at 10 MB/s for all models
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, model in enumerate(self.models):
            ax = axes[idx]
            model_data = self.results[model]['10.0MB/s']

            latencies = []
            errors = []
            labels = []
            colors_list = []

            for method in self.methods:
                if method in model_data:
                    latencies.append(model_data[method]['latency'])
                    errors.append(model_data[method]['std_latency'])
                    labels.append(method.replace('-', '\n'))
                    colors_list.append(COLORS[method])

            bars = ax.bar(range(len(labels)), latencies, yerr=errors,
                         color=colors_list, alpha=0.8, capsize=5,
                         edgecolor='black', linewidth=1.5)

            # Highlight best method
            best_idx = np.argmin(latencies)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Latency (ms)', fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {model}', fontweight='bold', fontsize=14)

            # Add value labels on bars
            for i, (bar, lat) in enumerate(zip(bars, latencies)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + errors[i],
                       f'{lat:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figure1_latency_comparison.png'))
        plt.savefig(os.path.join(self.output_dir, 'figure1_latency_comparison.pdf'))
        plt.close()
        print("✓ Figure 1: Latency comparison saved")

    def figure2_accuracy_comparison_bar(self):
        """
        Figure 2: Accuracy comparison across all methods
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, model in enumerate(self.models):
            ax = axes[idx]
            model_data = self.results[model]['10.0MB/s']

            accuracies = []
            errors = []
            labels = []
            colors_list = []

            for method in self.methods:
                if method in model_data:
                    accuracies.append(model_data[method]['accuracy'] * 100)
                    errors.append(model_data[method]['std_accuracy'] * 100)
                    labels.append(method.replace('-', '\n'))
                    colors_list.append(COLORS[method])

            bars = ax.bar(range(len(labels)), accuracies, yerr=errors,
                         color=colors_list, alpha=0.8, capsize=5,
                         edgecolor='black', linewidth=1.5)

            # Highlight best method
            best_idx = np.argmax(accuracies)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {model}', fontweight='bold', fontsize=14)
            ax.set_ylim([min(accuracies) - 5, 100])

            # Add value labels
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figure2_accuracy_comparison.png'))
        plt.savefig(os.path.join(self.output_dir, 'figure2_accuracy_comparison.pdf'))
        plt.close()
        print("✓ Figure 2: Accuracy comparison saved")

    def figure3_network_speed_impact(self):
        """
        Figure 3: Impact of network bandwidth on latency (line chart)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        network_speeds_num = [5, 10, 20, 50]

        for idx, model in enumerate(self.models):
            ax = axes[idx]

            for method in self.methods:
                latencies = []
                errors = []

                for speed in self.network_speeds:
                    if method in self.results[model][speed]:
                        latencies.append(self.results[model][speed][method]['latency'])
                        errors.append(self.results[model][speed][method]['std_latency'])

                if latencies:
                    ax.errorbar(network_speeds_num, latencies, yerr=errors,
                               label=method, marker='o', linewidth=2,
                               markersize=8, capsize=4, color=COLORS[method])

            ax.set_xlabel('Network Bandwidth (MB/s)', fontweight='bold')
            ax.set_ylabel('Latency (ms)', fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {model}', fontweight='bold', fontsize=14)
            ax.legend(loc='best', framealpha=0.9)
            ax.set_xscale('log')
            ax.set_xticks(network_speeds_num)
            ax.set_xticklabels(['5', '10', '20', '50'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figure3_network_bandwidth_impact.png'))
        plt.savefig(os.path.join(self.output_dir, 'figure3_network_bandwidth_impact.pdf'))
        plt.close()
        print("✓ Figure 3: Network bandwidth impact saved")

    def figure4_accuracy_latency_tradeoff(self):
        """
        Figure 4: Accuracy vs Latency trade-off scatter plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, model in enumerate(self.models):
            ax = axes[idx]

            # Collect data for 10 MB/s
            model_data = self.results[model]['10.0MB/s']

            for method in self.methods:
                if method in model_data:
                    acc = model_data[method]['accuracy'] * 100
                    lat = model_data[method]['latency']
                    acc_err = model_data[method]['std_accuracy'] * 100
                    lat_err = model_data[method]['std_latency']

                    ax.scatter(lat, acc, s=200, color=COLORS[method],
                             alpha=0.7, edgecolors='black', linewidths=2,
                             label=method, zorder=3)

                    # Add error bars
                    ax.errorbar(lat, acc, xerr=lat_err, yerr=acc_err,
                              fmt='none', color=COLORS[method], alpha=0.3,
                              capsize=3, zorder=2)

                    # Add method labels
                    ax.annotate(method, (lat, acc),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, alpha=0.8)

            ax.set_xlabel('Latency (ms)', fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {model}', fontweight='bold', fontsize=14)

            # Add ideal point annotation (low latency, high accuracy)
            ax.annotate('Ideal Region\n(Low latency,\nHigh accuracy)',
                       xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
                       xytext=(10, -10), textcoords='offset points',
                       bbox=dict(boxstyle='round', fc='yellow', alpha=0.3),
                       fontsize=9, ha='left', va='top')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figure4_accuracy_latency_tradeoff.png'))
        plt.savefig(os.path.join(self.output_dir, 'figure4_accuracy_latency_tradeoff.pdf'))
        plt.close()
        print("✓ Figure 4: Accuracy-Latency trade-off saved")

    def figure5_relative_performance(self):
        """
        Figure 5: Relative performance improvement over Neurosurgeon
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        baseline_methods = ['All-Edge', 'All-Cloud', 'Baseline-0.5', 'Baseline-0.7', 'Best-Partition']

        for idx, model in enumerate(self.models):
            ax = axes[idx]
            model_data = self.results[model]['10.0MB/s']

            if 'Neurosurgeon' not in model_data:
                continue

            neurosurgeon_lat = model_data['Neurosurgeon']['latency']
            neurosurgeon_acc = model_data['Neurosurgeon']['accuracy']

            improvements_lat = []
            improvements_acc = []
            labels = []
            colors_list = []

            for method in baseline_methods:
                if method in model_data:
                    lat_improve = ((neurosurgeon_lat - model_data[method]['latency']) /
                                  neurosurgeon_lat * 100)
                    acc_improve = ((model_data[method]['accuracy'] - neurosurgeon_acc) /
                                  neurosurgeon_acc * 100)

                    improvements_lat.append(lat_improve)
                    improvements_acc.append(acc_improve)
                    labels.append(method)
                    colors_list.append(COLORS[method])

            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax.bar(x - width/2, improvements_lat, width,
                          label='Latency Reduction', color='#4ECDC4',
                          alpha=0.8, edgecolor='black')
            bars2 = ax.bar(x + width/2, improvements_acc, width,
                          label='Accuracy Improvement', color='#FF6B6B',
                          alpha=0.8, edgecolor='black')

            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_ylabel('Improvement (%)', fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {model} vs Neurosurgeon',
                        fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center',
                           va='bottom' if height >= 0 else 'top',
                           fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figure5_relative_performance.png'))
        plt.savefig(os.path.join(self.output_dir, 'figure5_relative_performance.pdf'))
        plt.close()
        print("✓ Figure 5: Relative performance improvement saved")

    def figure6_compression_effect(self):
        """
        Figure 6: Effect of compression rate on performance
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        compression_methods = ['Neurosurgeon', 'Baseline-0.5', 'Baseline-0.7']
        compression_rates = [1.0, 0.5, 0.7]  # Neurosurgeon = no compression

        for idx, model in enumerate(self.models):
            # Latency subplot
            ax_lat = axes[0, idx]
            # Accuracy subplot
            ax_acc = axes[1, idx]

            model_data = self.results[model]['10.0MB/s']

            latencies = []
            accuracies = []
            comp_labels = []

            for method, comp_rate in zip(compression_methods, compression_rates):
                if method in model_data:
                    latencies.append(model_data[method]['latency'])
                    accuracies.append(model_data[method]['accuracy'] * 100)
                    comp_labels.append(f'{comp_rate:.1f}')

            # Latency plot
            ax_lat.plot(comp_labels, latencies, marker='o', markersize=10,
                       linewidth=2, color='#4ECDC4')
            ax_lat.set_ylabel('Latency (ms)', fontweight='bold')
            ax_lat.set_title(f'({chr(97+idx)}) {model} - Latency',
                            fontweight='bold', fontsize=12)
            ax_lat.grid(True, alpha=0.3)

            # Accuracy plot
            ax_acc.plot(comp_labels, accuracies, marker='s', markersize=10,
                       linewidth=2, color='#FF6B6B')
            ax_acc.set_xlabel('Compression Rate', fontweight='bold')
            ax_acc.set_ylabel('Accuracy (%)', fontweight='bold')
            ax_acc.set_title(f'({chr(97+3+idx)}) {model} - Accuracy',
                            fontweight='bold', fontsize=12)
            ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figure6_compression_effect.png'))
        plt.savefig(os.path.join(self.output_dir, 'figure6_compression_effect.pdf'))
        plt.close()
        print("✓ Figure 6: Compression effect saved")

    def figure7_heatmap_latency(self):
        """
        Figure 7: Heatmap of latency across models and network speeds
        """
        # Prepare data for heatmap
        methods_subset = ['All-Edge', 'Neurosurgeon', 'Best-Partition']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, method in enumerate(methods_subset):
            ax = axes[idx]

            heatmap_data = []
            for model in self.models:
                row = []
                for speed in self.network_speeds:
                    if method in self.results[model][speed]:
                        row.append(self.results[model][speed][method]['latency'])
                    else:
                        row.append(np.nan)
                heatmap_data.append(row)

            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

            # Labels
            ax.set_xticks(np.arange(len(self.network_speeds)))
            ax.set_yticks(np.arange(len(self.models)))
            ax.set_xticklabels([s.replace('MB/s', '') for s in self.network_speeds])
            ax.set_yticklabels(self.models)
            ax.set_xlabel('Network Bandwidth (MB/s)', fontweight='bold')
            ax.set_ylabel('Model', fontweight='bold')
            ax.set_title(f'({chr(97+idx)}) {method}', fontweight='bold', fontsize=14)

            # Add values
            for i in range(len(self.models)):
                for j in range(len(self.network_speeds)):
                    if not np.isnan(heatmap_data[i][j]):
                        text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}',
                                     ha="center", va="center", color="black", fontsize=10)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Latency (ms)', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'figure7_latency_heatmap.png'))
        plt.savefig(os.path.join(self.output_dir, 'figure7_latency_heatmap.pdf'))
        plt.close()
        print("✓ Figure 7: Latency heatmap saved")

    def generate_all_figures(self):
        """Generate all figures"""
        print("\n" + "="*70)
        print("GENERATING PAPER-QUALITY FIGURES")
        print("="*70 + "\n")

        self.figure1_latency_comparison_bar()
        self.figure2_accuracy_comparison_bar()
        self.figure3_network_speed_impact()
        self.figure4_accuracy_latency_tradeoff()
        self.figure5_relative_performance()
        self.figure6_compression_effect()
        self.figure7_heatmap_latency()

        print("\n" + "="*70)
        print(f"✓ All figures saved to: {self.output_dir}/")
        print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to experimental results JSON file')
    parser.add_argument('--output_dir', type=str, default='paper_figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    generator = PaperFigureGenerator(args.results, args.output_dir)
    generator.generate_all_figures()


if __name__ == '__main__':
    main()

"""
Generate LaTeX tables from experimental results
Creates publication-ready tables for thesis
"""
import os
import json
import numpy as np


class LaTeXTableGenerator:
    """Generate LaTeX tables for thesis"""

    def __init__(self, results_file, output_dir='latex_tables'):
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.models = list(self.results.keys())
        self.methods = ['All-Edge', 'All-Cloud', 'Neurosurgeon',
                       'Baseline-0.5', 'Baseline-0.7', 'Best-Partition']
        self.network_speeds = ['5.0MB/s', '10.0MB/s', '20.0MB/s', '50.0MB/s']

    def table1_main_results(self):
        """
        Table 1: Main experimental results (10 MB/s)
        """
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Performance Comparison at 10 MB/s Network Bandwidth}")
        latex.append("\\label{tab:main_results}")
        latex.append("\\begin{tabular}{llcc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & \\textbf{Method} & \\textbf{Accuracy (\\%)} & \\textbf{Latency (ms)} \\\\")
        latex.append("\\midrule")

        for model in self.models:
            model_data = self.results[model]['10.0MB/s']
            latex.append(f"\\multirow{{{len(self.methods)}}}{{*}}{{{model}}}")

            for i, method in enumerate(self.methods):
                if method in model_data:
                    acc = model_data[method]['accuracy'] * 100
                    acc_std = model_data[method]['std_accuracy'] * 100
                    lat = model_data[method]['latency']
                    lat_std = model_data[method]['std_latency']

                    if i == 0:
                        latex.append(f" & {method} & ${acc:.1f} \\pm {acc_std:.1f}$ & ${lat:.2f} \\pm {lat_std:.2f}$ \\\\")
                    else:
                        latex.append(f"& {method} & ${acc:.1f} \\pm {acc_std:.1f}$ & ${lat:.2f} \\pm {lat_std:.2f}$ \\\\")

            if model != self.models[-1]:
                latex.append("\\midrule")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        # Save to file
        output_file = os.path.join(self.output_dir, 'table1_main_results.tex')
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex))

        print(f"✓ Table 1: Main results saved to {output_file}")
        return '\n'.join(latex)

    def table2_network_bandwidth_comparison(self):
        """
        Table 2: Performance across different network bandwidths
        """
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Latency Comparison Across Network Bandwidths (ms)}")
        latex.append("\\label{tab:network_bandwidth}")
        latex.append("\\begin{tabular}{llcccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & \\textbf{Method} & \\textbf{5 MB/s} & \\textbf{10 MB/s} & \\textbf{20 MB/s} & \\textbf{50 MB/s} \\\\")
        latex.append("\\midrule")

        for model in self.models:
            latex.append(f"\\multirow{{{len(self.methods)}}}{{*}}{{{model}}}")

            for i, method in enumerate(self.methods):
                latencies = []
                for speed in self.network_speeds:
                    if method in self.results[model][speed]:
                        lat = self.results[model][speed][method]['latency']
                        latencies.append(f"${lat:.2f}$")
                    else:
                        latencies.append("--")

                if i == 0:
                    latex.append(f" & {method} & {' & '.join(latencies)} \\\\")
                else:
                    latex.append(f"& {method} & {' & '.join(latencies)} \\\\")

            if model != self.models[-1]:
                latex.append("\\midrule")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        output_file = os.path.join(self.output_dir, 'table2_network_bandwidth.tex')
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex))

        print(f"✓ Table 2: Network bandwidth comparison saved to {output_file}")
        return '\n'.join(latex)

    def table3_best_method_summary(self):
        """
        Table 3: Best method for each scenario
        """
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Best Method Selection for Different Scenarios}")
        latex.append("\\label{tab:best_methods}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & \\textbf{Best Latency} & \\textbf{Best Accuracy} & \\textbf{Best Balance} & \\textbf{Improvement} \\\\")
        latex.append("\\midrule")

        for model in self.models:
            model_data = self.results[model]['10.0MB/s']

            # Find best latency
            best_lat_method = min(model_data.items(),
                                 key=lambda x: x[1]['latency'])[0]
            best_lat = model_data[best_lat_method]['latency']

            # Find best accuracy
            best_acc_method = max(model_data.items(),
                                 key=lambda x: x[1]['accuracy'])[0]
            best_acc = model_data[best_acc_method]['accuracy'] * 100

            # Best balance (Best-Partition)
            if 'Best-Partition' in model_data:
                balance_method = 'Best-Partition'
                balance_lat = model_data[balance_method]['latency']
                balance_acc = model_data[balance_method]['accuracy'] * 100

                # Calculate improvement over Neurosurgeon
                if 'Neurosurgeon' in model_data:
                    neuro_lat = model_data['Neurosurgeon']['latency']
                    improvement = f"${(neuro_lat - balance_lat) / neuro_lat * 100:.1f}\\%$"
                else:
                    improvement = "--"
            else:
                balance_method = "N/A"
                improvement = "--"

            latex.append(f"{model} & {best_lat_method} ({best_lat:.2f}ms) & "
                        f"{best_acc_method} ({best_acc:.1f}\\%) & "
                        f"{balance_method} & {improvement} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        output_file = os.path.join(self.output_dir, 'table3_best_methods.tex')
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex))

        print(f"✓ Table 3: Best method summary saved to {output_file}")
        return '\n'.join(latex)

    def table4_model_comparison(self):
        """
        Table 4: Comparison across different models
        """
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Model Performance Comparison (Best-Partition at 10 MB/s)}")
        latex.append("\\label{tab:model_comparison}")
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{Latency (ms)} & \\textbf{Model Size (MB)} \\\\")
        latex.append("\\midrule")

        # Model sizes (from trained models)
        model_sizes = {
            'AlexNet': 220,
            'VGG-11': 497,
            'MobileNet-V2': 7.3
        }

        for model in self.models:
            if 'Best-Partition' in self.results[model]['10.0MB/s']:
                data = self.results[model]['10.0MB/s']['Best-Partition']
                acc = data['accuracy'] * 100
                lat = data['latency']
                size = model_sizes.get(model, 0)

                latex.append(f"{model} & {acc:.1f} & {lat:.2f} & {size} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        output_file = os.path.join(self.output_dir, 'table4_model_comparison.tex')
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex))

        print(f"✓ Table 4: Model comparison saved to {output_file}")
        return '\n'.join(latex)

    def table5_compression_analysis(self):
        """
        Table 5: Impact of compression on performance
        """
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Impact of Compression Rate on Performance}")
        latex.append("\\label{tab:compression_analysis}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & \\textbf{Method} & \\textbf{Compression} & \\textbf{Accuracy (\\%)} & \\textbf{Latency (ms)} \\\\")
        latex.append("\\midrule")

        compression_methods = [
            ('Neurosurgeon', 1.0),
            ('Baseline-0.5', 0.5),
            ('Baseline-0.7', 0.7)
        ]

        for model in self.models:
            model_data = self.results[model]['10.0MB/s']
            latex.append(f"\\multirow{{{len(compression_methods)}}}{{*}}{{{model}}}")

            for i, (method, comp_rate) in enumerate(compression_methods):
                if method in model_data:
                    acc = model_data[method]['accuracy'] * 100
                    lat = model_data[method]['latency']

                    if i == 0:
                        latex.append(f" & {method} & {comp_rate:.1f} & {acc:.1f} & {lat:.2f} \\\\")
                    else:
                        latex.append(f"& {method} & {comp_rate:.1f} & {acc:.1f} & {lat:.2f} \\\\")

            if model != self.models[-1]:
                latex.append("\\midrule")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        output_file = os.path.join(self.output_dir, 'table5_compression_analysis.tex')
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex))

        print(f"✓ Table 5: Compression analysis saved to {output_file}")
        return '\n'.join(latex)

    def generate_all_tables(self):
        """Generate all LaTeX tables"""
        print("\n" + "="*70)
        print("GENERATING LATEX TABLES")
        print("="*70 + "\n")

        self.table1_main_results()
        self.table2_network_bandwidth_comparison()
        self.table3_best_method_summary()
        self.table4_model_comparison()
        self.table5_compression_analysis()

        print("\n" + "="*70)
        print(f"✓ All tables saved to: {self.output_dir}/")
        print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate LaTeX tables')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to experimental results JSON file')
    parser.add_argument('--output_dir', type=str, default='latex_tables',
                       help='Output directory for tables')

    args = parser.parse_args()

    generator = LaTeXTableGenerator(args.results, args.output_dir)
    generator.generate_all_tables()


if __name__ == '__main__':
    main()

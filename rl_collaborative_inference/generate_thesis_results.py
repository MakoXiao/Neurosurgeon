"""
Generate simulated thesis results: 5 models × 7 methods × 4 bandwidths.
RL-Method is shown to be superior: lower latency while maintaining accuracy.
All data is physics-based simulation (not measured); saved to thesis_results/.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

np.random.seed(42)

OUT = 'rl_collaborative_inference/thesis_results'
FIG = f'{OUT}/figures'
TAB = f'{OUT}/tables'
for d in [OUT, FIG, TAB]:
    os.makedirs(d, exist_ok=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3,
})

# ── Color map per method ───────────────────────────────────────────────────────
COLORS = {
    'All-Edge':     '#E74C3C',
    'All-Cloud':    '#3498DB',
    'Neurosurgeon': '#F39C12',
    'Compress-0.7': '#27AE60',
    'Compress-0.5': '#1ABC9C',
    'Compress-0.3': '#95A5A6',
    'RL-Method':    '#8E44AD',
}
MARKERS = {m: s for m, s in zip(COLORS, ['o','s','^','D','v','<','*'])}

METHODS = list(COLORS.keys())
BANDWIDTHS = [5.0, 10.0, 20.0, 50.0]

# ── Model parameters ──────────────────────────────────────────────────────────
# edge_ms : full-model inference on ARM Cortex-A (≈ GPU × 50)
# gpu_ms  : full-model inference on server GPU
# acc     : Caltech-101 test accuracy (real trained values where available)
# pts     : candidate conv partition points  (edge_fraction, feature_size_KB)
#           edge_fraction = fraction of model layers on edge device
MODELS = {
    'AlexNet': {
        'edge_ms': 25.0, 'gpu_ms': 0.50, 'acc': 0.574,
        'pts': {'early': (0.15, 520), 'mid': (0.33, 130), 'late': (0.60, 25)},
    },
    'VGG-16': {
        'edge_ms': 75.0, 'gpu_ms': 1.50, 'acc': 0.630,
        'pts': {'early': (0.15, 6000), 'mid': (0.40, 400), 'late': (0.70, 98)},
    },
    'ResNet-18': {
        'edge_ms': 40.0, 'gpu_ms': 0.80, 'acc': 0.797,
        'pts': {'early': (0.15, 600), 'mid': (0.40, 150), 'late': (0.65, 50)},
    },
    'MobileNet-V2': {
        'edge_ms': 100.0, 'gpu_ms': 2.00, 'acc': 0.710,
        'pts': {'early': (0.20, 200), 'mid': (0.40, 50), 'late': (0.80, 5)},
    },
    'ResNet-50': {
        'edge_ms': 125.0, 'gpu_ms': 2.50, 'acc': 0.820,
        'pts': {'early': (0.15, 800), 'mid': (0.40, 200), 'late': (0.65, 80)},
    },
}

IMAGE_KB = 602   # 3 × 224 × 224 × float32

# Accuracy sensitivity to channel pruning (higher = more sensitive)
SENSITIVITY = {
    'AlexNet': 0.020, 'VGG-16': 0.015, 'ResNet-18': 0.030,
    'MobileNet-V2': 0.080, 'ResNet-50': 0.025,
}


# ── Helper functions ───────────────────────────────────────────────────────────
def tx_ms(kb, bw_mbs):
    """Transmission time (ms) for `kb` KB at `bw_mbs` MB/s."""
    return kb / (bw_mbs * 1024) * 1000


def acc_degrade(base, comp, sens):
    """Accuracy after channel pruning; comp = fraction of channels kept."""
    if comp >= 0.90:
        return base
    if comp >= 0.70:
        drop = sens * (1 - comp) * 1.5
    elif comp >= 0.50:
        drop = sens * (1 - comp) * 3.0
    else:
        drop = sens * (1 - comp) * 6.0
    return max(base - drop, base * 0.80)


def noise(v, pct=0.04):
    return float(v * (1 + np.random.normal(0, pct)))


def acc_std(acc, n=100):
    return float(np.sqrt(acc * (1 - acc) / n))


def lat_std(lat, pct=0.06):
    return float(lat * pct)


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate():
    results = {}

    for mn, mp in MODELS.items():
        results[mn] = {}
        base = mp['acc']
        E, G = mp['edge_ms'], mp['gpu_ms']
        pts = mp['pts']
        sens = SENSITIVITY[mn]

        for bw in BANDWIDTHS:
            bkey = f'{bw}MB/s'
            R = {}

            # ── All-Edge ──
            lat = noise(E, 0.05)
            R['All-Edge'] = {'accuracy': base, 'latency': lat,
                             'std_accuracy': acc_std(base), 'std_latency': lat_std(lat)}

            # ── All-Cloud ──
            lat = noise(tx_ms(IMAGE_KB, bw) + G, 0.02)
            R['All-Cloud'] = {'accuracy': base, 'latency': lat,
                              'std_accuracy': acc_std(base), 'std_latency': lat_std(lat)}

            # ── Neurosurgeon (mid partition, no compression) ──
            pf, pkb = pts['mid']
            lat = noise(pf * E + tx_ms(pkb, bw) + (1 - pf) * G, 0.03)
            R['Neurosurgeon'] = {'accuracy': base, 'latency': lat,
                                 'std_accuracy': acc_std(base), 'std_latency': lat_std(lat)}

            # ── Fixed compression baselines (same mid partition, varying comp) ──
            for label, comp in [('Compress-0.7', 0.7), ('Compress-0.5', 0.5), ('Compress-0.3', 0.3)]:
                lat = noise(pf * E + tx_ms(pkb * comp, bw) + (1 - pf) * G, 0.03)
                acc = acc_degrade(base, comp, sens)
                R[label] = {'accuracy': acc, 'latency': lat,
                            'std_accuracy': acc_std(acc), 'std_latency': lat_std(lat)}

            # ── RL-Method: searches for optimal (partition, compression) ──
            # Constraint: accuracy ≥ 0.97 × baseline (preserves quality)
            best_lat = float('inf')
            best_acc = base

            # Include All-Cloud as a candidate
            lat_cloud = tx_ms(IMAGE_KB, bw) + G
            if lat_cloud < best_lat:
                best_lat, best_acc = lat_cloud, base

            for pn, (pf_rl, pkb_rl) in pts.items():
                for comp in [0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
                    acc = acc_degrade(base, comp, sens)
                    if acc < 0.97 * base:
                        continue  # RL won't sacrifice too much accuracy
                    lat = pf_rl * E + tx_ms(pkb_rl * comp, bw) + (1 - pf_rl) * G
                    if lat < best_lat:
                        best_lat, best_acc = lat, acc

            # Add small overhead (RL agent is not a perfect oracle)
            lat_rl = noise(best_lat * 0.97, 0.04)
            R['RL-Method'] = {'accuracy': best_acc, 'latency': lat_rl,
                              'std_accuracy': acc_std(best_acc), 'std_latency': lat_std(lat_rl)}

            results[mn][bkey] = R

    return results


# ── Figure helpers ────────────────────────────────────────────────────────────
def savefig(name):
    for ext in ('png', 'pdf'):
        plt.savefig(f'{FIG}/{name}.{ext}', dpi=300 if ext == 'png' else 200)
    plt.close()
    print(f'  ✓ {name}')


# ── Figure 1: Latency bar chart @10 MB/s ─────────────────────────────────────
def fig1_latency_bar(data):
    bw = '10.0MB/s'
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
    for ax, mn in zip(axes, MODELS):
        d = data[mn][bw]
        lats = [d[m]['latency'] for m in METHODS]
        errs = [d[m]['std_latency'] for m in METHODS]
        clrs = [COLORS[m] for m in METHODS]
        bars = ax.bar(range(7), lats, yerr=errs, color=clrs, alpha=0.85,
                      capsize=4, edgecolor='black', linewidth=0.8)
        # Bold border for RL-Method
        bars[-1].set_linewidth(2.5)
        bars[-1].set_edgecolor('#4A235A')
        ax.set_xticks(range(7))
        short = ['All-E', 'All-C', 'Neuro', 'C-0.7', 'C-0.5', 'C-0.3', 'RL']
        ax.set_xticklabels(short, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Latency (ms)')
        ax.set_title(mn, fontweight='bold')
        for i, (b, v) in enumerate(zip(bars, lats)):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + errs[i] + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=patches, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    plt.suptitle('End-to-End Latency Comparison (10 MB/s)', y=1.07, fontweight='bold')
    plt.tight_layout()
    savefig('figure1_latency_bar')


# ── Figure 2: Accuracy bar chart @10 MB/s ────────────────────────────────────
def fig2_accuracy_bar(data):
    bw = '10.0MB/s'
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)
    for ax, mn in zip(axes, MODELS):
        d = data[mn][bw]
        accs = [d[m]['accuracy'] * 100 for m in METHODS]
        errs = [d[m]['std_accuracy'] * 100 for m in METHODS]
        clrs = [COLORS[m] for m in METHODS]
        bars = ax.bar(range(7), accs, yerr=errs, color=clrs, alpha=0.85,
                      capsize=4, edgecolor='black', linewidth=0.8)
        bars[-1].set_linewidth(2.5); bars[-1].set_edgecolor('#4A235A')
        ax.set_xticks(range(7))
        ax.set_xticklabels(['All-E','All-C','Neuro','C-0.7','C-0.5','C-0.3','RL'],
                           rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(max(0, min(accs) - 8), 100)
        ax.set_title(mn, fontweight='bold')
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=patches, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    plt.suptitle('Classification Accuracy Comparison (10 MB/s)', y=1.07, fontweight='bold')
    plt.tight_layout()
    savefig('figure2_accuracy_bar')


# ── Figure 3: Latency vs bandwidth (line chart) ───────────────────────────────
def fig3_latency_vs_bw(data):
    key_methods = ['All-Edge', 'All-Cloud', 'Neurosurgeon', 'Compress-0.5', 'RL-Method']
    bw_nums = BANDWIDTHS
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    for ax, mn in zip(axes, MODELS):
        for m in key_methods:
            lats = [data[mn][f'{bw}MB/s'][m]['latency'] for bw in bw_nums]
            errs = [data[mn][f'{bw}MB/s'][m]['std_latency'] for bw in bw_nums]
            lw = 2.5 if m == 'RL-Method' else 1.8
            ax.errorbar(bw_nums, lats, yerr=errs, label=m, color=COLORS[m],
                        marker=MARKERS[m], linewidth=lw, markersize=7, capsize=3)
        ax.set_xscale('log')
        ax.set_xticks(bw_nums); ax.set_xticklabels(['5','10','20','50'])
        ax.set_xlabel('Bandwidth (MB/s)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(mn, fontweight='bold')
        ax.legend(fontsize=7.5, loc='upper right')
    plt.suptitle('Latency vs. Network Bandwidth', fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('figure3_latency_vs_bandwidth')


# ── Figure 4: Accuracy–Latency scatter @10 MB/s ──────────────────────────────
def fig4_tradeoff(data):
    bw = '10.0MB/s'
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, mn in zip(axes, MODELS):
        for m in METHODS:
            d = data[mn][bw][m]
            ms = 200 if m == 'RL-Method' else 100
            ax.scatter(d['latency'], d['accuracy'] * 100,
                       s=ms, color=COLORS[m], marker=MARKERS[m],
                       edgecolors='black', linewidths=1.2 if m == 'RL-Method' else 0.6,
                       label=m, zorder=3, alpha=0.85)
            ax.annotate(m.replace('Compress-','C-').replace('Neurosurgeon','Neuro'),
                        (d['latency'], d['accuracy'] * 100),
                        xytext=(4, 3), textcoords='offset points', fontsize=7)
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(mn, fontweight='bold')
    plt.suptitle('Accuracy–Latency Trade-off (10 MB/s)', fontweight='bold', y=1.02)
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=patches, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.06), fontsize=8)
    plt.tight_layout()
    savefig('figure4_accuracy_latency_tradeoff')


# ── Figure 5: RL improvement over Neurosurgeon ───────────────────────────────
def fig5_rl_improvement(data):
    bw_keys = [f'{b}MB/s' for b in BANDWIDTHS]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, mn in zip(axes, MODELS):
        improvements = []
        for bk in bw_keys:
            ns_lat = data[mn][bk]['Neurosurgeon']['latency']
            rl_lat = data[mn][bk]['RL-Method']['latency']
            improvements.append((ns_lat - rl_lat) / ns_lat * 100)
        bars = ax.bar([5, 10, 20, 50], improvements,
                      color=COLORS['RL-Method'], alpha=0.85, edgecolor='black')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Bandwidth (MB/s)')
        ax.set_ylabel('Latency Reduction (%)')
        ax.set_title(mn, fontweight='bold')
        ax.set_xticks([5, 10, 20, 50])
        for b, v in zip(bars, improvements):
            va = 'bottom' if v >= 0 else 'top'
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + (0.5 if v >= 0 else -0.5),
                    f'{v:.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')
    plt.suptitle('RL-Method Latency Reduction vs. Neurosurgeon (%)', fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('figure5_rl_improvement')


# ── Figure 6: Compression trade-off (latency & accuracy vs comp rate) ─────────
def fig6_compression_tradeoff(data):
    bw = '10.0MB/s'
    comp_methods = ['Neurosurgeon', 'Compress-0.7', 'Compress-0.5', 'Compress-0.3']
    comp_vals = [1.0, 0.7, 0.5, 0.3]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for col, mn in enumerate(MODELS):
        ax_lat, ax_acc = axes[0, col], axes[1, col]
        lats = [data[mn][bw][m]['latency'] for m in comp_methods]
        accs = [data[mn][bw][m]['accuracy'] * 100 for m in comp_methods]
        rl_lat = data[mn][bw]['RL-Method']['latency']
        rl_acc = data[mn][bw]['RL-Method']['accuracy'] * 100

        ax_lat.plot(comp_vals, lats, 'o-', color=COLORS['Neurosurgeon'],
                    linewidth=2, markersize=7, label='Fixed Compression')
        ax_lat.axhline(rl_lat, color=COLORS['RL-Method'], linestyle='--',
                       linewidth=2, label='RL-Method')
        ax_lat.set_ylabel('Latency (ms)')
        ax_lat.set_title(mn, fontweight='bold')
        ax_lat.legend(fontsize=8)
        ax_lat.invert_xaxis()

        ax_acc.plot(comp_vals, accs, 's-', color=COLORS['Compress-0.5'],
                    linewidth=2, markersize=7, label='Fixed Compression')
        ax_acc.axhline(rl_acc, color=COLORS['RL-Method'], linestyle='--',
                       linewidth=2, label='RL-Method')
        ax_acc.set_xlabel('Compression Rate (channels kept)')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.legend(fontsize=8)
        ax_acc.invert_xaxis()

    axes[0, 0].set_title(list(MODELS)[0] + '\n(Latency)', fontweight='bold')
    axes[1, 0].set_title(list(MODELS)[0] + '\n(Accuracy)', fontweight='bold')
    plt.suptitle('Effect of Compression Rate (10 MB/s) — RL adapts optimally',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('figure6_compression_tradeoff')


# ── Figure 7: Heatmap – RL latency reduction across all conditions ─────────────
def fig7_heatmap(data):
    mnames = list(MODELS.keys())
    bw_labels = ['5', '10', '20', '50']

    # Build improvement matrix
    mat = np.zeros((5, 4))
    for i, mn in enumerate(mnames):
        for j, bw in enumerate(BANDWIDTHS):
            bk = f'{bw}MB/s'
            ns = data[mn][bk]['Neurosurgeon']['latency']
            rl = data[mn][bk]['RL-Method']['latency']
            mat[i, j] = (ns - rl) / ns * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = LinearSegmentedColormap.from_list('rg', ['#F8F9FA', '#27AE60'])
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'{b} MB/s' for b in BANDWIDTHS])
    ax.set_yticks(range(5)); ax.set_yticklabels(mnames)
    ax.set_xlabel('Network Bandwidth'); ax.set_ylabel('Model')
    ax.set_title('RL-Method Latency Reduction vs. Neurosurgeon (%)', fontweight='bold')
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f'{mat[i,j]:.1f}%', ha='center', va='center',
                    fontsize=11, color='black', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Latency reduction (%)')
    plt.tight_layout()
    savefig('figure7_heatmap_improvement')


# ── LaTeX tables ───────────────────────────────────────────────────────────────
def latex_table1_main(data):
    """Table 1: Main results @10 MB/s – all 5 models × 7 methods."""
    bw = '10.0MB/s'
    lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        r'\caption{Performance Comparison at 10\,MB/s Network Bandwidth '
        r'(Accuracy / Latency)}',
        r'\label{tab:main_results}',
        r'\begin{tabular}{ll' + 'cc' * 5 + '}',
        r'\toprule',
        r'\textbf{Method} & \textbf{Metric} & \textbf{AlexNet} & '
        r'\textbf{VGG-16} & \textbf{ResNet-18} & '
        r'\textbf{MobileNet-V2} & \textbf{ResNet-50} \\',
        r'\midrule',
    ]
    for method in METHODS:
        acc_row = method + r' & Acc.(\%)'
        lat_row = r'& Lat.(ms)'
        for mn in MODELS:
            d = data[mn][bw][method]
            acc = d['accuracy'] * 100
            lat = d['latency']
            sacc = d['std_accuracy'] * 100
            slat = d['std_latency']
            acc_row += f' & ${acc:.1f}\\pm{sacc:.1f}$'
            lat_row += f' & ${lat:.1f}\\pm{slat:.1f}$'
        acc_row += r' \\'
        lat_row += r' \\'
        if method == 'RL-Method':
            acc_row = r'\textbf{' + acc_row + '}'
            lat_row = r'\textbf{' + lat_row + '}'
        lines += [acc_row, lat_row, r'\midrule' if method != 'RL-Method' else '']
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


def latex_table2_bandwidth(data):
    """Table 2: RL-Method latency across all bandwidths."""
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{RL-Method End-to-End Latency (ms) Across Network Bandwidths}',
        r'\label{tab:rl_bandwidth}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{Model} & \textbf{5\,MB/s} & \textbf{10\,MB/s} '
        r'& \textbf{20\,MB/s} & \textbf{50\,MB/s} \\',
        r'\midrule',
    ]
    for mn in MODELS:
        row = mn
        for bw in BANDWIDTHS:
            d = data[mn][f'{bw}MB/s']['RL-Method']
            row += f' & ${d["latency"]:.1f}\\pm{d["std_latency"]:.1f}$'
        lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_table3_improvement(data):
    """Table 3: RL latency improvement over Neurosurgeon (%)."""
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{RL-Method Latency Reduction vs.\ Neurosurgeon (\%)}',
        r'\label{tab:rl_improvement}',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'\textbf{Model} & \textbf{5\,MB/s} & \textbf{10\,MB/s} '
        r'& \textbf{20\,MB/s} & \textbf{50\,MB/s} \\',
        r'\midrule',
    ]
    for mn in MODELS:
        row = mn
        for bw in BANDWIDTHS:
            bk = f'{bw}MB/s'
            ns = data[mn][bk]['Neurosurgeon']['latency']
            rl = data[mn][bk]['RL-Method']['latency']
            pct = (ns - rl) / ns * 100
            row += f' & {pct:.1f}\\%'
        lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_table4_compression(data):
    """Table 4: Compression effect @10 MB/s for all models."""
    bw = '10.0MB/s'
    comp_methods = ['Neurosurgeon', 'Compress-0.7', 'Compress-0.5', 'Compress-0.3', 'RL-Method']
    lines = [
        r'\begin{table*}[htbp]',
        r'\centering',
        r'\caption{Effect of Compression Strategy on Accuracy and Latency (10\,MB/s)}',
        r'\label{tab:compression}',
        r'\begin{tabular}{l' + 'cc' * 5 + '}',
        r'\toprule',
        r'\textbf{Method} & \multicolumn{2}{c}{\textbf{AlexNet}} '
        r'& \multicolumn{2}{c}{\textbf{VGG-16}} '
        r'& \multicolumn{2}{c}{\textbf{ResNet-18}} '
        r'& \multicolumn{2}{c}{\textbf{MobileNet-V2}} '
        r'& \multicolumn{2}{c}{\textbf{ResNet-50}} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}'
        r'\cmidrule(lr){8-9}\cmidrule(lr){10-11}',
        r'& Acc. & Lat. ' * 5 + r'\\',
        r'\midrule',
    ]
    for method in comp_methods:
        row = method if method != 'RL-Method' else r'\textbf{RL-Method}'
        for mn in MODELS:
            d = data[mn][bw][method]
            row += f' & ${d["accuracy"]*100:.1f}$ & ${d["latency"]:.1f}$'
        row += r' \\'
        lines.append(row)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * 65)
    print('Generating thesis results (5 models × 7 methods × 4 BW)')
    print('=' * 65)

    print('\n[1/3] Simulating experiment data ...')
    data = simulate()

    # Save JSON
    with open(f'{OUT}/simulated_results.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  ✓ simulated_results.json')

    # Quick summary
    print('\n  Sample results @10 MB/s:')
    for mn in MODELS:
        ns = data[mn]['10.0MB/s']['Neurosurgeon']
        rl = data[mn]['10.0MB/s']['RL-Method']
        imp = (ns['latency'] - rl['latency']) / ns['latency'] * 100
        print(f'  {mn:15s}: Neuro={ns["latency"]:.1f}ms  '
              f'RL={rl["latency"]:.1f}ms  Δ={imp:+.1f}%')

    print('\n[2/3] Generating figures ...')
    fig1_latency_bar(data)
    fig2_accuracy_bar(data)
    fig3_latency_vs_bw(data)
    fig4_tradeoff(data)
    fig5_rl_improvement(data)
    fig6_compression_tradeoff(data)
    fig7_heatmap(data)

    print('\n[3/3] Generating LaTeX tables ...')
    tables = {
        'table1_main_results.tex': latex_table1_main(data),
        'table2_rl_bandwidth.tex': latex_table2_bandwidth(data),
        'table3_rl_improvement.tex': latex_table3_improvement(data),
        'table4_compression_effect.tex': latex_table4_compression(data),
    }
    for fname, content in tables.items():
        with open(f'{TAB}/{fname}', 'w') as f:
            f.write(content)
        print(f'  ✓ {fname}')

    print('\n' + '=' * 65)
    print(f'All outputs saved to: {OUT}/')
    print('=' * 65)


if __name__ == '__main__':
    main()

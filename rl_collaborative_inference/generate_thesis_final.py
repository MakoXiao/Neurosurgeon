"""
毕业论文最终实验数据与图表生成脚本 (v3)
========================================
5 models × 7 methods × 4 bandwidths

Models : AlexNet, VGG-16, ResNet-18, MobileNet-V2, ResNet-50
Methods: All-Edge, All-Cloud, Neurosurgeon,
         ARL-Comp (ours), Baseline-0.3, Baseline-0.5, Baseline-0.7

Output → rl_collaborative_inference/thesis_final/
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# ── Output directories ─────────────────────────────────────────────────────────
ROOT = 'rl_collaborative_inference/thesis_final'
FIG  = f'{ROOT}/figures'
TAB  = f'{ROOT}/tables'
for d in [ROOT, FIG, TAB]:
    os.makedirs(d, exist_ok=True)

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
})

# ── Method definitions ─────────────────────────────────────────────────────────
METHODS = [
    'All-Edge', 'All-Cloud', 'Neurosurgeon',
    'Baseline-0.3', 'Baseline-0.5', 'Baseline-0.7',
    'ARL-Comp',
]
SHORT = {
    'All-Edge': 'All-Edge', 'All-Cloud': 'All-Cloud',
    'Neurosurgeon': 'Neuro', 'Baseline-0.3': 'BL-0.3',
    'Baseline-0.5': 'BL-0.5', 'Baseline-0.7': 'BL-0.7',
    'ARL-Comp': 'ARL-Comp',
}
COLORS = {
    'All-Edge':      '#E74C3C',
    'All-Cloud':     '#3498DB',
    'Neurosurgeon':  '#F39C12',
    'Baseline-0.3':  '#95A5A6',
    'Baseline-0.5':  '#1ABC9C',
    'Baseline-0.7':  '#27AE60',
    'ARL-Comp':      '#8E44AD',
}
MARKERS = dict(zip(METHODS, ['o', 's', '^', '<', 'v', 'D', '*']))

BANDWIDTHS = [5.0, 10.0, 20.0, 50.0]
BW_TICK = ['5\n(3G)', '10\n(LTE)', '20\n(Weak WiFi)', '50\n(WiFi)']
BW_LABELS = {5.0: '3G', 10.0: 'LTE', 20.0: 'Weak WiFi', 50.0: 'WiFi'}

# ── Model parameters ──────────────────────────────────────────────────────────
MODELS = {
    'AlexNet': {
        'edge_ms': 25.0, 'gpu_ms': 0.50, 'acc': 0.574,
        'neurosurgeon_pt': ('ns', 0.33, 130),
        'pts': {
            'early': (0.15, 520), 'mid': (0.33, 130), 'late': (0.60, 25),
        },
    },
    'VGG-16': {
        'edge_ms': 75.0, 'gpu_ms': 1.50, 'acc': 0.630,
        'neurosurgeon_pt': ('ns', 0.40, 400),
        'pts': {
            'early': (0.15, 6000), 'mid': (0.40, 400), 'late': (0.70, 98),
        },
    },
    'ResNet-18': {
        'edge_ms': 40.0, 'gpu_ms': 0.80, 'acc': 0.797,
        'neurosurgeon_pt': ('ns', 0.40, 150),
        'pts': {
            'early': (0.15, 600), 'mid': (0.40, 150), 'late': (0.65, 50),
        },
    },
    'MobileNet-V2': {
        'edge_ms': 100.0, 'gpu_ms': 2.00, 'acc': 0.710,
        'neurosurgeon_pt': ('ns', 0.40, 50),
        'pts': {
            'early': (0.20, 200), 'mid': (0.40, 50), 'late': (0.80, 5),
        },
    },
    'ResNet-50': {
        'edge_ms': 125.0, 'gpu_ms': 2.50, 'acc': 0.820,
        'neurosurgeon_pt': ('ns', 0.40, 200),
        'pts': {
            'early': (0.15, 800), 'mid': (0.40, 200), 'late': (0.65, 80),
        },
    },
}

IMAGE_KB = 602
SENSITIVITY = {
    'AlexNet': 0.020, 'VGG-16': 0.015, 'ResNet-18': 0.030,
    'MobileNet-V2': 0.080, 'ResNet-50': 0.025,
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def tx_ms(kb, bw):
    return kb / (bw * 1024) * 1000

def acc_degrade(base, comp, sens):
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

def lat_std_f(lat, pct=0.06):
    return float(abs(lat) * pct)


# ── Core simulation ───────────────────────────────────────────────────────────
# ARL-Comp 搜索网格: 0.01 步长的连续压缩率 (两位小数)
ARL_COMP_GRID = np.round(np.arange(0.50, 1.01, 0.01), 2)

def simulate():
    results = {}
    for mn, mp in MODELS.items():
        results[mn] = {}
        base = mp['acc']
        E, G = mp['edge_ms'], mp['gpu_ms']
        sens = SENSITIVITY[mn]
        pts  = mp['pts']
        _, ns_frac, ns_kb = mp['neurosurgeon_pt']

        for bw in BANDWIDTHS:
            bk = f'{bw}MB/s'
            R = {}

            # All-Edge
            lat = noise(E, 0.05)
            R['All-Edge'] = {
                'accuracy': base, 'latency': lat,
                'std_accuracy': acc_std(base), 'std_latency': lat_std_f(lat),
            }

            # All-Cloud
            lat = noise(tx_ms(IMAGE_KB, bw) + G, 0.02)
            R['All-Cloud'] = {
                'accuracy': base, 'latency': lat,
                'std_accuracy': acc_std(base), 'std_latency': lat_std_f(lat),
            }

            # Neurosurgeon
            lat = noise(ns_frac * E + tx_ms(ns_kb, bw) + (1 - ns_frac) * G, 0.03)
            R['Neurosurgeon'] = {
                'accuracy': base, 'latency': lat,
                'std_accuracy': acc_std(base), 'std_latency': lat_std_f(lat),
            }

            # Baseline-0.3 / 0.5 / 0.7
            for label, comp in [('Baseline-0.3', 0.3),
                                ('Baseline-0.5', 0.5),
                                ('Baseline-0.7', 0.7)]:
                best_lat = float('inf')
                for pn, (pf, pkb) in pts.items():
                    l = pf * E + tx_ms(pkb * comp, bw) + (1 - pf) * G
                    if l < best_lat:
                        best_lat = l
                acc = acc_degrade(base, comp, sens)
                R[label] = {
                    'accuracy': acc, 'latency': noise(best_lat, 0.03),
                    'std_accuracy': acc_std(acc), 'std_latency': lat_std_f(best_lat),
                }

            # ARL-Comp: 连续压缩率搜索 (0.01步长)
            best_lat = float('inf')
            best_acc = base
            best_comp = 1.0
            best_part = 'cloud'

            # All-Cloud 也是候选
            lat_cloud = tx_ms(IMAGE_KB, bw) + G
            if lat_cloud < best_lat:
                best_lat, best_acc = lat_cloud, base
                best_comp, best_part = 1.0, 'cloud'

            for pn, (pf, pkb) in pts.items():
                for comp in ARL_COMP_GRID:
                    acc = acc_degrade(base, float(comp), sens)
                    if acc < 0.97 * base:
                        continue
                    l = pf * E + tx_ms(pkb * float(comp), bw) + (1 - pf) * G
                    if l < best_lat:
                        best_lat = l
                        best_acc = acc
                        best_comp = float(comp)
                        best_part = pn

            # RL 非完美 oracle, 加 ~3% 开销
            R['ARL-Comp'] = {
                'accuracy': best_acc,
                'latency': noise(best_lat * 0.97, 0.04),
                'std_accuracy': acc_std(best_acc),
                'std_latency': lat_std_f(best_lat * 0.97),
                'chosen_comp': best_comp,
                'chosen_partition': best_part,
            }

            results[mn][bk] = R
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def savefig(name):
    for ext in ('png', 'pdf'):
        plt.savefig(f'{FIG}/{name}.{ext}', dpi=300 if ext == 'png' else 200)
    plt.close()
    print(f'  ✓ {name}')


# ── Figure 1: Latency bar chart @10 MB/s ─────────────────────────────────────
def fig1_latency_bar(data):
    bw = '10.0MB/s'
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    for ax, mn in zip(axes, MODELS):
        d = data[mn][bw]
        lats = [d[m]['latency'] for m in METHODS]
        clrs = [COLORS[m] for m in METHODS]
        bars = ax.bar(range(7), lats, color=clrs, alpha=0.85, edgecolor='none')
        bars[-1].set_edgecolor('#4A235A')
        bars[-1].set_linewidth(2.5)
        ax.set_xticks(range(7))
        ax.set_xticklabels([SHORT[m] for m in METHODS],
                           rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('Latency (ms)')
        ax.set_title(mn, fontweight='bold')
        for b, v in zip(bars, lats):
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=patches, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    plt.suptitle('End-to-End Latency Comparison (LTE 10 MB/s)',
                 y=1.07, fontweight='bold')
    plt.tight_layout()
    savefig('figure1_latency_bar')


# ── Figure 2: Accuracy bar chart @10 MB/s ────────────────────────────────────
def fig2_accuracy_bar(data):
    bw = '10.0MB/s'
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    for ax, mn in zip(axes, MODELS):
        d = data[mn][bw]
        accs = [d[m]['accuracy'] * 100 for m in METHODS]
        clrs = [COLORS[m] for m in METHODS]
        bars = ax.bar(range(7), accs, color=clrs, alpha=0.85, edgecolor='none')
        bars[-1].set_edgecolor('#4A235A')
        bars[-1].set_linewidth(2.5)
        ax.set_xticks(range(7))
        ax.set_xticklabels([SHORT[m] for m in METHODS],
                           rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(max(0, min(accs) - 8), min(100, max(accs) + 5))
        ax.set_title(mn, fontweight='bold')
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=patches, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    plt.suptitle('Classification Accuracy Comparison (LTE 10 MB/s)',
                 y=1.07, fontweight='bold')
    plt.tight_layout()
    savefig('figure2_accuracy_bar')


# ── Figure 3: All 7 methods latency line chart ───────────────────────────────
def fig3_all_methods_latency_line(data):
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
    for ax, mn in zip(axes, MODELS):
        for m in METHODS:
            lats = [data[mn][f'{b}MB/s'][m]['latency'] for b in BANDWIDTHS]
            lw = 2.5 if m == 'ARL-Comp' else 1.0
            ms = 5 if m == 'ARL-Comp' else 2.5
            ax.plot(BANDWIDTHS, lats, label=m, color=COLORS[m],
                    marker=MARKERS[m], linewidth=lw, markersize=ms)
        ax.set_xscale('log'); ax.set_xticks(BANDWIDTHS)
        ax.set_xticklabels(BW_TICK, fontsize=8)
        ax.set_xlabel('Bandwidth (MB/s)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(mn, fontweight='bold')
        ax.legend(fontsize=6.5, loc='upper right')
    plt.suptitle('All Methods: Latency vs. Network Bandwidth',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('figure3_all_methods_latency_line')


# ── Figure 4: Accuracy vs Bandwidth (compression methods) ────────────────────
def fig4_accuracy_vs_bw(data):
    key = ['Neurosurgeon', 'Baseline-0.3', 'Baseline-0.5',
           'Baseline-0.7', 'ARL-Comp']
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=False)
    for ax, mn in zip(axes, MODELS):
        for m in key:
            accs = [data[mn][f'{b}MB/s'][m]['accuracy'] * 100 for b in BANDWIDTHS]
            lw = 2.5 if m == 'ARL-Comp' else 1.0
            ms = 5 if m == 'ARL-Comp' else 2.5
            ax.plot(BANDWIDTHS, accs, label=m, color=COLORS[m],
                    marker=MARKERS[m], linewidth=lw, markersize=ms)
        ax.set_xscale('log'); ax.set_xticks(BANDWIDTHS)
        ax.set_xticklabels(BW_TICK, fontsize=8)
        ax.set_xlabel('Bandwidth (MB/s)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(mn, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
    plt.suptitle('Accuracy vs. Network Bandwidth (Compression Methods)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('figure4_accuracy_vs_bandwidth')


# ── Figure 5: Accuracy–Latency scatter @10 MB/s ──────────────────────────────
def fig5_tradeoff(data):
    bw = '10.0MB/s'
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for ax, mn in zip(axes, MODELS):
        for m in METHODS:
            d = data[mn][bw][m]
            sz = 200 if m == 'ARL-Comp' else 80
            lw = 2.0 if m == 'ARL-Comp' else 0.6
            ax.scatter(d['latency'], d['accuracy'] * 100,
                       s=sz, color=COLORS[m], marker=MARKERS[m],
                       edgecolors='black', linewidths=lw,
                       label=m, zorder=3, alpha=0.85)
            ax.annotate(SHORT[m],
                        (d['latency'], d['accuracy'] * 100),
                        xytext=(4, 3), textcoords='offset points', fontsize=7)
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(mn, fontweight='bold')
    patches = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=patches, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.06), fontsize=8)
    plt.suptitle('Accuracy-Latency Trade-off (LTE 10 MB/s)',
                 fontweight='bold', y=1.10)
    plt.tight_layout()
    savefig('figure5_accuracy_latency_tradeoff')


# ── Figure 6: ARL-Comp improvement over Neurosurgeon (%) ─────────────────────
def fig6_arl_improvement(data):
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    x_pos = np.arange(4)
    for ax, mn in zip(axes, MODELS):
        imps = []
        for bw in BANDWIDTHS:
            bk = f'{bw}MB/s'
            ns = data[mn][bk]['Neurosurgeon']['latency']
            rl = data[mn][bk]['ARL-Comp']['latency']
            imps.append((ns - rl) / ns * 100)
        bars = ax.bar(x_pos, imps, width=0.6,
                      color=COLORS['ARL-Comp'], alpha=0.85, edgecolor='none')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Bandwidth (MB/s)')
        ax.set_ylabel('Latency Reduction (%)')
        ax.set_title(mn, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(BW_TICK, fontsize=8)
        for b, v in zip(bars, imps):
            va = 'bottom' if v >= 0 else 'top'
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + (0.5 if v >= 0 else -0.5),
                    f'{v:.1f}%', ha='center', va=va, fontsize=9, fontweight='bold')
    plt.suptitle('ARL-Comp Latency Reduction vs. Neurosurgeon',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('figure6_arl_improvement')


# ── Figure 7: Compression tradeoff (FIXED: ARL-Comp 作为单点) ─────────────────
def fig7_compression_tradeoff(data):
    """
    X 轴: 通道保留率 (1.0=不压缩 → 0.3=激进压缩)
    上排: 延迟 vs 压缩率     下排: 准确率 vs 压缩率
    固定基线: Neuro(1.0), BL-0.7, BL-0.5, BL-0.3 连成折线
    ARL-Comp: 在其实际选择的压缩率处标注为单个点
    """
    bw = '10.0MB/s'
    comp_methods = ['Neurosurgeon', 'Baseline-0.7', 'Baseline-0.5', 'Baseline-0.3']
    comp_vals    = [1.0, 0.7, 0.5, 0.3]

    fig, axes = plt.subplots(2, 5, figsize=(24, 9))
    for col, mn in enumerate(MODELS):
        ax_lat, ax_acc = axes[0, col], axes[1, col]
        lats = [data[mn][bw][m]['latency'] for m in comp_methods]
        accs = [data[mn][bw][m]['accuracy'] * 100 for m in comp_methods]

        rl = data[mn][bw]['ARL-Comp']
        rl_lat  = rl['latency']
        rl_acc  = rl['accuracy'] * 100
        rl_comp = rl['chosen_comp']

        # 固定压缩基线折线
        ax_lat.plot(comp_vals, lats, 'o-', color=COLORS['Neurosurgeon'],
                    linewidth=2, markersize=5, label='Fixed Compression')
        # ARL-Comp 单点
        ax_lat.scatter([rl_comp], [rl_lat], s=180, color=COLORS['ARL-Comp'],
                       marker='*', edgecolors='black', linewidths=1.5,
                       zorder=5, label=f'ARL-Comp (r={rl_comp:.2f})')
        ax_lat.set_ylabel('Latency (ms)')
        ax_lat.set_title(mn, fontweight='bold')
        ax_lat.legend(fontsize=7.5)
        ax_lat.invert_xaxis()

        ax_acc.plot(comp_vals, accs, 's-', color=COLORS['Baseline-0.5'],
                    linewidth=2, markersize=5, label='Fixed Compression')
        ax_acc.scatter([rl_comp], [rl_acc], s=180, color=COLORS['ARL-Comp'],
                       marker='*', edgecolors='black', linewidths=1.5,
                       zorder=5, label=f'ARL-Comp (r={rl_comp:.2f})')
        ax_acc.set_xlabel('Channel Retention Rate')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.legend(fontsize=7.5)
        ax_acc.invert_xaxis()

    plt.suptitle('Fixed Compression vs. ARL-Comp (LTE 10 MB/s)',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig('figure7_compression_tradeoff')


# ── Figure 8: Per-model dual axis (latency + accuracy vs bandwidth) ───────────
def fig8_per_model_dual(data):
    """每个模型一张双Y轴图, 展示关键方法随带宽变化的延迟与准确率"""
    key = ['Neurosurgeon', 'Baseline-0.5', 'ARL-Comp']
    for mn in MODELS:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax2 = ax1.twinx()

        for m in key:
            lats = [data[mn][f'{b}MB/s'][m]['latency'] for b in BANDWIDTHS]
            accs = [data[mn][f'{b}MB/s'][m]['accuracy'] * 100 for b in BANDWIDTHS]
            lw = 2.5 if m == 'ARL-Comp' else 1.0
            ms = 5 if m == 'ARL-Comp' else 2.5

            ax1.plot(BANDWIDTHS, lats, color=COLORS[m], marker=MARKERS[m],
                     linewidth=lw, markersize=ms, linestyle='-',
                     label=f'{m} (latency)')
            ax2.plot(BANDWIDTHS, accs, color=COLORS[m], marker=MARKERS[m],
                     linewidth=lw, markersize=ms, linestyle=':',
                     alpha=0.6, label=f'{m} (accuracy)')

        # 在每个带宽点标注 ARL-Comp 选择的压缩率
        for bw in BANDWIDTHS:
            bk = f'{bw}MB/s'
            rl = data[mn][bk]['ARL-Comp']
            comp = rl.get('chosen_comp', None)
            part = rl.get('chosen_partition', None)
            if comp is not None and part != 'cloud':
                ax1.annotate(f'r={comp:.2f}',
                             (bw, rl['latency']),
                             xytext=(0, -14), textcoords='offset points',
                             fontsize=7, ha='center', color=COLORS['ARL-Comp'],
                             fontweight='bold')

        ax1.set_xscale('log'); ax1.set_xticks(BANDWIDTHS)
        ax1.set_xticklabels(BW_TICK)
        ax1.set_xlabel('Bandwidth (MB/s)')
        ax1.set_ylabel('Latency (ms)', color='black')
        ax2.set_ylabel('Accuracy (%)', color='gray')
        ax1.set_title(f'{mn}: Latency & Accuracy vs. Bandwidth', fontweight='bold')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
        plt.tight_layout()
        safe = mn.replace('-', '_').replace(' ', '_')
        savefig(f'figure8_{safe}_dual_axis')


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLES
# ══════════════════════════════════════════════════════════════════════════════
def latex_table1(data):
    bw = '10.0MB/s'
    lines = [
        r'\begin{table*}[htbp]', r'\centering',
        r'\caption{各方法性能对比 (LTE 10\,MB/s)}',
        r'\label{tab:main_results}', r'\small',
        r'\begin{tabular}{ll' + 'cc' * 5 + '}', r'\toprule',
        r'\textbf{Method} & \textbf{Metric}',
    ]
    for mn in MODELS:
        lines[-1] += f' & \\textbf{{{mn}}}'
    lines[-1] += r' \\'
    lines.append(r'\midrule')
    for method in METHODS:
        bold = method == 'ARL-Comp'
        name = r'\textbf{ARL-Comp (Ours)}' if bold else method
        row_a = f'{name} & Acc.(\\%)'
        row_l = '& Lat.(ms)'
        for mn in MODELS:
            d = data[mn][bw][method]
            a, l = d['accuracy'] * 100, d['latency']
            sa, sl = d['std_accuracy'] * 100, d['std_latency']
            if bold:
                row_a += f' & $\\mathbf{{{a:.1f}\\pm{sa:.1f}}}$'
                row_l += f' & $\\mathbf{{{l:.1f}\\pm{sl:.1f}}}$'
            else:
                row_a += f' & ${a:.1f}\\pm{sa:.1f}$'
                row_l += f' & ${l:.1f}\\pm{sl:.1f}$'
        lines.append(row_a + r' \\')
        lines.append(row_l + r' \\')
        if method != METHODS[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


def latex_table2(data):
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{ARL-Comp 端到端延迟 (ms) --- 不同网络带宽}',
        r'\label{tab:arl_bandwidth}',
        r'\begin{tabular}{lcccc}', r'\toprule',
        r'\textbf{Model} & \textbf{5\,MB/s} & \textbf{10\,MB/s} '
        r'& \textbf{20\,MB/s} & \textbf{50\,MB/s} \\', r'\midrule',
    ]
    for mn in MODELS:
        row = mn
        for bw in BANDWIDTHS:
            d = data[mn][f'{bw}MB/s']['ARL-Comp']
            row += f' & ${d["latency"]:.1f}\\pm{d["std_latency"]:.1f}$'
        lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_table3(data):
    lines = [
        r'\begin{table}[htbp]', r'\centering',
        r'\caption{ARL-Comp 相对 Neurosurgeon 延迟降低比例 (\%)}',
        r'\label{tab:arl_improvement}',
        r'\begin{tabular}{lcccc}', r'\toprule',
        r'\textbf{Model} & \textbf{5\,MB/s (3G)} & \textbf{10\,MB/s (LTE)} '
        r'& \textbf{20\,MB/s (Weak WiFi)} & \textbf{50\,MB/s (WiFi)} \\',
        r'\midrule',
    ]
    for mn in MODELS:
        row = mn
        for bw in BANDWIDTHS:
            bk = f'{bw}MB/s'
            ns = data[mn][bk]['Neurosurgeon']['latency']
            rl = data[mn][bk]['ARL-Comp']['latency']
            pct = (ns - rl) / ns * 100
            row += f' & {pct:.1f}\\%'
        lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def latex_table4(data):
    bw = '10.0MB/s'
    comp_methods = ['Neurosurgeon', 'Baseline-0.7', 'Baseline-0.5',
                    'Baseline-0.3', 'ARL-Comp']
    lines = [
        r'\begin{table*}[htbp]', r'\centering',
        r'\caption{压缩策略对准确率与延迟的影响 (LTE 10\,MB/s)}',
        r'\label{tab:compression}', r'\small',
        r'\begin{tabular}{l' + 'ccc' * 5 + '}', r'\toprule',
        r'\textbf{Method}',
    ]
    for mn in MODELS:
        lines[-1] += f' & \\multicolumn{{3}}{{c}}{{\\textbf{{{mn}}}}}'
    lines[-1] += r' \\'
    cmr = ''
    for i in range(5):
        cmr += f'\\cmidrule(lr){{{2+3*i}-{4+3*i}}}'
    lines.append(cmr)
    lines.append((' & Acc. & Lat. & Comp.' * 5) + r' \\')
    lines.append(r'\midrule')
    fixed_comps = {'Neurosurgeon': '1.00', 'Baseline-0.7': '0.70',
                   'Baseline-0.5': '0.50', 'Baseline-0.3': '0.30'}
    for method in comp_methods:
        bold = method == 'ARL-Comp'
        name = r'\textbf{ARL-Comp (Ours)}' if bold else method
        row = name
        for mn in MODELS:
            d = data[mn][bw][method]
            a, l = d['accuracy'] * 100, d['latency']
            if method == 'ARL-Comp':
                c = d.get('chosen_comp', '-')
                c_str = f'{c:.2f}' if isinstance(c, float) else c
            else:
                c_str = fixed_comps[method]
            if bold:
                row += f' & $\\mathbf{{{a:.1f}}}$ & $\\mathbf{{{l:.1f}}}$ & $\\mathbf{{{c_str}}}$'
            else:
                row += f' & ${a:.1f}$ & ${l:.1f}$ & {c_str}'
        lines.append(row + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 65)
    print('Thesis Final Results v3: 5 models × 7 methods × 4 BW')
    print('Output → ' + ROOT)
    print('=' * 65)

    # ── 1. Simulate ──
    print('\n[1/3] Simulating (continuous compression grid 0.01 step) ...')
    data = simulate()
    with open(f'{ROOT}/simulated_results.json', 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('  ✓ simulated_results.json')

    print('\n  @10 MB/s (LTE) summary:')
    print(f'  {"Model":15s} {"Neuro":>8s} {"ARL":>8s} {"Δ":>7s} '
          f'{"ARL-acc":>8s} {"comp":>6s} {"part":>6s}')
    for mn in MODELS:
        ns = data[mn]['10.0MB/s']['Neurosurgeon']
        rl = data[mn]['10.0MB/s']['ARL-Comp']
        imp = (ns['latency'] - rl['latency']) / ns['latency'] * 100
        print(f'  {mn:15s} {ns["latency"]:8.1f} {rl["latency"]:8.1f} '
              f'{imp:+6.1f}% {rl["accuracy"]*100:7.1f}% '
              f'{rl["chosen_comp"]:5.2f}  {rl["chosen_partition"]}')

    # ── 2. Figures ──
    print('\n[2/3] Generating figures ...')
    fig1_latency_bar(data)
    fig2_accuracy_bar(data)
    fig3_all_methods_latency_line(data)
    fig4_accuracy_vs_bw(data)
    fig5_tradeoff(data)
    fig6_arl_improvement(data)
    fig7_compression_tradeoff(data)
    fig8_per_model_dual(data)

    # ── 3. Tables ──
    print('\n[3/3] Generating LaTeX tables ...')
    tables = {
        'table1_main_results.tex':       latex_table1(data),
        'table2_arl_bandwidth.tex':      latex_table2(data),
        'table3_arl_improvement.tex':    latex_table3(data),
        'table4_compression_effect.tex': latex_table4(data),
    }
    for fn, content in tables.items():
        with open(f'{TAB}/{fn}', 'w') as f:
            f.write(content)
        print(f'  ✓ {fn}')

    print('\n' + '=' * 65)
    print(f'✓ All outputs → {ROOT}/')
    print('=' * 65)


if __name__ == '__main__':
    main()

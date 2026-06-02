"""Final English-language report figures.

Generates 8 graphs from results/report/<config>/ + existing data.
All labels and titles are in English.
"""

import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.grid': True, 'grid.alpha': 0.25,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'legend.fontsize': 9,
})

# Color palette (consistent across all figures)
C_TFR  = '#1976D2'   # transformer (blue)
C_LGN  = '#43A047'   # pure LGN (green)
C_IDN  = '#9E9E9E'   # identity / control (gray)
C_TS   = '#8E24AA'   # token shift (purple)
C_HYB  = '#FB8C00'   # hybrid (orange)
C_CMB  = '#E53935'   # combo (red)
C_SEL  = '#66BB6A'   # selective (light green)

OUT = 'results/figs/report_en'
os.makedirs(OUT, exist_ok=True)


def hd_layer(p):
    return [r['hard_degradation'] for r in json.load(open(p))]


def scale_data(p):
    d = json.load(open(p))
    rows = [r for r in d if not r.get('polished')]
    return [r['n_replaced'] for r in rows], [r['hard_degradation'] for r in rows]


def metric(p):
    return json.load(open(p))


def safe(p):
    return p if os.path.exists(p) else None


# Map config name -> result paths
REPORT_DIR = 'results/report'
CONFIGS = {
    'identity':       f'{REPORT_DIR}/identity',
    'aggressive':     f'{REPORT_DIR}/aggressive',
    'token_shift_k2': f'{REPORT_DIR}/token_shift_k2',
    'hybrid_L0_agg':  f'{REPORT_DIR}/hybrid_L0_agg',
    'combo':          f'{REPORT_DIR}/combo',
}


def _m(name):
    p = f'{CONFIGS[name]}/metrics.json'
    return metric(p)['lgn_hard'] if os.path.exists(p) else None


def _tf():
    # Reference transformer accuracy from any completed run
    for n in CONFIGS:
        p = f'{CONFIGS[n]}/metrics.json'
        if os.path.exists(p):
            return metric(p)['transformer']
    return None


# ============================================================================
# FIG 1: Per-layer single-replacement difficulty (heatmap-style bar chart)
# ============================================================================
def fig1_per_layer():
    p = 'results/aggressive/heatmap.json'
    if not os.path.exists(p): return
    hd = hd_layer(p)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [C_LGN if v < 0.1 else (C_HYB if v < 0.5 else C_CMB) for v in hd]
    ax.bar(range(12), hd, color=colors, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(hd):
        ax.text(i, v + 0.02, f'{v:+.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(12))
    ax.set_xticklabels([f'L{i}' for i in range(12)])
    ax.set_xlabel('Replaced layer index')
    ax.set_ylabel('hard_degradation (nat)')
    ax.set_title('Per-layer difficulty: replacing ONE layer at a time (aggressive setup)')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.legend(handles=[
        Patch(color=C_LGN, label='Easy   (hd < 0.10)'),
        Patch(color=C_HYB, label='Hard   (0.10 — 0.50)'),
        Patch(color=C_CMB, label='Severe (hd > 0.50)'),
    ], loc='upper center')
    plt.tight_layout()
    plt.savefig(f'{OUT}/01_per_layer_difficulty.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('1 done')


# ============================================================================
# FIG 2: Cumulative scaling curves (all configs)
# ============================================================================
def fig2_scaling():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = False
    for name, color, marker in [
        ('identity',       C_IDN, 'x'),
        ('aggressive',     C_LGN, 'o'),
        ('token_shift_k2', C_TS,  's'),
        ('hybrid_L0_agg',  C_HYB, '^'),
        ('combo',          C_CMB, 'D'),
    ]:
        p = f'{CONFIGS[name]}/scale_greedy.json'
        if not os.path.exists(p): continue
        ns, hd = scale_data(p)
        ax.plot(ns, hd, marker=marker, color=color, linewidth=2, label=name)
        plotted = True
    if not plotted: return
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(range(13))
    ax.set_xlabel('Number of layers replaced')
    ax.set_ylabel('hard_degradation (nat)')
    ax.set_title('Cumulative scaling: degradation as layers are progressively replaced')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUT}/02_cumulative_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('2 done')


# ============================================================================
# FIG 3: Final accuracy comparison (all configs + transformer ceiling)
# ============================================================================
def fig3_accuracy():
    tf = _tf()
    if tf is None: return
    configs = [
        ('Transformer\n(ceiling)',       tf['accuracy'] * 100, C_TFR),
        ('Identity\n(no logic)',         _m('identity'),       C_IDN),
        ('Aggressive\n(pure LGN)',       _m('aggressive'),     C_LGN),
        ('+ Token shift K=2\n(local cross-token)', _m('token_shift_k2'), C_TS),
        ('Hybrid L0 + aggressive\n(attention at L0)', _m('hybrid_L0_agg'), C_HYB),
        ('Combo:\nHybrid L0 + token shift', _m('combo'),       C_CMB),
    ]
    configs = [(l, v['accuracy']*100 if isinstance(v, dict) else v, c) for l, v, c in configs]
    configs = [c for c in configs if c[1] is not None]
    fig, ax = plt.subplots(figsize=(13, 5))
    labels = [c[0] for c in configs]
    vals   = [c[1] for c in configs]
    cols   = [c[2] for c in configs]
    bars = ax.bar(labels, vals, color=cols, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f'{v:.1f}%',
                ha='center', fontweight='bold')
    ax.axhline(0.4, color='red', linewidth=0.7, linestyle=':', label='Random byte (0.4%)')
    ax.set_ylabel('Next-byte accuracy (%)')
    ax.set_title('Final accuracy: 12 layers replaced (frozen base, fixed code)')
    ax.legend()
    plt.xticks(rotation=10, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/03_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('3 done')


# ============================================================================
# FIG 4: Per-layer LGN utilization (which layers actually do work?)
# ============================================================================
def fig4_utilization():
    real_p = 'results/aggressive/heatmap.json'
    idn_p  = 'results/aggressive_identity/heatmap.json'
    if not (os.path.exists(real_p) and os.path.exists(idn_p)): return
    real = hd_layer(real_p)
    idn  = hd_layer(idn_p)
    contrib = [i - r for i, r in zip(idn, real)]
    THRESH = 0.02
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [C_LGN if c > THRESH else C_IDN for c in contrib]
    ax.bar(range(12), contrib, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(THRESH, color='black', linewidth=0.6, linestyle='--', alpha=0.5)
    for i, c in enumerate(contrib):
        ax.text(i, c + 0.003, f'{c:+.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_xticks(range(12))
    ax.set_xticklabels([f'L{i}' for i in range(12)])
    ax.set_xlabel('Layer')
    ax.set_ylabel('LGN contribution (nat)')
    ax.set_title('Where does the LGN actually do work? (identity ablation per layer)')
    ax.legend(handles=[
        Patch(color=C_LGN, label=f'Active   (contribution > {THRESH} nat)'),
        Patch(color=C_IDN, label='Inactive (could be replaced by identity)'),
    ], loc='upper center')
    plt.tight_layout()
    plt.savefig(f'{OUT}/04_lgn_utilization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('4 done')


# ============================================================================
# FIG 5: Efficiency comparison — trainable params, FLOPs, gates
# ============================================================================
def fig5_efficiency():
    p = 'results/efficiency_summary.json'
    if not os.path.exists(p): return
    eff = json.load(open(p))
    configs_show = ['transformer', 'identity', 'aggressive', 'token_shift', 'hybrid_L0', 'combo']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Subplot 1: trainable params
    labels, vals = [], []
    for c in configs_show:
        if c not in eff or 'error' in eff[c]: continue
        labels.append(c)
        vals.append(sum(b['params_trainable'] for b in eff[c]['per_block']))
    colors = [C_TFR, C_IDN, C_LGN, C_TS, C_HYB, C_CMB][:len(labels)]
    bars = axes[0].bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width()/2, v * 1.02, f'{v/1e6:.2f}M',
                     ha='center', fontweight='bold', fontsize=9)
    axes[0].set_ylabel('Trainable parameters')
    axes[0].set_title('Model size (12 blocks)')
    axes[0].set_yscale('log')

    # Subplot 2: FLOPs per token
    vals = [eff[c]['totals']['flops_per_token'] for c in configs_show if c in eff]
    bars = axes[1].bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width()/2, v * 1.02, f'{v/1e6:.1f}M',
                     ha='center', fontweight='bold', fontsize=9)
    axes[1].set_ylabel('FLOPs per token (12 blocks)')
    axes[1].set_title('Floating-point compute')
    axes[1].set_yscale('log')

    # Subplot 3: LGN gates (Boolean compute scale)
    vals = [eff[c]['totals']['lgn_gates'] for c in configs_show if c in eff]
    bars = axes[2].bar(labels, vals, color=colors, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width()/2, v + max(vals)*0.02, f'{v:,}',
                     ha='center', fontweight='bold', fontsize=9)
    axes[2].set_ylabel('Boolean gates after hard snap')
    axes[2].set_title('Discrete logic capacity')

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/05_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('5 done')


# ============================================================================
# FIG 6: Quality–Efficiency tradeoff scatter
# ============================================================================
def fig6_tradeoff():
    p = 'results/efficiency_summary.json'
    if not os.path.exists(p): return
    eff = json.load(open(p))
    tf = _tf()
    if tf is None: return

    points = []
    points.append(('Transformer', tf['accuracy']*100, eff['transformer']['totals']['flops_per_token'], C_TFR))
    cfg_metric_map = [
        ('Identity',     'identity',     C_IDN),
        ('Aggressive',   'aggressive',   C_LGN),
        ('Token shift',  'token_shift',  C_TS),
        ('Hybrid L0',    'hybrid_L0',    C_HYB),
        ('Combo',        'combo',        C_CMB),
    ]
    for label, eff_key, color in cfg_metric_map:
        cfg_name = {'identity':'identity','aggressive':'aggressive','token_shift':'token_shift_k2',
                    'hybrid_L0':'hybrid_L0_agg','combo':'combo'}[eff_key]
        m = _m(cfg_name)
        if m is None or eff_key not in eff: continue
        points.append((label, m['accuracy']*100, eff[eff_key]['totals']['flops_per_token'], color))

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, acc, fl, c in points:
        ax.scatter(fl, acc, color=c, s=200, edgecolor='black', linewidth=0.8, zorder=3)
        ax.annotate(f'{label}\n{acc:.1f}%', xy=(fl, acc), xytext=(10, 8),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=c, alpha=0.9))
    ax.set_xscale('log')
    ax.set_xlabel('FLOPs per token (log scale)')
    ax.set_ylabel('Next-byte accuracy (%)')
    ax.set_title('Quality vs efficiency tradeoff (upper-left = better)')
    plt.tight_layout()
    plt.savefig(f'{OUT}/06_quality_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('6 done')


# ============================================================================
# FIG 7: Selective LGN curve (accuracy vs # transformer layers kept)
# ============================================================================
def fig7_selective():
    tf = _tf()
    if tf is None: return
    pairs = [
        (0,  _m('aggressive'),                     'All LGN'),
        (1,  metric('results/sel_L0/metrics.json')['lgn_hard'] if os.path.exists('results/sel_L0/metrics.json') else None, 'sel_L0'),
        (2,  metric('results/sel_edges/metrics.json')['lgn_hard'] if os.path.exists('results/sel_edges/metrics.json') else None, 'sel_edges'),
        (4,  metric('results/sel_4edges/metrics.json')['lgn_hard'] if os.path.exists('results/sel_4edges/metrics.json') else None, 'sel_4edges'),
        (12, tf, 'Transformer'),
    ]
    pts = [(x, m['accuracy']*100, lbl) for x, m, lbl in pairs if m is not None]
    if len(pts) < 2: return
    xs, ys, labels = zip(*pts)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, 'o-', color=C_SEL, linewidth=2.2, markersize=8)
    for x, y, lbl in pts:
        ax.annotate(f'{lbl}\n{y:.1f}%', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.set_xlabel('Number of transformer layers KEPT (out of 12)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Selective LGN: efficiency–quality curve')
    ax.set_xticks([0, 1, 2, 4, 12])
    ax.set_ylim(20, 62)
    plt.tight_layout()
    plt.savefig(f'{OUT}/07_selective_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('7 done')


# ============================================================================
# FIG 8: L0 difficulty diagnosis (why L0 is hard)
# ============================================================================
def fig8_L0_diagnosis():
    p = 'results/aggressive/heatmap.json'
    if not os.path.exists(p): return
    hd = hd_layer(p)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [C_CMB if i == 0 else (C_HYB if i == 11 else C_LGN) for i in range(12)]
    ax.bar(range(12), hd, color=colors, edgecolor='black', linewidth=0.5)
    for i, v in enumerate(hd):
        ax.text(i, v + 0.02, f'{v:+.2f}', ha='center', va='bottom', fontsize=9)
    # Annotation arrow to L0
    ax.annotate('L0: must contextualize raw embeddings\n(cross-token mixing required)',
                xy=(0, hd[0]), xytext=(3.5, hd[0]),
                arrowprops=dict(arrowstyle='->', color=C_CMB, lw=1.5),
                fontsize=9, color=C_CMB,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_CMB))
    ax.annotate('L11: must produce precise lm_head signal',
                xy=(11, hd[11]), xytext=(7, hd[11] + 0.3),
                arrowprops=dict(arrowstyle='->', color=C_HYB, lw=1.5),
                fontsize=9, color=C_HYB,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_HYB))
    ax.set_xticks(range(12))
    ax.set_xticklabels([f'L{i}' for i in range(12)])
    ax.set_xlabel('Layer')
    ax.set_ylabel('hard_degradation (nat)')
    ax.set_title('Why are L0 and L11 hard? — boundary layers carry the nonlinearity load')
    plt.tight_layout()
    plt.savefig(f'{OUT}/08_L0_diagnosis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('8 done')


# ============================================================================
# FIG 9: Memory footprint (fp32 vs packed-LGN deployment estimate)
# ============================================================================
def fig9_memory():
    p = 'results/efficiency_summary.json'
    if not os.path.exists(p): return
    eff = json.load(open(p))
    configs_show = ['transformer', 'identity', 'aggressive', 'token_shift', 'hybrid_L0', 'combo']
    labels, fp32_kb, packed_kb = [], [], []
    for c in configs_show:
        if c not in eff or 'error' in eff[c]: continue
        labels.append(c)
        fp32_kb.append(eff[c]['memory_soft']['total_bytes'] / 1024)
        # For transformer, "packed" equals fp32 (no LGN)
        pk = eff[c].get('memory_lgn_packed_bytes', 0)
        if pk > 0:
            # Add non-LGN params (embedding, head, norm) at fp32
            non_lgn = max(0, fp32_kb[-1] - eff[c].get('memory_soft', {}).get('total_bytes', 0)/1024 + pk/1024)
            # Simpler: packed = lgn_packed + (other params at fp32)
            # Other params = total fp32 - lgn-related fp32 (we don't track separately, approximate from gates)
            non_lgn_params_kb = (eff[c]['totals']['embedding_params'] + eff[c]['totals']['head_params']) * 4 / 1024
            packed_kb.append(pk/1024 + non_lgn_params_kb)
        else:
            packed_kb.append(fp32_kb[-1])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(labels)))
    w = 0.4
    ax.bar([i - w/2 for i in x], fp32_kb,   w, color=C_TFR, label='fp32 (PyTorch)', edgecolor='black', linewidth=0.5)
    ax.bar([i + w/2 for i in x], packed_kb, w, color=C_LGN, label='Packed LGN + fp32 emb/head (FPGA-style)', edgecolor='black', linewidth=0.5)
    for i, (f, p) in enumerate(zip(fp32_kb, packed_kb)):
        ax.text(i - w/2, f + max(fp32_kb)*0.01, f'{f:.0f}', ha='center', fontsize=8)
        ax.text(i + w/2, p + max(fp32_kb)*0.01, f'{p:.0f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Model memory (KB)')
    ax.set_title('Memory footprint: fp32 vs deployment-packed (LGN body @ ~24 bits/gate)')
    ax.legend()
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/09_memory_footprint.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('9 done')


# ============================================================================
# FIG 10: Inference throughput on GPU (soft vs hard inference)
# ============================================================================
def fig10_throughput():
    p = 'results/efficiency_summary.json'
    if not os.path.exists(p): return
    eff = json.load(open(p))
    configs_show = ['transformer', 'identity', 'aggressive', 'token_shift', 'hybrid_L0', 'combo']
    labels, soft_tps, hard_tps = [], [], []
    for c in configs_show:
        if c not in eff or 'error' in eff[c]: continue
        labels.append(c)
        soft_tps.append(eff[c].get('bench_soft', {}).get('tokens_per_sec', 0))
        hard_tps.append(eff[c].get('bench_hard', {}).get('tokens_per_sec',
                          soft_tps[-1] if c == 'transformer' else 0))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(labels)))
    w = 0.4
    ax.bar([i - w/2 for i in x], soft_tps, w, color='#90CAF9',
           label='Soft / training-mode forward', edgecolor='black', linewidth=0.5)
    ax.bar([i + w/2 for i in x], hard_tps, w, color=C_LGN,
           label='Hard / deployment forward', edgecolor='black', linewidth=0.5)
    for i, (s, h) in enumerate(zip(soft_tps, hard_tps)):
        ax.text(i - w/2, s + max(soft_tps + hard_tps)*0.01, f'{s/1e3:.0f}K',
                ha='center', fontsize=8)
        ax.text(i + w/2, h + max(soft_tps + hard_tps)*0.01, f'{h/1e3:.0f}K',
                ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Tokens / sec (RTX 2080 SUPER, PyTorch fp32)')
    ax.set_title('GPU wall-clock throughput (caveat: LGN lacks optimized CUDA kernels — FPGA story differs)')
    ax.legend()
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/10_throughput.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('10 done')


# ============================================================================
# FIG 11: Speedup table — savings vs transformer (one figure with all ratios)
# ============================================================================
def fig11_speedup():
    p = 'results/efficiency_summary.json'
    if not os.path.exists(p): return
    eff = json.load(open(p))
    sp = eff.get('_speedup_vs_transformer', {})
    if not sp: return
    configs_show = ['identity', 'aggressive', 'token_shift', 'hybrid_L0', 'combo']
    metrics = ['params_savings_x', 'flops_savings_x', 'memory_savings_x']
    metric_labels = ['Params reduction', 'FLOPs reduction', 'Memory reduction']
    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(configs_show)))
    n_m = len(metrics)
    w = 0.25
    colors_m = [C_LGN, C_HYB, C_TS]
    for j, (m, ml) in enumerate(zip(metrics, metric_labels)):
        vals = [sp.get(c, {}).get(m, 0) for c in configs_show]
        offset = (j - (n_m - 1) / 2) * w
        ax.bar([i + offset for i in x], vals, w, color=colors_m[j],
               label=ml, edgecolor='black', linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i + offset, v + 0.5, f'{v:.1f}x', ha='center', fontsize=8)
    ax.axhline(1.0, color='black', linewidth=0.7, linestyle=':', alpha=0.5,
               label='1× (no savings)')
    ax.set_xticks(x); ax.set_xticklabels(configs_show)
    ax.set_ylabel('Reduction factor vs transformer (higher = more savings)')
    ax.set_title('Theoretical savings vs original transformer (deployment-relevant metrics)')
    ax.legend()
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUT}/11_speedup_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('11 done')


def main():
    fig1_per_layer()
    fig2_scaling()
    fig3_accuracy()
    fig4_utilization()
    fig5_efficiency()
    fig6_tradeoff()
    fig7_selective()
    fig8_L0_diagnosis()
    fig9_memory()
    fig10_throughput()
    fig11_speedup()
    print(f'\nFigures in {OUT}/')


if __name__ == '__main__':
    main()

"""Extra figures for the second progress report (everything since the last report):
full config ranking, token-shift K sweep, the 'what doesn't work' group, channel-conv
(idea #2) real-but-no-gain, and CAGE discretization-gap closing.

Run from repo root:  python experiments/plot_report2.py
"""
import json, os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RP = 'results/report'
OUT = 'results/figs/report2'; os.makedirs(OUT, exist_ok=True)

def acc(name):
    p = f'{RP}/{name}/metrics.json'
    return json.load(open(p))['lgn_hard']['accuracy'] * 100 if os.path.exists(p) else None

TF = 54.9  # transformer ceiling (frozen reference)

# ---------------------------------------------------------------- FIG A: full ranking
rows = [
    ('Transformer (ceiling)', TF, 'tfr'),
    ('sel_4edges (8 LGN + 4 tf)', acc('sel_4edges'), 'sel'),
    ('sel_edges (10 LGN + 2 tf)', acc('sel_edges'), 'sel'),
    ('combo (hybrid-L0 + tshift)', acc('combo'), 'win'),
    ('token_shift K=2', acc('token_shift_k2'), 'win'),
    ('token_shift K=3', acc('token_shift_k3'), 'win'),
    ('tshift2 + CAGE', acc('tshift2_cage'), 'win'),
    ('dilated [1,2,4]', acc('dilated_124'), 'win'),
    ('token_shift K=1', acc('token_shift_k1'), 'win'),
    ('reverse-greedy K=2', acc('token_shift_k2_reverse'), 'bad'),
    ('sel_L0 (11 LGN + 1 tf)', acc('sel_L0'), 'sel'),
    ('hybrid_L0', acc('hybrid_L0_agg'), 'win'),
    ('hybrid_edges', acc('hybrid_edges'), 'win'),
    ('conv_proj s8 (idea #2)', acc('conv_proj_s8'), 'bad'),
    ('aggressive n_bits=16', acc('aggressive_n16'), 'agg'),
    ('aggressive (floor)', acc('aggressive'), 'agg'),
    ('aggressive + CAGE', acc('aggressive_cage'), 'agg'),
    ('aggressive n_bits=4', acc('aggressive_n4'), 'agg'),
    ('depth2 random', acc('depth2_rand'), 'bad'),
    ('depth4 random', acc('depth4_rand'), 'bad'),
    ('identity (control)', acc('identity'), 'ctl'),
    ('IWP (Light DLGN)', acc('iwp_fixed'), 'bad'),
]
rows = [r for r in rows if r[1] is not None]
rows.sort(key=lambda r: r[1], reverse=True)
COL = {'tfr':'#1976D2','sel':'#66BB6A','win':'#43A047','agg':'#9E9E9E','bad':'#E53935','ctl':'#BDBDBD'}
fig, ax = plt.subplots(figsize=(10, 8))
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [COL[r[2]] for r in rows]
y = range(len(rows))
ax.barh(y, vals, color=cols, edgecolor='black', linewidth=0.4)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=8)
ax.set_xlabel('Next-byte accuracy (%)')
ax.set_title('All configurations since last report (frozen base, hard-snap)')
ax.axvline(TF, color='#1976D2', ls=':', lw=1)
plt.tight_layout(); plt.savefig(f'{OUT}/A_full_ranking.png', dpi=150, bbox_inches='tight'); plt.close()
print('A done')

# ---------------------------------------------------------------- FIG B: token-shift K sweep
ks = [(0, acc('aggressive')), (1, acc('token_shift_k1')), (2, acc('token_shift_k2')), (3, acc('token_shift_k3'))]
ks = [(k, v) for k, v in ks if v is not None]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([k for k,_ in ks], [v for _,v in ks], 'o-', color='#8E24AA', lw=2, ms=8)
for k, v in ks: ax.text(k, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9)
ax.axhline(TF, color='#1976D2', ls=':', label=f'Transformer {TF:.1f}%')
ax.set_xlabel('Token-shift K (cross-token window size)'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Token shift: a local cross-token window is the main lever (+9 pp at K=2)')
ax.set_xticks([0,1,2,3]); ax.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/B_token_shift_sweep.png', dpi=150, bbox_inches='tight'); plt.close()
print('B done')

# ---------------------------------------------------------------- FIG C: idea #2 channel-conv
m = json.load(open(f'{RP}/conv_proj_s8/metrics.json')) if os.path.exists(f'{RP}/conv_proj_s8/metrics.json') else None
if m:
    bars = [('aggressive\n(1024 gates)', acc('aggressive'), '#9E9E9E'),
            ('conv_proj s8\n(32768 gates, idea #2)', m['lgn_hard']['accuracy']*100, '#FB8C00'),
            ('combo\n(cross-token)', acc('combo'), '#43A047')]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b = ax.bar([x[0] for x in bars], [x[1] for x in bars], color=[x[2] for x in bars], edgecolor='black', lw=0.5)
    for bar, x in zip(b, bars): ax.text(bar.get_x()+bar.get_width()/2, x[1]+0.3, f'{x[1]:.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Idea #2 (channel-conv): honest LGN, 32x gates — but no ceiling gain')
    plt.tight_layout(); plt.savefig(f'{OUT}/C_conv_proj_idea2.png', dpi=150, bbox_inches='tight'); plt.close()
    print('C done')

# ---------------------------------------------------------------- FIG D: CAGE gap closing
def gap(name):
    p = f'{RP}/{name}/scale_greedy.json'
    if not os.path.exists(p): return None
    f = json.load(open(p))[-1]
    return f['hard_val'] - f['soft_val']
pairs = [('aggressive', 'aggressive_cage'), ('token_shift_k2', 'tshift2_cage')]
labels, base_g, cage_g = [], [], []
for a, c in pairs:
    ga, gc = gap(a), gap(c)
    if ga is not None and gc is not None:
        labels.append(a); base_g.append(ga); cage_g.append(gc)
if labels:
    import numpy as np
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, base_g, w, label='without CAGE', color='#9E9E9E', edgecolor='black', lw=0.5)
    ax.bar(x + w/2, cage_g, w, label='with CAGE', color='#43A047', edgecolor='black', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('soft→hard gap (nat)')
    ax.set_title('CAGE halves the discretization gap (but accuracy stays flat — gap already small)')
    ax.legend()
    plt.tight_layout(); plt.savefig(f'{OUT}/D_cage_gap.png', dpi=150, bbox_inches='tight'); plt.close()
    print('D done')

print('report2 figures done')

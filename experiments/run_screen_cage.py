"""Screening for CAGE + binary regularization variants.

Tests 2026 paper CAGE (Align Forward Adapt Backward, arxiv 2603.14157) and
RDDLGN binary regularization (arxiv 2508.06097, 2025) on top of our aggressive
LGN setup. Same screening protocol as run_screen.py: 4 layers, cheap finetune.
"""

import json
import os
import subprocess
import sys

CK = 'results/baseline.pt'
OUT = 'results/screen'
PYTHON = sys.executable
LAYERS = ['0', '5', '10', '11']

SCREEN_FLAGS = [
    '--imitation_steps', '100',
    '--finetune_steps',  '500',
    '--anneal_in_finetune',
    '--eval_iters',      '10',
    '--checkpoint',      CK,
    '--layers',          *LAYERS,
]

CONFIGS = [
    # ── CAGE alone (hard forward + adaptive backward τ_b) ──
    ('aggressive_cage',           ['--learn_pool', '--cage']),
    # ── Binary reg alone (RDDLGN) ──
    ('aggressive_binreg005',      ['--learn_pool', '--bin_reg_weight', '0.05']),
    ('aggressive_binreg020',      ['--learn_pool', '--bin_reg_weight', '0.20']),
    # ── CAGE + binreg combo ──
    ('aggressive_cage_binreg',    ['--learn_pool', '--cage', '--bin_reg_weight', '0.05']),
    # ── Cross-token x CAGE (proven winner combo + CAGE) ──
    ('tshift2_cage',              ['--learn_pool', '--token_shift', '2', '--cage']),
    # ── Hybrid L0 + CAGE (best hybrid + CAGE) ──
    ('hybridL0_cage',             ['--learn_pool', '--hybrid_layers', '0', '--cage']),
    # ── Full stack: hybrid + tshift + CAGE + binreg ──
    ('combo_cage_binreg',         ['--learn_pool', '--hybrid_layers', '0',
                                    '--token_shift', '2', '--cage',
                                    '--bin_reg_weight', '0.05']),
    # ── Gumbel-STE (Mind the Gap 2025) for comparison vs CAGE ──
    ('aggressive_gumbel',         ['--learn_pool', '--gumbel_ste']),
]


def screen_one(name, extra):
    out_dir = f'{OUT}/{name}'
    done = f'{out_dir}/heatmap.json'
    if os.path.exists(done):
        print(f'[skip] {name}')
        return
    print(f'\n{"="*60}\n[cage-screen] {name}\n{"="*60}')
    cmd = [PYTHON, 'run.py', 'heatmap'] + SCREEN_FLAGS + extra + ['--results_dir', out_dir]
    print(' '.join(cmd))
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        print(f'!!! {name} FAILED ({r.returncode})')
    else:
        print(f'[done] {name}')


def summarize():
    base_path = f'{OUT}/aggressive/heatmap.json'
    if not os.path.exists(base_path):
        print('No aggressive baseline.')
        return
    base = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(base_path))}

    rows = []
    all_names = ['aggressive'] + [c[0] for c in CONFIGS]
    for name in all_names:
        path = f'{OUT}/{name}/heatmap.json'
        if not os.path.exists(path):
            continue
        hd = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(path))}
        tot = sum(hd.values())
        dvs = sum(hd[L] - base[L] for L in hd if L in base)
        rows.append((name, hd, tot, dvs))
    rows.sort(key=lambda r: r[2])

    print('\n' + '='*78)
    print(f'{"Config":<26} {"L0":>7} {"L5":>7} {"L10":>7} {"L11":>7} {"SUM":>7} {"vs_agg":>8}')
    print('-'*78)
    for name, hd, tot, dvs in rows:
        line = f'{name:<26}'
        for L in [0, 5, 10, 11]:
            line += f' {hd.get(L, 0):+.3f}'
        verdict = ' BEST' if dvs < -0.4 else (' good' if dvs < -0.1 else (' worse' if dvs > 0.05 else ' flat'))
        line += f' {tot:+.3f} {dvs:+.3f}{verdict}'
        print(line)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        screen_one(name, extra)
    summarize()
    print('\nAll CAGE/binreg screenings done.')


if __name__ == '__main__':
    main()

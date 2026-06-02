"""Fast architecture screening: 4 representative layers × cheap fine-tune.

Rationale: per-layer `hard_degradation` strongly correlates with final cumulative-scaling
accuracy. So instead of running 12-layer cumulative scaling (~3h) for every new idea, we
run a per-layer heatmap on a representative SUBSET (L0, L5, L10, L11) with reduced
steps. Each config: ~10-15 minutes. Then only top candidates get full scaling.

Usage:
    python run_screen.py                 # runs all CONFIGS, ~2-3h
    python run_screen.py --rerun NAME    # forces re-run of one config

Output: results/screen/<name>/screen.json with per-layer hd.
At the end: ranking table comparing every config to aggressive baseline.
"""

import argparse
import json
import os
import subprocess
import sys

CK   = 'results/baseline.pt'
OUT  = 'results/screen'
PYTHON = sys.executable

# 4 representative layers — boundary + 1 middle for sanity.
# L0  = severe hd (1.06 in aggressive) — cross-token bottleneck
# L5  = easy middle layer (sanity check)
# L10 = hard pre-output layer
# L11 = hardest (lm_head precision)
LAYERS = ['0', '5', '10', '11']

# Cheap screening hyperparameters (vs report production: 200/3000/30).
SCREEN_FLAGS = [
    '--imitation_steps', '100',
    '--finetune_steps',  '500',
    '--anneal_in_finetune',
    '--eval_iters',      '10',   # was 30
    '--checkpoint',      CK,
    '--layers',          *LAYERS,
]

# Each config: (name, extra_flags). The aggressive default is binary_io + no_in_proj + sum_pool.
CONFIGS = [
    # ──────────────── BASELINE (must run for delta comparison) ────────────────
    ('aggressive',          ['--learn_pool']),

    # ──────────────── IDEA #2: Conv1d projections ────────────────
    # Replace Linear in/out_proj with causal Conv1d (cross-token via kernel)
    # Note: conv configs need --no-no_in_proj --no-sum_pool to enable real projections.
    ('conv3',               ['--no-no_in_proj', '--no-sum_pool',
                             '--binary_io', '--n_bits', '8',
                             '--conv_in_k', '3', '--conv_out_k', '3']),
    ('conv5',               ['--no-no_in_proj', '--no-sum_pool',
                             '--binary_io', '--n_bits', '8',
                             '--conv_in_k', '5', '--conv_out_k', '5']),
    ('conv3_in_only',       ['--no-no_in_proj',
                             '--binary_io', '--n_bits', '8',
                             '--conv_in_k', '3',
                             '--learn_pool']),  # sum_pool stays on
    ('conv7',               ['--no-no_in_proj', '--no-sum_pool',
                             '--binary_io', '--n_bits', '8',
                             '--conv_in_k', '7', '--conv_out_k', '7']),

    # ──────────────── COMBO ────────────────
    # Conv + token_shift K=2 (the proven best cross-token idea)
    ('conv3_tshift2',       ['--no-no_in_proj', '--no-sum_pool',
                             '--binary_io', '--n_bits', '8',
                             '--conv_in_k', '3', '--conv_out_k', '3',
                             '--token_shift', '2']),
    # Conv + hybrid L0 (keep attention, replace MLP with conv-LGN)
    ('conv3_hybridL0',      ['--no-no_in_proj', '--no-sum_pool',
                             '--binary_io', '--n_bits', '8',
                             '--conv_in_k', '3', '--conv_out_k', '3',
                             '--hybrid_layers', '0']),

    # ──────────────── BONUS: light Linear variants ────────────────
    # Just enable Linear in_proj/out_proj (no conv) — sanity for the "any projection"
    # baseline. If this matches conv3, the gain isn't from cross-token but from having
    # a projection at all.
    ('linear_proj',         ['--no-no_in_proj', '--no-sum_pool',
                             '--binary_io', '--n_bits', '8']),
]


def screen_one(name, extra):
    out_dir = f'{OUT}/{name}'
    done = f'{out_dir}/heatmap.json'
    if os.path.exists(done):
        print(f'[skip] {name}')
        return
    print(f'\n{"="*60}\n[screen] {name}\n{"="*60}')
    cmd = [PYTHON, 'run.py', 'heatmap'] + SCREEN_FLAGS + extra + ['--results_dir', out_dir]
    print(' '.join(cmd))
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0:
        print(f'!!! {name} FAILED ({r.returncode})')
        return
    print(f'[done] {name}')


def summarize():
    """After all configs are done, build a comparison table."""
    base_path = f'{OUT}/aggressive/heatmap.json'
    if not os.path.exists(base_path):
        print('No aggressive baseline — skipping summary.')
        return
    base = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(base_path))}

    rows = []
    for name, _ in CONFIGS:
        path = f'{OUT}/{name}/heatmap.json'
        if not os.path.exists(path):
            continue
        hd = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(path))}
        # Delta vs aggressive (negative = better, positive = worse)
        deltas = {L: hd[L] - base[L] for L in hd if L in base}
        rows.append({
            'name':    name,
            'hd':      hd,
            'total':   sum(hd.values()),
            'delta':   sum(deltas.values()),
            'best_L':  min(deltas, key=deltas.get) if deltas else None,
            'best_v':  min(deltas.values()) if deltas else 0.0,
        })

    # Rank by total hd (ascending = better)
    rows.sort(key=lambda r: r['total'])

    print('\n' + '='*78)
    print(f'{"Config":<22} {"L0":>7} {"L5":>7} {"L10":>7} {"L11":>7} {"Σhd":>7} {"Δvs agg":>9}')
    print('-'*78)
    for r in rows:
        L0  = r['hd'].get(0,  float('nan'))
        L5  = r['hd'].get(5,  float('nan'))
        L10 = r['hd'].get(10, float('nan'))
        L11 = r['hd'].get(11, float('nan'))
        mark = '★' if r['delta'] < -0.05 else (' ' if r['delta'] < 0.05 else '✗')
        print(f'{r["name"]:<22} {L0:+.3f} {L5:+.3f} {L10:+.3f} {L11:+.3f}  '
              f'{r["total"]:+.3f}  {r["delta"]:+.3f} {mark}')
    print('='*78)
    print('★ = clear improvement, ✗ = worse, blank = neutral')

    # Save summary as JSON
    with open(f'{OUT}/summary.json', 'w') as f:
        json.dump(rows, f, indent=2, default=str)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rerun', type=str, default=None, help='force-rerun a specific config')
    p.add_argument('--only_summary', action='store_true', help='just print summary')
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.rerun:
        import shutil
        d = f'{OUT}/{args.rerun}'
        if os.path.exists(d):
            shutil.rmtree(d)
        print(f'[rerun] cleared {d}')

    if not args.only_summary:
        for name, extra in CONFIGS:
            screen_one(name, extra)

    summarize()
    print('\nDone.')


if __name__ == '__main__':
    main()

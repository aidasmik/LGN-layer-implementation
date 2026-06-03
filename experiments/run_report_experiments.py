"""Comprehensive overnight experiment runner for the final LGN report.

All experiments use the FIXED code. Skip-if-done (re-runs are safe).
Total ETA: ~8-10 hours.

Three phases:
  A. Core scaling configs (5)            — main comparison
  B. Selective LGN configs (3)            — efficiency-quality curve
  C. Robustness / sweeps                  — variance, K sweep, n_bits sweep, etc.

Each entry: (name, mode, extra_flags). Mode = 'scale' or 'heatmap'.
"""

import os
import subprocess
import sys

# Run from the repo root regardless of where this script is invoked from
# (paths below + the `run.py` subprocess are relative to the repo root).
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CK = 'results/baseline.pt'
HM = 'results/aggressive/heatmap.json'
OUT_BASE = 'results/report'

# Shared base flags for scaling runs
SCALE_BASE = [
    '--strategy', 'greedy',
    '--imitation_steps', '200',
    '--anneal_in_finetune',
    '--finetune_steps', '3000',
    '--learn_pool',
    '--heatmap', HM,
    '--checkpoint', CK,
]

# Shared base flags for heatmap runs (per-layer single replacement)
HEATMAP_BASE = [
    '--imitation_steps', '200',
    '--anneal_in_finetune',
    '--finetune_steps', '3000',
    '--learn_pool',
    '--checkpoint', CK,
]

CONFIGS = [
    # ========================================================================
    # PHASE A: Core scaling — the main comparison configs (~2.5h)
    # ========================================================================
    ('identity',          'scale',   ['--identity_logic']),
    ('aggressive',        'scale',   []),
    ('token_shift_k2',    'scale',   ['--token_shift', '2']),
    ('hybrid_L0_agg',     'scale',   ['--hybrid_layers', '0']),
    ('combo',             'scale',   ['--hybrid_layers', '0', '--token_shift', '2']),

    # ========================================================================
    # PHASE B: Selective LGN — efficiency-quality curve (~1.5h)
    # ========================================================================
    ('sel_L0',            'scale',   ['--protected_layers', '0']),
    ('sel_edges',         'scale',   ['--protected_layers', '0', '11']),
    ('sel_4edges',        'scale',   ['--protected_layers', '0', '1', '10', '11']),

    # ========================================================================
    # PHASE C: Robustness & sweeps
    # ========================================================================

    # Per-layer heatmaps with FIXED code (used in figs 1, 4, 8)
    ('agg_heatmap',       'heatmap', []),
    ('agg_identity_hmap', 'heatmap', ['--identity_logic']),

    # Variance: 2 extra seeds for the main aggressive config (~1h)
    ('aggressive_s7',     'scale',   ['--seed', '7']),
    ('aggressive_s42',    'scale',   ['--seed', '42']),

    # Token shift K sweep (~1h)
    ('token_shift_k1',    'scale',   ['--token_shift', '1']),
    ('token_shift_k3',    'scale',   ['--token_shift', '3']),

    # Hybrid both edges — keep attention at L0 AND L11 (~30 min)
    ('hybrid_edges',      'scale',   ['--hybrid_layers', '0', '11']),

    # n_bits sweep — input granularity (~1h)
    ('aggressive_n4',     'scale',   ['--n_bits', '4']),
    ('aggressive_n16',    'scale',   ['--n_bits', '16']),
]


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    for name, mode, extra in CONFIGS:
        out_dir = f'{OUT_BASE}/{name}'
        # Skip if already complete (either scale or heatmap output present)
        done_marker = f'{out_dir}/metrics.json' if mode == 'scale' else f'{out_dir}/heatmap.json'
        if os.path.exists(done_marker):
            print(f'[skip] {name} (already complete)')
            continue
        print(f'\n{"=" * 60}\n[run {mode}] {name}\n{"=" * 60}')
        base = SCALE_BASE if mode == 'scale' else HEATMAP_BASE
        cmd = [sys.executable, 'run.py', mode] + base + extra + ['--results_dir', out_dir]
        print(' '.join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f'!!! {name} FAILED with code {result.returncode}')
        else:
            print(f'[done] {name}')
    print('\nAll experiments finished.')


if __name__ == '__main__':
    main()

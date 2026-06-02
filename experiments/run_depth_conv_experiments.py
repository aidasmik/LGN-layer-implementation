"""Phase D experiments: depth + random_from (idea #4) and Conv1d projections (idea #2).

These extend the report batch with two new architectural axes:
  * depth + random_from: stacked LGN sublayers with reservoir-style random interconnect
  * conv_in_k / conv_out_k: causal Conv1d in_proj / out_proj for cross-token mixing

Each config writes to results/report/<name>/metrics.json (skip-if-done).
"""

import os
import subprocess
import sys

CK = 'results/baseline.pt'
HM = 'results/aggressive/heatmap.json'
OUT_BASE = 'results/report'

# Aggressive scaling base (matches main batch).
SCALE_BASE = [
    '--strategy', 'greedy',
    '--imitation_steps', '200',
    '--anneal_in_finetune',
    '--finetune_steps', '3000',
    '--learn_pool',
    '--heatmap', HM,
    '--checkpoint', CK,
]

# For conv configs we need to DISABLE no_in_proj and sum_pool (conv replaces them).
CONV_BASE = [
    '--strategy', 'greedy',
    '--imitation_steps', '200',
    '--anneal_in_finetune',
    '--finetune_steps', '3000',
    '--no-no_in_proj',     # need a real in_proj for conv to replace
    '--no-sum_pool',       # need a real out_proj for conv to replace
    '--heatmap', HM,
    '--checkpoint', CK,
]


CONFIGS = [
    # ────────────────────────────────────────────────────────────────────────
    # IDEA #4: stacked depth with reservoir-style random interconnect
    # ────────────────────────────────────────────────────────────────────────
    # depth=2, all learnable — minimal stacking sanity check
    ('depth2_learn',      'scale',   SCALE_BASE, ['--depth', '2']),
    # depth=2 random — 1 learnable + 1 random
    ('depth2_rand',       'scale',   SCALE_BASE, ['--depth', '2', '--random_from', '1']),
    # depth=4 random — KEY config (1 learnable + 3 random, reservoir-style)
    ('depth4_rand',       'scale',   SCALE_BASE, ['--depth', '4', '--random_from', '1']),
    # depth=4 all learnable — comparison (more params, slower)
    ('depth4_learn',      'scale',   SCALE_BASE, ['--depth', '4']),
    # depth=4 random + token_shift K=2 — combine cross-token + deep expressivity
    ('depth4_rand_tshift2', 'scale', SCALE_BASE, ['--depth', '4', '--random_from', '1',
                                                  '--token_shift', '2']),
    # Per-layer heatmap with depth=4 rand to see where depth helps most
    ('depth4_rand_hmap',  'heatmap', SCALE_BASE, ['--depth', '4', '--random_from', '1']),

    # ────────────────────────────────────────────────────────────────────────
    # IDEA #2: Conv1d in/out projections (causal, kernel size K)
    # ────────────────────────────────────────────────────────────────────────
    # Conv kernel 3 on both in and out — local 3-tap temporal mixing
    # NOTE: requires binary_io=True; binary STE only on the conv OUTPUT before LGN body.
    # learn_pool is omitted (only used with sum_pool, which is OFF here).
    ('conv3',             'scale',   CONV_BASE, ['--binary_io', '--n_bits', '8',
                                                  '--conv_in_k', '3', '--conv_out_k', '3']),
    # Conv kernel 5 — wider receptive field
    ('conv5',             'scale',   CONV_BASE, ['--binary_io', '--n_bits', '8',
                                                  '--conv_in_k', '5', '--conv_out_k', '5']),
    # Conv input only (compress side uses sum_pool — cheaper)
    ('conv3_in_only',     'scale',   ['--strategy', 'greedy',
                                       '--imitation_steps', '200',
                                       '--anneal_in_finetune',
                                       '--finetune_steps', '3000',
                                       '--no-no_in_proj',
                                       '--sum_pool',  # keep sum_pool out
                                       '--learn_pool',
                                       '--heatmap', HM,
                                       '--checkpoint', CK],
                          ['--binary_io', '--n_bits', '8', '--conv_in_k', '3']),

    # ────────────────────────────────────────────────────────────────────────
    # COMBO: best ideas combined
    # ────────────────────────────────────────────────────────────────────────
    ('conv3_depth4_rand', 'scale',   CONV_BASE, ['--binary_io', '--n_bits', '8',
                                                  '--conv_in_k', '3', '--conv_out_k', '3',
                                                  '--depth', '4', '--random_from', '1']),
]


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    for name, mode, base, extra in CONFIGS:
        out_dir = f'{OUT_BASE}/{name}'
        done_marker = f'{out_dir}/metrics.json' if mode == 'scale' else f'{out_dir}/heatmap.json'
        if os.path.exists(done_marker):
            print(f'[skip] {name} (already complete)')
            continue
        print(f'\n{"=" * 60}\n[run {mode}] {name}\n{"=" * 60}')
        cmd = [sys.executable, 'run.py', mode] + base + extra + ['--results_dir', out_dir]
        print(' '.join(cmd))
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f'!!! {name} FAILED with code {result.returncode}')
        else:
            print(f'[done] {name}')
    print('\nAll depth/conv experiments finished.')


if __name__ == '__main__':
    main()

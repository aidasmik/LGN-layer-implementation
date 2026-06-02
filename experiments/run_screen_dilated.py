"""Screening for Idea E: dilated token shift (wide look-back span).

Tests whether an exponential tap pattern [1,2,4,8,(16)] beats the contiguous
token_shift K=2 winner, by reaching further back in the sequence at low channel cost.
Screening protocol: 4 layers (0,5,10,11), cheap finetune. Compare L0 hd especially.
"""

import json
import os
import subprocess
import sys

CK = 'results/baseline.pt'
OUT = 'results/screen'
PYTHON = sys.executable
LAYERS = ['0', '5', '10', '11']

BASE = [
    '--imitation_steps', '100',
    '--finetune_steps',  '500',
    '--anneal_in_finetune',
    '--eval_iters',      '10',
    '--checkpoint',      CK,
    '--layers',          *LAYERS,
    '--learn_pool',
]

CONFIGS = [
    # Contiguous baseline at screening params (for fair comparison)
    ('tshift_k2_screen',   ['--token_shift', '2']),
    # Dilated variants (Idea E)
    ('dilated_124',        ['--shift_taps', '1', '2', '4']),
    ('dilated_1248',       ['--shift_taps', '1', '2', '4', '8']),
    ('dilated_124816',     ['--shift_taps', '1', '2', '4', '8', '16']),
    # Best dilated + CAGE
    ('dilated_1248_cage',  ['--shift_taps', '1', '2', '4', '8', '--cage']),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        out_dir = f'{OUT}/{name}'
        if os.path.exists(f'{out_dir}/heatmap.json'):
            print(f'[skip] {name}')
            continue
        print(f'\n{"="*60}\n[dilated-screen] {name}\n{"="*60}')
        cmd = [PYTHON, 'run.py', 'heatmap'] + BASE + extra + ['--results_dir', out_dir]
        print(' '.join(cmd))
        r = subprocess.run(cmd, capture_output=False)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED')

    # Summary vs aggressive baseline + tshift_k2_screen
    base = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(f'{OUT}/aggressive/heatmap.json'))}
    print('\n' + '='*72)
    print(f'{"Config":<22} {"L0":>7} {"L5":>7} {"L10":>7} {"L11":>7} {"SUM":>7} {"vs_agg":>8}')
    print('-'*72)
    names = ['aggressive', 'tshift_k2_screen', 'dilated_124', 'dilated_1248',
             'dilated_124816', 'dilated_1248_cage']
    rows = []
    for name in names:
        p = f'{OUT}/{name}/heatmap.json'
        if not os.path.exists(p):
            continue
        d = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(p))}
        rows.append((name, d, sum(d.values()), sum(d.values())-sum(base.values())))
    rows.sort(key=lambda r: r[2])
    for name, d, t, v in rows:
        line = f'{name:<22}'
        for L in [0,5,10,11]: line += f' {d.get(L,0):+.3f}'
        print(line + f' {t:+.3f} {v:+.3f}')
    print('\nDilated screening done.')


if __name__ == '__main__':
    main()

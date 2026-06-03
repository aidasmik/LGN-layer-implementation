"""Phase B: full 12-layer cumulative scaling for the CAGE winners from screening.

Two configs only (~3h each):
  * aggressive_cage   — clean CAGE delta over aggressive
  * tshift2_cage      — best honest LGN combo from screening
"""

import os
import subprocess
import sys

# Run from the repo root regardless of invocation directory.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CK = 'results/baseline.pt'
HM = 'results/aggressive/heatmap.json'
OUT_BASE = 'results/report'

SCALE_BASE = [
    '--strategy', 'greedy',
    '--imitation_steps', '200',
    '--anneal_in_finetune',
    '--finetune_steps', '3000',
    '--learn_pool',
    '--heatmap', HM,
    '--checkpoint', CK,
]

CONFIGS = [
    ('aggressive_cage',  SCALE_BASE + ['--cage']),
    ('tshift2_cage',     SCALE_BASE + ['--cage', '--token_shift', '2']),
]


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    for name, extra in CONFIGS:
        out_dir = f'{OUT_BASE}/{name}'
        if os.path.exists(f'{out_dir}/metrics.json'):
            print(f'[skip] {name} (already complete)')
            continue
        print(f'\n{"="*60}\n[scale] {name}\n{"="*60}')
        cmd = [sys.executable, 'run.py', 'scale'] + extra + ['--results_dir', out_dir]
        print(' '.join(cmd))
        r = subprocess.run(cmd, capture_output=False)
        if r.returncode != 0:
            print(f'!!! {name} FAILED ({r.returncode})')
        else:
            print(f'[done] {name}')
    print('\nAll CAGE scaling experiments done.')


if __name__ == '__main__':
    main()

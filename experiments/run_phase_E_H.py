"""Phase E+H production scaling.

Idea E (dilated token shift): full 12-layer scaling for two tap patterns.
Idea H (joint polish + KL): re-scale token_shift_k2 with a final joint-polish pass
  that coordinates all LGN layers + distills the transformer's logits.
"""

import os
import subprocess
import sys

CK = 'results/baseline.pt'
HM = 'results/aggressive/heatmap.json'
OUT = 'results/report'

COMMON = [
    '--strategy', 'greedy',
    '--imitation_steps', '200',
    '--anneal_in_finetune',
    '--finetune_steps', '3000',
    '--learn_pool',
    '--heatmap', HM,
    '--checkpoint', CK,
]

CONFIGS = [
    # Idea E: dilated taps (span 4) — best in screening
    ('dilated_124',          COMMON + ['--shift_taps', '1', '2', '4']),
    # Idea E: widest span (16) — real test of long look-back (3000 steps to learn it)
    ('dilated_124816',       COMMON + ['--shift_taps', '1', '2', '4', '8', '16']),
    # Idea H: token_shift_k2 + joint polish + KL distillation (coordination fix)
    ('tshift2_polish_kl',    COMMON + ['--token_shift', '2',
                                       '--joint_polish_steps', '1500',
                                       '--joint_polish_kl_weight', '0.5']),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        out = f'{OUT}/{name}'
        if os.path.exists(f'{out}/metrics.json'):
            print(f'[skip] {name}'); continue
        print(f'\n{"="*60}\n[scale] {name}\n{"="*60}')
        cmd = [sys.executable, 'run.py', 'scale'] + extra + ['--results_dir', out]
        print(' '.join(cmd))
        r = subprocess.run(cmd)
        print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED')
    print('\nPhase E+H done.')


if __name__ == '__main__':
    main()

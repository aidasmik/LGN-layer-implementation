"""Identity ablation screening: verify whether the LGN body is actually doing
work in each conv variant, or whether the wrapping Linear/Conv projections
carry the signal alone.

Method: for each architecture, run the same screening setup as run_screen.py,
but with --identity_logic, which replaces every LearnedLogicLayer.forward with
'return x' (i.e., LGN does nothing). Compare per-layer hard_degradation against
the non-identity version:
  * identity_hd >> non_identity_hd  -> LGN is doing real work (architecture
    is honest)
  * identity_hd ~= non_identity_hd  -> LGN is decorative; the projections
    (Linear or Conv) carry the signal (FAKE LGN, reject)
"""

import json
import os
import subprocess
import sys

CK   = 'results/baseline.pt'
OUT  = 'results/screen'
PYTHON = sys.executable
LAYERS = ['0', '5', '10', '11']

SCREEN_FLAGS = [
    '--imitation_steps', '100',
    '--finetune_steps',  '500',
    '--anneal_in_finetune',
    '--eval_iters',      '10',
    '--checkpoint',      CK,
    '--layers',          *LAYERS,
    '--identity_logic',           # KEY: LGN does nothing
]

# Same architectures as run_screen.py but with identity LGN.
CONFIGS = [
    ('aggressive_identity',  ['--learn_pool']),
    ('conv3_identity',       ['--no-no_in_proj', '--no-sum_pool',
                              '--binary_io', '--n_bits', '8',
                              '--conv_in_k', '3', '--conv_out_k', '3']),
    ('conv3_in_only_identity', ['--no-no_in_proj',
                                '--binary_io', '--n_bits', '8',
                                '--conv_in_k', '3',
                                '--learn_pool']),
    ('conv3_hybridL0_identity', ['--no-no_in_proj', '--no-sum_pool',
                                 '--binary_io', '--n_bits', '8',
                                 '--conv_in_k', '3', '--conv_out_k', '3',
                                 '--hybrid_layers', '0']),
    ('linear_proj_identity',  ['--no-no_in_proj', '--no-sum_pool',
                               '--binary_io', '--n_bits', '8']),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, extra in CONFIGS:
        out_dir = f'{OUT}/{name}'
        done = f'{out_dir}/heatmap.json'
        if os.path.exists(done):
            print(f'[skip] {name}')
            continue
        print(f'\n{"="*60}\n[id-screen] {name}\n{"="*60}')
        cmd = [PYTHON, 'run.py', 'heatmap'] + SCREEN_FLAGS + extra + ['--results_dir', out_dir]
        print(' '.join(cmd))
        r = subprocess.run(cmd, capture_output=False)
        if r.returncode != 0:
            print(f'!!! {name} FAILED ({r.returncode})')
        else:
            print(f'[done] {name}')

    # Compare identity vs non-identity for each architecture
    print('\n' + '='*78)
    print('LGN contribution check: hd(identity) - hd(non-identity)')
    print('Big positive value => LGN does real work | ~0 => LGN decorative (FAKE)')
    print('='*78)
    pairs = [
        ('aggressive',  'aggressive_identity'),
        ('conv3',       'conv3_identity'),
        ('conv3_in_only', 'conv3_in_only_identity'),
        ('conv3_hybridL0', 'conv3_hybridL0_identity'),
        ('linear_proj', 'linear_proj_identity'),
    ]
    header = f'{"Arch":<18} {"L0(id-no)":>11} {"L5(id-no)":>11} {"L10(id-no)":>12} {"L11(id-no)":>12} {"LGN_help":>10}'
    print(header)
    print('-'*78)
    for arch, ident in pairs:
        p_arch  = f'{OUT}/{arch}/heatmap.json'
        p_ident = f'{OUT}/{ident}/heatmap.json'
        if not (os.path.exists(p_arch) and os.path.exists(p_ident)):
            print(f'{arch:<18} (missing)')
            continue
        a  = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(p_arch))}
        i  = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(p_ident))}
        deltas = {L: i[L] - a[L] for L in a if L in i}
        line = f'{arch:<18}'
        for L in [0, 5, 10, 11]:
            d = deltas.get(L, float('nan'))
            line += f' {d:+9.3f}  '
        total = sum(deltas.values())
        verdict = 'REAL ' if total > 0.5 else ('FAKE' if total < 0.1 else 'weak')
        line += f' {total:+8.3f} {verdict}'
        print(line)
    print('='*78)


if __name__ == '__main__':
    main()

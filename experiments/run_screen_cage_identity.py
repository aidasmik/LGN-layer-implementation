"""Identity ablation for CAGE configs: verify the LGN body is actually doing work.

If aggressive_cage_identity has hd >> aggressive_cage hd -> LGN body is honest contributor.
If close -> LGN decorative (fake), CAGE wouldn't be a real LGN improvement.
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
    '--identity_logic',  # LGN body = pass-through; measure projections-only contribution
]

CONFIGS = [
    # CAGE configs to check
    ('aggressive_cage_identity',     ['--learn_pool', '--cage']),
    ('tshift2_cage_identity',        ['--learn_pool', '--token_shift', '2', '--cage']),
    ('combo_cage_binreg_identity',   ['--learn_pool', '--hybrid_layers', '0',
                                       '--token_shift', '2', '--cage',
                                       '--bin_reg_weight', '0.05']),
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

    print('\n' + '='*78)
    print('LGN-body contribution check for CAGE configs')
    print('Large positive (id_hd > arch_hd) => LGN does real work | ~0 => fake LGN')
    print('='*78)
    pairs = [
        ('aggressive_cage',     'aggressive_cage_identity'),
        ('tshift2_cage',        'tshift2_cage_identity'),
        ('combo_cage_binreg',   'combo_cage_binreg_identity'),
    ]
    print(f'{"Arch":<22} {"L0(id-arch)":>12} {"L5":>9} {"L10":>9} {"L11":>9} {"LGN_help":>10}')
    print('-'*78)
    for arch, ident in pairs:
        p_arch  = f'{OUT}/{arch}/heatmap.json'
        p_ident = f'{OUT}/{ident}/heatmap.json'
        if not (os.path.exists(p_arch) and os.path.exists(p_ident)):
            print(f'{arch:<22} (missing)')
            continue
        a  = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(p_arch))}
        i  = {r['layer_idx']: r['hard_degradation'] for r in json.load(open(p_ident))}
        line = f'{arch:<22}'
        for L in [0, 5, 10, 11]:
            d = i.get(L, float('nan')) - a.get(L, float('nan'))
            line += f' {d:+9.3f} '
        total = sum(i[L] - a[L] for L in a if L in i)
        verdict = 'REAL' if total > 0.5 else ('FAKE' if total < 0.1 else 'weak')
        line += f' {total:+8.3f} {verdict}'
        print(line)


if __name__ == '__main__':
    main()

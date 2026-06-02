import os, subprocess, sys
CK = 'results/baseline.pt'
HM = 'results/aggressive/heatmap.json'

CONFIGS = [
    # token_shift K=2 + reverse_greedy (best simple combo)
    ('token_shift_k2_reverse',  [
        '--strategy', 'reverse_greedy',
        '--imitation_steps', '200', '--anneal_in_finetune', '--finetune_steps', '3000',
        '--learn_pool', '--token_shift', '2',
        '--heatmap', HM, '--checkpoint', CK,
    ]),
    # aggressive_cage + reverse_greedy + tshift (full stack)
    ('tshift2_cage_reverse',  [
        '--strategy', 'reverse_greedy',
        '--imitation_steps', '200', '--anneal_in_finetune', '--finetune_steps', '3000',
        '--learn_pool', '--cage', '--token_shift', '2',
        '--heatmap', HM, '--checkpoint', CK,
    ]),
]
os.makedirs('results/report', exist_ok=True)
for name, extra in CONFIGS:
    out = f'results/report/{name}'
    if os.path.exists(f'{out}/metrics.json'):
        print(f'[skip] {name}'); continue
    print(f'\n{"="*60}\n[scale] {name}\n{"="*60}')
    cmd = [sys.executable, 'run.py', 'scale'] + extra + ['--results_dir', out]
    print(' '.join(cmd))
    r = subprocess.run(cmd)
    print(f'[done] {name}' if r.returncode == 0 else f'!!! {name} FAILED')
print('\nAll reverse_greedy scaling done.')

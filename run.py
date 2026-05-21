import argparse
import json
import os
import torch

from lgn import ExperimentConfig, make_gpt
from pipeline import WikiText2, train_baseline, estimate_loss, run_heatmap, run_scaling


def _common_args(p):
    # model
    p.add_argument('--n_embd',          type=int,   default=128)
    p.add_argument('--n_layer',         type=int,   default=12)
    p.add_argument('--n_head',          type=int,   default=4)
    p.add_argument('--dropout',         type=float, default=0.0)
    # logic architecture
    p.add_argument('--width_mult',      type=int,   default=2)
    p.add_argument('--depth',           type=int,   default=1)
    p.add_argument('--k',               type=int,   default=4)
    p.add_argument('--activation',      type=str,   default='sigmoid',
                   choices=['sigmoid', 'tanh', 'relu', 'hardsigmoid', 'none'])
    p.add_argument('--conn_init_scale', type=float, default=0.02)
    p.add_argument('--gate_init_scale', type=float, default=0.02)
    p.add_argument('--edge_depth',      type=int,   default=0,
                   help='depth override for edge layers (0 and n_layer-1); 0 disables')
    p.add_argument('--edge_width_mult', type=int,   default=0,
                   help='width_mult override for edge layers; 0 disables')
    p.add_argument('--hybrid_layers',   type=int,   nargs='*', default=[],
                   help='layer indices that keep original attention; logic replaces MLP only')
    p.add_argument('--identity_logic',  action='store_true',
                   help='ablation: LearnedLogicLayer returns input as output')

    # ===================================================================
    # AGGRESSIVE SETUP IS THE DEFAULT (no trained Linear around the LGN).
    # binary_io + no_in_proj + sum_pool are ON by default — this is the
    # setup where the logic gates actually do the work.
    #   * --classic            -> original Linear-sandwich setup (all OFF)
    #   * --no-binary_io / --no-sum_pool / --no-no_in_proj -> toggle one
    # ===================================================================
    p.add_argument('--classic',         action='store_true',
                   help='ORIGINAL Linear-sandwich setup (trained in_proj + out_proj around LGN). '
                        'By DEFAULT the aggressive setup is used (no Linear; LGN does the work).')
    p.add_argument('--binary_io',       action=argparse.BooleanOptionalAction, default=True,
                   help='binarize LGN inputs to {0,1} via STE (default: ON)')
    p.add_argument('--n_bits',          type=int, default=8,
                   help='bits per scalar in binarization (1 = plain threshold, >1 = thermometer; aggressive uses 8)')
    p.add_argument('--sum_pool',        action=argparse.BooleanOptionalAction, default=True,
                   help='replace out_proj with fixed group-sum aggregation (default: ON)')
    p.add_argument('--no_in_proj',      action=argparse.BooleanOptionalAction, default=True,
                   help='remove the trained Linear before LGN; LGN reads the embedding directly (default: ON)')
    # ===================================================================

    p.add_argument('--skip_gate',       action='store_true',
                   help='learnable scalar gating the LGN contribution (block can do less)')
    # training
    p.add_argument('--baseline_steps',  type=int,   default=5_000)
    p.add_argument('--imitation_steps', type=int,   default=1_000)
    p.add_argument('--finetune_steps',  type=int,   default=1_000)
    p.add_argument('--per_layer_anneal', action='store_true',
                   help='scale imitation steps by layer difficulty')
    p.add_argument('--ft_log_sharpness', action='store_true', default=True,
                   help='print per-layer sharpness during fine-tuning')
    p.add_argument('--ft_eval_hard',    action='store_true',
                   help='evaluate hard-snapped model periodically during fine-tuning')
    p.add_argument('--imit_loss',       type=str, default='mse', choices=['mse', 'kl'],
                   help='imitation loss: mse (match activations) or kl (match output distribution)')
    p.add_argument('--ste',             action='store_true',
                   help='straight-through estimator during fine-tuning (forward=hard, backward=soft)')
    # misc
    p.add_argument('--results_dir',     type=str,   default='results')
    p.add_argument('--checkpoint',      type=str,   default=None)


def _build_cfg(args):
    cfg = ExperimentConfig()
    # model
    cfg.model.n_embd    = args.n_embd
    cfg.model.n_layer   = args.n_layer
    cfg.model.n_head    = args.n_head
    cfg.model.dropout   = args.dropout
    # logic architecture
    cfg.logic.width_mult      = args.width_mult
    cfg.logic.depth           = args.depth
    cfg.logic.k               = args.k
    cfg.logic.activation      = args.activation
    cfg.logic.conn_init_scale = args.conn_init_scale
    cfg.logic.gate_init_scale = args.gate_init_scale
    cfg.logic.edge_depth      = args.edge_depth
    cfg.logic.edge_width_mult = args.edge_width_mult
    cfg.logic.hybrid_layers   = args.hybrid_layers
    cfg.logic.identity_logic  = args.identity_logic
    # Aggressive setup is the default; --classic flips back to the Linear-sandwich setup.
    if args.classic:
        cfg.logic.binary_io  = False
        cfg.logic.no_in_proj = False
        cfg.logic.sum_pool   = False
        cfg.logic.n_bits     = 1
    else:
        cfg.logic.binary_io  = args.binary_io
        cfg.logic.no_in_proj = args.no_in_proj
        cfg.logic.sum_pool   = args.sum_pool
        cfg.logic.n_bits     = args.n_bits
    cfg.logic.skip_gate       = args.skip_gate
    # training
    cfg.train.baseline_steps   = args.baseline_steps
    cfg.train.imitation_steps  = args.imitation_steps
    cfg.train.finetune_steps   = args.finetune_steps
    cfg.train.per_layer_anneal = args.per_layer_anneal
    cfg.train.ft_log_sharpness = args.ft_log_sharpness
    cfg.train.ft_eval_hard     = args.ft_eval_hard
    cfg.train.imit_loss        = args.imit_loss
    cfg.train.ste              = args.ste
    cfg.results_dir = args.results_dir
    os.makedirs(cfg.results_dir, exist_ok=True)
    return cfg


def _load_or_train(cfg, model, data, args):
    ckpt = args.checkpoint or os.path.join(cfg.results_dir, 'baseline.pt')
    if os.path.exists(ckpt):
        print(f'Loading baseline from {ckpt}')
        model.load_state_dict(torch.load(ckpt, map_location=next(model.parameters()).device))
    else:
        train_baseline(model, data, cfg.train)
        torch.save(model.state_dict(), os.path.join(cfg.results_dir, 'baseline.pt'))


def cmd_heatmap(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)
    cfg = _build_cfg(args)
    data = WikiText2(cfg.data, device)
    model, gpt_cfg = make_gpt(cfg.model, cfg.data, device)
    print(f'Model: {cfg.model.n_layer}L x {cfg.model.n_embd}d  ({sum(p.numel() for p in model.parameters()):,} params)')
    _load_or_train(cfg, model, data, args)
    save_path = os.path.join(cfg.results_dir, 'heatmap.json')
    run_heatmap(model, gpt_cfg, data, cfg, save_path=save_path)
    print(f'\nSaved -> {save_path}')


def cmd_scale(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1337)
    cfg = _build_cfg(args)
    data = WikiText2(cfg.data, device)
    model, gpt_cfg = make_gpt(cfg.model, cfg.data, device)
    _load_or_train(cfg, model, data, args)
    heatmap_results = None
    if args.strategy == 'greedy':
        with open(args.heatmap) as f:
            heatmap_results = json.load(f)
    save_path = os.path.join(cfg.results_dir, f'scale_{args.strategy}.json')
    run_scaling(model, gpt_cfg, data, cfg,
                strategy=args.strategy, heatmap_results=heatmap_results, save_path=save_path,
                protected_layers=args.protected_layers)
    print(f'\nSaved -> {save_path}')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_h = sub.add_parser('heatmap')
    _common_args(p_h)

    p_s = sub.add_parser('scale')
    _common_args(p_s)
    p_s.add_argument('--strategy',          type=str, default='greedy', choices=['greedy', 'uniform'])
    p_s.add_argument('--heatmap',           type=str, default='results/heatmap.json')
    p_s.add_argument('--protected_layers',  type=int, nargs='*', default=[],
                     help='layer indices to never replace (e.g. --protected_layers 0 11)')

    args = parser.parse_args()
    {'heatmap': cmd_heatmap, 'scale': cmd_scale}[args.cmd](args)


if __name__ == '__main__':
    main()

"""Full 12-layer cumulative scaling for idea #2 (channel-conv projections, stride=8),
then HARD-snap next-byte accuracy — directly comparable to the production table.

Greedy easiest-first order from results/aggressive/heatmap.json. Per layer: imitation
(200) + LM fine-tune (3000) with temperature anneal. After all 12 layers: soft eval +
hard (discrete-gate) eval, reporting loss / perplexity / accuracy.

Run from repo root (GPU free):  python experiments/conv_proj_scaling.py
"""

import copy
import json
import os
import sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from lgn import ExperimentConfig, make_gpt
from pipeline import WikiText2, get_layer_io, estimate_metrics

# reuse the block definition from the probe
import importlib.util
_spec = importlib.util.spec_from_file_location('conv_proj_probe', 'experiments/conv_proj_probe.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
ConvProjLGN = _m.ConvProjLGN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CK = 'results/baseline.pt'
HM = 'results/aggressive/heatmap.json'
# NOTE on W and memory: the 32->1 compressor with kernel=stride=S lands on C only if the
# LGN width W = 32 * (C*S) = 4096*S. Training the FIRST layer (L0) needs the full backward
# graph through all 12 wide blocks, so W must stay small enough that 12 gate tensors fit in
# 8 GB. The probe confirmed the REAL-LGN verdict is stride-independent (S=8 and S=16 both
# REAL), so the accuracy run uses S=2 (W=8192) — the same channel-bottleneck mechanism at a
# memory-feasible width.
STRIDE = 8                     # W = 4096*8 = 32768 gates/layer (32x aggressive) — user's exact spec
IMIT, FT, BS = 200, 3000, 8    # gradient checkpointing keeps all 12 wide blocks in 8 GB
OUT = 'results/report/conv_proj_s8'


def anneal(step, total, t0=2.0, t1=0.1):
    return t0 + (t1 - t0) * step / max(total - 1, 1)


def main():
    cfg = ExperimentConfig()
    cfg.train.batch_size = BS
    cfg.train.eval_iters = 50
    base, gpt_cfg = make_gpt(cfg.model, cfg.data, DEVICE)
    base.load_state_dict(torch.load(CK, map_location=DEVICE))
    base.eval()
    data = WikiText2(cfg.data, DEVICE)

    order = [r['layer_idx'] for r in sorted(json.load(open(HM)), key=lambda r: r['hard_degradation'])]
    print(f'order: {order}', flush=True)

    base_m = estimate_metrics(base, data, cfg.train.eval_iters, BS)
    print(f'transformer: loss={base_m["loss"]:.4f} ppl={base_m["perplexity"]:.2f} acc={base_m["accuracy"]*100:.2f}%', flush=True)

    live = copy.deepcopy(base)
    blocks = {}                    # layer_idx -> ConvProjLGN
    for n, li in enumerate(order, 1):
        blk = ConvProjLGN(C=gpt_cfg.n_embd, stride=STRIDE, seed=1000 + li).to(DEVICE)
        blk.use_ckpt = True            # gradient checkpointing on every wide block
        live.replace_layer(li, blk)
        blocks[li] = blk
        for p in live.parameters():
            p.requires_grad = False
        for p in blk.parameters():
            p.requires_grad = True
        trainable = [p for p in blk.parameters() if p.requires_grad]

        # imitation
        blk.set_temperature(2.0); blk.train()
        opt = torch.optim.AdamW(trainable, lr=2e-3)
        for _ in range(IMIT):
            lin, ltgt = get_layer_io(base, li, data, BS, input_model=live)
            blk.train()
            F.mse_loss(blk(lin), ltgt).backward()
            opt.step(); opt.zero_grad(set_to_none=True)

        # LM fine-tune (only the new block anneals; settled blocks stay sharp)
        opt = torch.optim.AdamW(trainable, lr=2e-3)
        live.train()
        for s in range(FT):
            blk.set_temperature(anneal(s, FT))
            _, lm = live(*data.get_batch('train', BS))
            lm.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        blk.set_temperature(0.1)
        torch.cuda.empty_cache()
        print(f'[{n}/12] added L{li}', flush=True)

    # final eval — soft then hard (discrete gates)
    soft = estimate_metrics(live, data, cfg.train.eval_iters, BS)
    for b in blocks.values():
        b.hard_eval = True
    hard = estimate_metrics(live, data, cfg.train.eval_iters, BS)
    for b in blocks.values():
        b.hard_eval = False

    os.makedirs(OUT, exist_ok=True)
    res = {'transformer': base_m, 'lgn_soft': soft, 'lgn_hard': hard,
           'stride': STRIDE, 'gates_per_layer': blocks[order[0]].W, 'n_lgn_layers': 12}
    json.dump(res, open(f'{OUT}/metrics.json', 'w'), indent=2)
    print('\n=== conv_proj stride=8 (channel-conv, idea #2) ===', flush=True)
    print(f'transformer : acc={base_m["accuracy"]*100:.2f}%  loss={base_m["loss"]:.3f}', flush=True)
    print(f'LGN soft    : acc={soft["accuracy"]*100:.2f}%  loss={soft["loss"]:.3f}', flush=True)
    print(f'LGN hard    : acc={hard["accuracy"]*100:.2f}%  loss={hard["loss"]:.3f}  ppl={hard["perplexity"]:.2f}', flush=True)
    print(f'gates/layer={blocks[order[0]].W} (32x aggressive); soft->hard gap={hard["loss"]-soft["loss"]:+.4f}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()

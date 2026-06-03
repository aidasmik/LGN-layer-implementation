"""Probe idea #2 EXACTLY as specified: channel-dimension Conv1d projections with stride,
instead of Linear / temporal-conv projections.

User spec:
  - expansion (in_proj):  Conv1d(in_ch=4,  out_ch=64, ...) over a reshaped feature axis
  - compression (out_proj): Conv1d(in_ch=32, out_ch=1, stride in 8..32)

Geometry (documented choices to make the chain self-consistent):
  x:(B,T,C=128)
    -> view (B*T, 4, 32)                       # in_ch=4, length=C/4=32
    -> Conv1d(4, 64, k=3, s=1, pad=1)          # EXPAND channels 4->64, length kept 32
    -> flatten (B*T, 64*32 = 2048)             # LGN feature width
    -> binarize (1-bit threshold)              # -> 2048 Boolean inputs
    -> LearnedLogicLayer(2048 -> W)            # W = 32 * (C*stride) so the compressor lands on C
    -> view (B*T, 32, C*stride)                # in_ch=32 for the compressor
    -> Conv1d(32, 1, k=stride, s=stride)       # COMPRESS 32->1, downsample length C*stride -> C
    -> (B*T, C) -> + residual

Because the stride>1 compressor is a DOWNSAMPLER, the LGN must EXPAND (2048 -> W), so a
pass-through "identity" LGN is dimensionally impossible. The honest fake-LGN test here is
therefore: REAL (trainable LGN) vs FROZEN (LGN params frozen at random init). If the
trained convs reach the same quality with a random frozen Boolean middle, the convs do the
work and the LGN is decorative (fake), exactly as the temporal-conv variants were.

Run from repo root (after the GPU is free):  python experiments/conv_proj_probe.py
"""

import copy
import os
import sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '.')
from lgn import (ExperimentConfig, make_gpt, LearnedLogicLayer, _binarize_ste)
from pipeline import WikiText2, get_layer_io, estimate_loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CK = 'results/baseline.pt'


class ConvProjLGN(nn.Module):
    """Channel-conv-projected LGN block (idea #2). Drop-in for a transformer block's
    output at one layer: norm -> expand-conv -> binarize -> LGN -> compress-conv -> +res."""
    def __init__(self, C=128, expand_in=4, expand_out=64, compress_in=32,
                 stride=8, seed=0, freeze_logic=False):
        super().__init__()
        self.C = C
        self.expand_in = expand_in
        self.compress_in = compress_in
        self.stride = stride
        self.norm = nn.LayerNorm(C)
        # EXPAND: (expand_in, C/expand_in) -> (expand_out, C/expand_in)
        self.in_conv = nn.Conv1d(expand_in, expand_out, kernel_size=3, stride=1, padding=1)
        feat = expand_out * (C // expand_in)            # 64*32 = 2048
        self.feat = feat
        # LGN body: feat -> W, W chosen so the compressor lands on exactly C
        self.comp_len = C * stride                      # length feeding the compressor
        W = compress_in * self.comp_len                 # e.g. 32 * 1024 = 32768
        self.W = W
        self.lgn = LearnedLogicLayer(feat, W, k=4, seed=seed)
        # COMPRESS: (compress_in, comp_len) -> (1, C) via stride==kernel downsampling
        self.out_conv = nn.Conv1d(compress_in, 1, kernel_size=stride, stride=stride, padding=0)
        if freeze_logic:
            for p in self.lgn.parameters():
                p.requires_grad = False
        self.freeze_logic = freeze_logic
        self.hard_eval = False  # when True, forward uses discrete (argmax) gates
        self.use_ckpt = False   # gradient checkpointing: recompute fwd in backward (saves memory)

    def set_temperature(self, t):
        self.lgn.set_temperature(t)

    def _impl(self, x, hard):
        B, T, C = x.shape
        h = self.norm(x).reshape(B * T, self.expand_in, C // self.expand_in)
        h = self.in_conv(h).reshape(B * T, self.feat)        # (B*T, 2048)
        h = _binarize_ste(torch.sigmoid(h))                  # Boolean inputs
        h = self.lgn(h, hard=hard)                           # (B*T, W)
        h = h.reshape(B * T, self.compress_in, self.comp_len)
        h = self.out_conv(h).reshape(B, T, C)                # (B,T,C)
        return x + h

    def forward(self, x, hard=False):
        hard = hard or self.hard_eval
        # During training, checkpoint: don't retain this block's big gate tensor for
        # backward — recompute it. Lets all 12 wide blocks fit in 8 GB at stride 8/16.
        # Guard: only checkpoint when backward will actually flow through this block —
        # either the input carries grad (downstream of the active layer) or this block
        # itself is trainable (the active layer, e.g. L0 whose embedding input has no grad).
        if self.use_ckpt and self.training and not hard:
            needs_bw = x.requires_grad or any(p.requires_grad for p in self.parameters())
            if needs_bw:
                from torch.utils.checkpoint import checkpoint
                return checkpoint(self._impl, x, hard, use_reentrant=False)
        return self._impl(x, hard)


@torch.no_grad()
def layer_soft_loss(model, data, cfg):
    return estimate_loss(model, data, cfg.train.eval_iters, cfg.train.batch_size)


def train_one(layer_idx, stride, freeze_logic, cfg, base, gpt_cfg, data,
              imit_steps=100, ft_steps=500):
    """Replace one layer's block-output path with a ConvProjLGN and train it
    (imitation + LM fine-tune) on the frozen base. Returns hard-eval degradation."""
    torch.manual_seed(0)
    live = copy.deepcopy(base)
    blk = ConvProjLGN(C=gpt_cfg.n_embd, stride=stride, seed=1000 + layer_idx,
                      freeze_logic=freeze_logic).to(DEVICE)
    live.replace_layer(layer_idx, blk)
    for p in live.parameters():
        p.requires_grad = False
    for p in blk.parameters():
        p.requires_grad = not (p is None)
    if freeze_logic:
        for p in blk.lgn.parameters():
            p.requires_grad = False
    trainable = [p for p in blk.parameters() if p.requires_grad]

    # Imitation
    blk.set_temperature(2.0)
    opt = torch.optim.AdamW(trainable, lr=2e-3)
    blk.train()
    for _ in range(imit_steps):
        lin, ltgt = get_layer_io(base, layer_idx, data, cfg.train.batch_size, input_model=live)
        blk.train()
        loss = F.mse_loss(blk(lin), ltgt)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

    # LM fine-tune with temperature anneal
    opt = torch.optim.AdamW(trainable, lr=2e-3)
    live.train()
    for s in range(ft_steps):
        blk.set_temperature(2.0 - (2.0 - 0.1) * s / max(ft_steps - 1, 1))
        _, lm = live(*data.get_batch('train', cfg.train.batch_size))
        lm.backward(); opt.step(); opt.zero_grad(set_to_none=True)

    blk.set_temperature(0.1)
    base_val = layer_soft_loss(base, data, cfg)
    soft = layer_soft_loss(live, data, cfg) - base_val
    return soft


def main():
    cfg = ExperimentConfig()
    cfg.train.batch_size = 8     # smaller batch: stride 16/32 give 65k/131k-wide LGN (8GB GPU)
    cfg.train.eval_iters = 20
    base, gpt_cfg = make_gpt(cfg.model, cfg.data, DEVICE)
    base.load_state_dict(torch.load(CK, map_location=DEVICE))
    base.eval()
    data = WikiText2(cfg.data, DEVICE)

    print(f'{"layer":>5} {"stride":>6} {"REAL soft_d":>12} {"FROZEN soft_d":>14} {"LGN_contrib":>12} {"verdict":>8}', flush=True)
    print('-' * 64, flush=True)
    for layer_idx in [0, 11]:
        for stride in [8, 16, 32]:
            real   = train_one(layer_idx, stride, False, cfg, base, gpt_cfg, data)
            frozen = train_one(layer_idx, stride, True,  cfg, base, gpt_cfg, data)
            contrib = frozen - real           # >0 => trainable LGN beats random => real work
            verdict = 'REAL' if contrib > 0.10 else ('FAKE' if contrib < 0.03 else 'weak')
            print(f'{layer_idx:>5} {stride:>6} {real:>12.4f} {frozen:>14.4f} {contrib:>12.4f} {verdict:>8}', flush=True)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

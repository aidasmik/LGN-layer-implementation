"""
Core definitions: config, logic gate layers, nanoGPT wrappers.
"""

import copy
import os
import sys
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LogicConfig:
    width_mult: int = 2
    depth: int = 1
    k: int = 4
    activation: str = 'sigmoid'
    conn_init_scale: float = 0.02
    gate_init_scale: float = 0.02
    edge_depth: int = 0           
    edge_width_mult: int = 0      
    hybrid_layers: list = field(default_factory=list)  
    identity_logic: bool = False  

    # AGGRESSIVE SETUP

    binary_io: bool = True       
    n_bits: int = 8               
    sum_pool: bool = True         
    no_in_proj: bool = True       
    # ===================================================================

    skip_gate: bool = False      

@dataclass
class TrainConfig:
    baseline_steps: int = 5_000
    baseline_lr: float = 1e-3
    batch_size: int = 32
    eval_iters: int = 30
    log_every: int = 500
    imitation_steps: int = 1_000
    imitation_lr: float = 2e-3
    temp_start: float = 2.0
    temp_end: float = 0.1
    ent_conn: float = 0.001
    ent_gate: float = 0.02
    finetune_steps: int = 1_000
    finetune_lr: float = 2e-3
    ft_ent_conn: float = 0.0005
    ft_ent_gate: float = 0.01
    per_layer_anneal: bool = False  # scale imitation steps by layer difficulty
    ft_log_sharpness: bool = True   # print per-layer sharpness
    ft_eval_hard: bool = False      # evaluate hard-snapped model
    imit_loss: str = 'mse'         
    ste: bool = False               # straight-through estimator 

@dataclass
class DataConfig:
    train_chars: int = 5_000_000
    val_chars: int = 500_000
    block_size: int = 64
    vocab_size: int = 256

@dataclass
class ModelConfig:
    n_layer: int = 12
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0

@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    results_dir: str = "results"
    seed: int = 1337

# ---------------------------------------------------------------------------
# nanoGPT wrappers
# ---------------------------------------------------------------------------

NANOGPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'nanogpt_layer_lab', 'nanoGPT')
)
if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)

from model import GPT, GPTConfig  # noqa: E402


def _patch_replace_layer():
    if hasattr(GPT, 'replace_layer'):
        return
    def replace_layer(self, idx, layer):
        if not 0 <= idx < len(self.transformer.h):
            raise IndexError(f"idx {idx} out of range")
        self.transformer.h[idx] = layer
    GPT.replace_layer = replace_layer

_patch_replace_layer()


def make_gpt(model_cfg, data_cfg, device='cuda'):
    cfg = GPTConfig(
        block_size=data_cfg.block_size,
        vocab_size=data_cfg.vocab_size,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        dropout=model_cfg.dropout,
    )
    return GPT(cfg).to(device), cfg

# ---------------------------------------------------------------------------
# Logic gate matrix  (16 gates expressed over [1, A, B, A*B])
# ---------------------------------------------------------------------------

LOGIC_GATE_MATRIX = torch.tensor([
    [0,  0,  0,  0],  # False
    [0,  0,  0,  1],  # AND
    [0,  1,  0, -1],  # A AND NOT B
    [0,  1,  0,  0],  # A
    [0,  0,  1, -1],  # NOT A AND B
    [0,  0,  1,  0],  # B
    [0,  1,  1, -2],  # XOR
    [0,  1,  1, -1],  # OR
    [1, -1, -1,  1],  # NOR
    [1, -1, -1,  2],  # XNOR
    [1,  0, -1,  0],  # NOT B
    [1,  0, -1,  1],  # A OR NOT B
    [1, -1,  0,  0],  # NOT A
    [1, -1,  0,  1],  # NOT A OR B
    [1,  0,  0, -1],  # NAND
    [1,  0,  0,  0],  # True
], dtype=torch.float32)


def diff_logic_gates(a, b):
    basis = torch.stack([torch.ones_like(a), a, b, a * b], dim=-1)
    return basis @ LOGIC_GATE_MATRIX.to(device=a.device, dtype=a.dtype).T  # (..., 16)


def annealed_temperature(step, total, start=2.0, end=0.1):
    return start * (end / start) ** (step / max(total - 1, 1))


def _apply_activation(h, name):
    """Pointwise activation dispatch shared by all logic gate blocks."""
    if name == 'sigmoid':     return torch.sigmoid(h)
    if name == 'tanh':        return torch.tanh(h)
    if name == 'relu':        return F.relu(h)
    if name == 'hardsigmoid': return F.hardsigmoid(h)
    return h  # 'none'


def _binarize_ste(h, threshold=0.5):
    """Threshold to {0, 1}. Forward: hard binary, backward: identity gradient."""
    binary = (h > threshold).to(dtype=h.dtype)
    return h + (binary - h).detach()


def _thermometer_ste(h, n_bits, training):
    """Thermometer encoding: each scalar in [0,1] becomes n_bits binary features.
    bit_i = (h > (i+1)/(n_bits+1)). Output shape: (..., D) -> (..., D*n_bits).
    Forward: hard binary. Backward: identity gradient w.r.t. h, broadcast across bits."""
    *prefix, D = h.shape
    # Thresholds spaced uniformly in (0, 1)
    levels = torch.linspace(1.0 / (n_bits + 1), n_bits / (n_bits + 1), n_bits,
                            device=h.device, dtype=h.dtype)
    expanded = h.unsqueeze(-1).expand(*prefix, D, n_bits)
    hard = (expanded > levels).to(dtype=h.dtype)
    if training:
        # Soft surrogate: piecewise-linear ramp around each threshold; gradient flows
        soft = (expanded - levels).clamp(0, 1)
        out = soft + (hard - soft).detach()
    else:
        out = hard
    return out.reshape(*prefix, D * n_bits)

# ---------------------------------------------------------------------------
# Soft learnable logic layer
# ---------------------------------------------------------------------------

class LearnedLogicLayer(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, seed=None, temperature=1.0,
                 conn_init_scale=0.02, gate_init_scale=0.02, identity=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.temperature = float(temperature)
        self.identity = identity
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        self.register_buffer('cand_a', torch.randint(0, in_dim, (out_dim, k), generator=g))
        self.register_buffer('cand_b', torch.randint(0, in_dim, (out_dim, k), generator=g))
        self.conn_logits_a = nn.Parameter(torch.randn(out_dim, k) * conn_init_scale)
        self.conn_logits_b = nn.Parameter(torch.randn(out_dim, k) * conn_init_scale)
        self.gate_logits   = nn.Parameter(torch.randn(out_dim, 16) * gate_init_scale)

    def set_temperature(self, t):
        self.temperature = float(t)

    def _sm(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        def ent(logits):
            p = self._sm(logits)
            return -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        return conn_w * (ent(self.conn_logits_a) + ent(self.conn_logits_b)) + gate_w * ent(self.gate_logits)

    @torch.no_grad()
    def sharpness(self):
        return {
            'conn_a': float(self._sm(self.conn_logits_a).max(dim=-1).values.mean()),
            'conn_b': float(self._sm(self.conn_logits_b).max(dim=-1).values.mean()),
            'gate':   float(self._sm(self.gate_logits).max(dim=-1).values.mean()),
        }

    def _select(self, x, cand, logits, hard, ste=False):
        gathered = x[:, cand]
        if ste:
            soft = self._sm(logits)
            hard_w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
            w = soft + (hard_w - soft).detach()  # forward=hard, backward=soft
        elif hard:
            w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
        else:
            w = self._sm(logits)
        return (gathered * w).sum(dim=-1)

    def forward(self, x, hard=False, ste=False):
        if self.identity:
            return x  # ablation: bypass all logic computation
        a = self._select(x, self.cand_a, self.conn_logits_a, hard, ste=ste)
        b = self._select(x, self.cand_b, self.conn_logits_b, hard, ste=ste)
        gates = diff_logic_gates(a, b)
        if ste:
            soft = self._sm(self.gate_logits)
            hard_gp = F.one_hot(self.gate_logits.argmax(dim=-1), 16).to(dtype=x.dtype)
            gp = soft + (hard_gp - soft).detach()
        elif hard:
            gp = F.one_hot(self.gate_logits.argmax(dim=-1), 16).to(dtype=x.dtype)
        else:
            gp = self._sm(self.gate_logits)
        return (gates * gp).sum(dim=-1)

# ---------------------------------------------------------------------------
# GPT block wrapper: norm -> sigmoid(proj) -> logic stack -> proj + residual
# ---------------------------------------------------------------------------

class LogicGateGPTLayer(nn.Module):
    def __init__(self, gpt_cfg, layer_idx, logic_width=None, depth=1, k=4, seed=1000,
                 activation='sigmoid', conn_init_scale=0.02, gate_init_scale=0.02,
                 identity_logic=False, binary_io=False, n_bits=1, sum_pool=False,
                 no_in_proj=False, skip_gate=False):
        super().__init__()
        C = gpt_cfg.n_embd
        self.C = C
        self.layer_idx = layer_idx
        self.logic_width = logic_width or C * 4
        self.activation = activation
        self.binary_io = binary_io
        self.n_bits    = n_bits if binary_io else 1
        self.sum_pool  = sum_pool
        self.no_in_proj = no_in_proj
        # When no_in_proj: LGN operates directly on n_embd binarized features (× n_bits)
        if no_in_proj:
            assert binary_io, "no_in_proj requires binary_io"
            bit_width = C * self.n_bits
        else:
            bit_width = self.logic_width * self.n_bits
        if sum_pool:
            assert bit_width % C == 0, (
                f"sum_pool requires bit_width ({bit_width}) divisible by n_embd ({C}). "
                f"Increase --width_mult or --n_bits.")
            self.group_size = bit_width // C
        self.norm     = nn.LayerNorm(C)
        if not no_in_proj:
            self.in_proj = nn.Linear(C, self.logic_width)
        self.logic    = nn.ModuleList([
            LearnedLogicLayer(bit_width, bit_width, k=k,
                              seed=seed + layer_idx * 100 + i,
                              conn_init_scale=conn_init_scale,
                              gate_init_scale=gate_init_scale,
                              identity=identity_logic)
            for i in range(depth)
        ])
        if not sum_pool:
            self.out_proj = nn.Linear(bit_width, C)
        self.dropout  = nn.Dropout(gpt_cfg.dropout)
        self.use_ste  = False  # straight-through estimator toggle (set during fine-tune)
        # Learnable scalar gating the LGN contribution to residual
        self.skip_gate = skip_gate
        if skip_gate:
            self.skip_alpha = nn.Parameter(torch.ones(1))

    def set_temperature(self, t):
        for l in self.logic: l.set_temperature(t)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _aggregate(self, h, B, T):
        """Convert (B*T, bit_width) → (B, T, C) for residual addition.

        With sum_pool: fixed group-sum, no trained parameters between LGN and residual.
        Without sum_pool: trained out_proj Linear."""
        if self.sum_pool:
            # Sum every `group_size` bits → one output channel.
            pooled = h.view(B * T, self.C, self.group_size).sum(dim=-1)
            # Center & scale so output ~ [-1, 1] (assumes uniform bit usage).
            normed = (pooled - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        return self.out_proj(h).view(B, T, self.C)

    def forward(self, x, hard=False):
        B, T, C = x.shape
        normed = self.norm(x).reshape(B * T, C)
        h = normed if self.no_in_proj else self.in_proj(normed)
        h = _apply_activation(h, self.activation)
        if self.binary_io:
            if self.n_bits > 1:
                h = _thermometer_ste(h, self.n_bits, self.training)
            else:
                h = _binarize_ste(h)
        ste = self.use_ste and not hard
        for l in self.logic:
            h = l(h, hard=hard, ste=ste)
        contrib = self.dropout(self._aggregate(h, B, T))
        if self.skip_gate:
            contrib = self.skip_alpha * contrib
        return x + contrib

# ---------------------------------------------------------------------------
# Hard (fully discrete) versions
# ---------------------------------------------------------------------------

class HardLogicLayer(nn.Module):
    def __init__(self, soft: LearnedLogicLayer):
        super().__init__()
        self.identity = getattr(soft, 'identity', False)
        with torch.no_grad():
            choice_a = soft.conn_logits_a.argmax(dim=-1)
            choice_b = soft.conn_logits_b.argmax(dim=-1)
            idx_a = soft.cand_a.gather(1, choice_a.unsqueeze(1)).squeeze(1)
            idx_b = soft.cand_b.gather(1, choice_b.unsqueeze(1)).squeeze(1)
            self.register_buffer('idx_a', idx_a.clone())
            self.register_buffer('idx_b', idx_b.clone())
            self.register_buffer('coeffs', LOGIC_GATE_MATRIX[soft.gate_logits.argmax(dim=-1).cpu()].clone())

    def forward(self, x):
        if self.identity:
            return x
        a, b = x[:, self.idx_a], x[:, self.idx_b]
        c = self.coeffs.to(device=x.device, dtype=x.dtype)
        return c[:, 0] + c[:, 1]*a + c[:, 2]*b + c[:, 3]*a*b


class HardLogicGateGPTLayer(nn.Module):
    def __init__(self, soft: LogicGateGPTLayer):
        super().__init__()
        self.layer_idx = soft.layer_idx
        self.activation = soft.activation
        self.binary_io  = getattr(soft, 'binary_io', False)
        self.n_bits     = getattr(soft, 'n_bits', 1)
        self.sum_pool   = getattr(soft, 'sum_pool', False)
        self.no_in_proj = getattr(soft, 'no_in_proj', False)
        self.skip_gate  = getattr(soft, 'skip_gate', False)
        self.C          = getattr(soft, 'C', None)
        self.group_size = getattr(soft, 'group_size', None)
        if self.skip_gate:
            self.register_buffer('skip_alpha', soft.skip_alpha.detach().clone())
        self.norm     = copy.deepcopy(soft.norm)
        if not self.no_in_proj:
            self.in_proj = copy.deepcopy(soft.in_proj)
        if not self.sum_pool:
            self.out_proj = copy.deepcopy(soft.out_proj)
        self.dropout  = copy.deepcopy(soft.dropout)
        self.logic    = nn.ModuleList([HardLogicLayer(l) for l in soft.logic])
        for p in self.parameters():
            p.requires_grad = False

    def _aggregate(self, h, B, T):
        if self.sum_pool:
            pooled = h.view(B * T, self.C, self.group_size).sum(dim=-1)
            normed = (pooled - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        return self.out_proj(h).view(B, T, self.C)

    def forward(self, x):
        B, T, C = x.shape
        normed = self.norm(x).reshape(B * T, C)
        h = normed if self.no_in_proj else self.in_proj(normed)
        h = _apply_activation(h, self.activation)
        if self.binary_io:
            if self.n_bits > 1:
                h = _thermometer_ste(h, self.n_bits, training=False)
            else:
                h = (h > 0.5).to(dtype=h.dtype)
        for l in self.logic:
            h = l(h)
        contrib = self.dropout(self._aggregate(h, B, T))
        if self.skip_gate:
            contrib = self.skip_alpha * contrib
        return x + contrib


# ---------------------------------------------------------------------------
# Hybrid layer: keep original (trained) attention sublayer, replace MLP only
# ---------------------------------------------------------------------------

class HybridLogicGateGPTLayer(nn.Module):
    """Drop-in replacement for a nanoGPT Block where the attention sublayer is
    copied frozen from the trained baseline and the MLP sublayer is replaced
    by a learnable logic circuit."""

    def __init__(self, gpt_cfg, layer_idx, original_block,
                 logic_width=None, depth=1, k=4, seed=1000,
                 activation='sigmoid', conn_init_scale=0.02, gate_init_scale=0.02,
                 identity_logic=False, binary_io=False, n_bits=1, sum_pool=False,
                 no_in_proj=False, skip_gate=False):
        # no_in_proj / skip_gate kept for kwarg compatibility; not implemented for hybrid path
        assert not no_in_proj, "no_in_proj is not supported in HybridLogicGateGPTLayer"
        assert not skip_gate, "skip_gate is not supported in HybridLogicGateGPTLayer"
        super().__init__()
        C = gpt_cfg.n_embd
        self.C = C
        self.layer_idx = layer_idx
        self.logic_width = logic_width or C * 4
        self.activation = activation
        self.binary_io = binary_io
        self.n_bits    = n_bits if binary_io else 1
        self.sum_pool  = sum_pool
        bit_width      = self.logic_width * self.n_bits
        if sum_pool:
            assert bit_width % C == 0, (
                f"sum_pool requires bit_width ({bit_width}) divisible by n_embd ({C}).")
            self.group_size = bit_width // C

        # FROZEN: attention sublayer copied verbatim from trained baseline
        self.ln_1 = copy.deepcopy(original_block.ln_1)
        self.attn = copy.deepcopy(original_block.attn)
        for p in self.ln_1.parameters(): p.requires_grad = False
        for p in self.attn.parameters(): p.requires_grad = False

        # TRAINABLE: logic MLP replacement
        self.ln_2     = nn.LayerNorm(C)
        self.in_proj  = nn.Linear(C, self.logic_width)
        self.logic    = nn.ModuleList([
            LearnedLogicLayer(bit_width, bit_width, k=k,
                              seed=seed + layer_idx * 100 + i,
                              conn_init_scale=conn_init_scale,
                              gate_init_scale=gate_init_scale,
                              identity=identity_logic)
            for i in range(depth)
        ])
        if not sum_pool:
            self.out_proj = nn.Linear(bit_width, C)
        self.dropout  = nn.Dropout(gpt_cfg.dropout)
        self.use_ste  = False  # straight-through estimator toggle

    def set_temperature(self, t):
        for l in self.logic: l.set_temperature(t)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _aggregate(self, h, B, T):
        if self.sum_pool:
            pooled = h.view(B * T, self.C, self.group_size).sum(dim=-1)
            normed = (pooled - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        return self.out_proj(h).view(B, T, self.C)

    def forward(self, x, hard=False):
        # Attention sublayer (frozen, identical to original)
        x = x + self.attn(self.ln_1(x))
        # Logic MLP replacement
        B, T, C = x.shape
        h = _apply_activation(self.in_proj(self.ln_2(x).reshape(B * T, C)), self.activation)
        if self.binary_io:
            if self.n_bits > 1:
                h = _thermometer_ste(h, self.n_bits, self.training)
            else:
                h = _binarize_ste(h)
        ste = self.use_ste and not hard
        for l in self.logic:
            h = l(h, hard=hard, ste=ste)
        return x + self.dropout(self._aggregate(h, B, T))


class HardHybridLogicGateGPTLayer(nn.Module):
    """Hard-snapped version of HybridLogicGateGPTLayer. Attention stays
    continuous and identical to the original; only the logic part is discretised."""

    def __init__(self, soft: HybridLogicGateGPTLayer):
        super().__init__()
        self.layer_idx  = soft.layer_idx
        self.activation = soft.activation
        self.binary_io  = getattr(soft, 'binary_io', False)
        self.n_bits     = getattr(soft, 'n_bits', 1)
        self.sum_pool   = getattr(soft, 'sum_pool', False)
        self.C          = getattr(soft, 'C', None)
        self.group_size = getattr(soft, 'group_size', None)
        self.ln_1     = copy.deepcopy(soft.ln_1)
        self.attn     = copy.deepcopy(soft.attn)
        self.ln_2     = copy.deepcopy(soft.ln_2)
        self.in_proj  = copy.deepcopy(soft.in_proj)
        if not self.sum_pool:
            self.out_proj = copy.deepcopy(soft.out_proj)
        self.dropout  = copy.deepcopy(soft.dropout)
        self.logic    = nn.ModuleList([HardLogicLayer(l) for l in soft.logic])
        for p in self.parameters():
            p.requires_grad = False

    def _aggregate(self, h, B, T):
        if self.sum_pool:
            pooled = h.view(B * T, self.C, self.group_size).sum(dim=-1)
            normed = (pooled - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        return self.out_proj(h).view(B, T, self.C)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        B, T, C = x.shape
        h = _apply_activation(self.in_proj(self.ln_2(x).reshape(B * T, C)), self.activation)
        if self.binary_io:
            if self.n_bits > 1:
                h = _thermometer_ste(h, self.n_bits, training=False)
            else:
                h = (h > 0.5).to(dtype=h.dtype)
        for l in self.logic:
            h = l(h)
        return x + self.dropout(self._aggregate(h, B, T))

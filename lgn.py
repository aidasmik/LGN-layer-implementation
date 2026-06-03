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
    # Width of the trained Linear path (only used when no_in_proj=False; no-op in aggressive).
    width_mult: int = 2
    depth: int = 1
    k: int = 4
    activation: str = 'sigmoid'
    conn_init_scale: float = 0.02
    gate_init_scale: float = 0.02
    hybrid_layers: list = field(default_factory=list)  # layers that keep frozen attention
    identity_logic: bool = False                        # ablation: LGN body = pass-through

    # AGGRESSIVE SETUP (the honest default — no trained float transform around the gates)
    binary_io: bool = True
    n_bits: int = 8
    sum_pool: bool = True
    no_in_proj: bool = True
    # Learnable per-channel affine on the sum_pool output (cheap residual-stat matching).
    learn_pool: bool = False
    # Fixed causal token shift: each position sees [x[t-K]..x[t]] — a local cross-token
    # receptive field for the pointwise LGN. K=0 disables. (The single mechanism, with
    # hybrid/selective, that actually raises accuracy.)
    token_shift: int = 0

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
    ste: bool = False               # straight-through estimator (forward hard, backward soft)
    # CAGE — Align Forward Adapt Backward (2026, arxiv 2603.14157).
    # Implies STE (hard forward) and adapts the BACKWARD-pass softmax temperature τ_b
    # based on an EMA of average commitment confidence. Closes the discretization gap
    # by construction (forward = inference). Schedule: τ_b in [tau_min, tau_max] linearly
    # interpolated by EMA confidence c_ema (1/K-1.0 -> tau_max-tau_min).
    cage: bool = False
    cage_tau_max: float = 3.0
    cage_tau_min: float = 0.5
    cage_ema:     float = 0.99
    # Direct (from-scratch) training: anneal temperature DURING fine-tune on LM loss
    # instead of during imitation. Lets the LGN learn its own solution, not imitate MLP.
    anneal_in_finetune: bool = False
    # Curriculum: decaying MSE-to-MLP term blended into fine-tune (weight*1.0 -> 0).
    # 0 = pure LM loss. Single-layer fine-tune only (heatmap).
    ft_imit_weight: float = 0.0
    # NOTE: 'freeze_unreplaced' was removed — the base model is ALWAYS frozen by
    # _make_logic_model / _add_logic_layer (only LGN layer params get requires_grad=True),
    # so the flag was redundant. All degradation numbers are already 'pure LGN' measurements.
    # Joint polish: after sequential scaling, fine-tune ALL LGN layers together to
    # coordinate them (fixes greedy myopia). 0 = disabled.
    joint_polish_steps: int = 0
    # System-level distillation during joint polish: KL of student logits to the
    # original transformer's logits. Global coordination signal (vs per-layer MLP). 0 = LM only.
    joint_polish_kl_weight: float = 0.0

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


def apply_token_shift(normed, taps):
    """Channel-aligned causal token shift.

    normed: (B, T, C). taps: list of positive ints, e.g. [1,2] for token_shift K=2.
    Returns (B, T, C*(len(taps)+1)) where contiguous blocks of (len(taps)+1) channels
    are ONE channel's time history [t, t-tap0, ...]. First tap positions zeroed (causal).
    """
    if not taps:
        return normed
    B, T, C = normed.shape
    parts = [normed]
    for tap in taps:
        shifted = torch.roll(normed, shifts=tap, dims=1)
        shifted[:, :tap] = 0
        parts.append(shifted)
    return torch.stack(parts, dim=-1).reshape(B, T, C * (len(taps) + 1))


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

    Forward: hard binary thermometer.
    Backward: TRUE identity STE. Each bit contributes gradient 1/n_bits w.r.t. h,
    summed across bits = 1. Previously used a clamped-ramp surrogate where total
    gradient scaled with h (vanishing near 0, exploding near 1) - that broke STE
    semantics and starved low-magnitude inputs of learning signal."""
    *prefix, D = h.shape
    levels = torch.linspace(1.0 / (n_bits + 1), n_bits / (n_bits + 1), n_bits,
                            device=h.device, dtype=h.dtype)
    expanded = h.unsqueeze(-1).expand(*prefix, D, n_bits)
    hard = (expanded > levels).to(dtype=h.dtype)
    if training:
        # Identity STE: backward grad = d(out_total)/dh = 1. Each of n_bits outputs
        # contributes h/n_bits in the soft path, so summed gradient w.r.t. h = 1.
        soft = expanded / n_bits
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
        # CAGE (Align Forward Adapt Backward, 2026): independent backward-pass temperature.
        # None = use self.temperature for backward (vanilla STE). When set, softmax(logits/τ_b)
        # is used in the STE backward path while forward stays hard argmax.
        self.backward_temp = None
        self.identity = identity
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        self.register_buffer('cand_a', torch.randint(0, in_dim, (out_dim, k), generator=g))
        self.register_buffer('cand_b', torch.randint(0, in_dim, (out_dim, k), generator=g))
        self.conn_logits_a = nn.Parameter(torch.randn(out_dim, k) * conn_init_scale)
        self.conn_logits_b = nn.Parameter(torch.randn(out_dim, k) * conn_init_scale)
        self.gate_logits = nn.Parameter(torch.randn(out_dim, 16) * gate_init_scale)

    def set_temperature(self, t):
        self.temperature = float(t)

    def set_backward_temp(self, t):
        """CAGE: set independent backward STE temperature (None = use self.temperature)."""
        self.backward_temp = None if t is None else float(t)

    def _sm(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)

    def _sm_back(self, logits):
        """Backward-pass softmax: uses backward_temp if set (CAGE), else self.temperature."""
        t = self.backward_temp if self.backward_temp is not None else self.temperature
        return F.softmax(logits / t, dim=-1)

    @torch.no_grad()
    def commitment(self):
        """CAGE: average max-softmax across this layer's logits, used to update τ_b.
        Higher = more committed to a single choice."""
        gate_c = float(self._sm(self.gate_logits).max(dim=-1).values.mean())
        conn_c = 0.5 * (float(self._sm(self.conn_logits_a).max(dim=-1).values.mean()) +
                        float(self._sm(self.conn_logits_b).max(dim=-1).values.mean()))
        return 0.5 * (gate_c + conn_c)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        def ent(logits):
            p = self._sm(logits)
            return -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        conn_term = conn_w * (ent(self.conn_logits_a) + ent(self.conn_logits_b))
        return conn_term + gate_w * ent(self.gate_logits)

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
            # STE (vanilla or CAGE if backward_temp set): forward=hard, backward=soft.
            soft = self._sm_back(logits)
            hard_w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
            w = soft + (hard_w - soft).detach()
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
            soft = self._sm_back(self.gate_logits)
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
                 no_in_proj=False, learn_pool=False, token_shift=0):
        super().__init__()
        C = gpt_cfg.n_embd
        self.C = C
        self.layer_idx = layer_idx
        self.logic_width = logic_width or C * 4
        self.activation = activation
        self.binary_io = binary_io
        self.n_bits    = n_bits if binary_io else 1
        self.sum_pool  = sum_pool
        self.learn_pool = learn_pool
        self.token_shift = token_shift
        # Causal token shift: each position sees [x[t-K]..x[t]]. eff_C = (K+1)*C.
        self._taps = list(range(1, token_shift + 1)) if token_shift > 0 else []
        eff_C = C * (len(self._taps) + 1)
        self.eff_C = eff_C
        self.no_in_proj = no_in_proj
        if no_in_proj:
            assert binary_io, "no_in_proj requires binary_io"
            bit_width = eff_C * self.n_bits
        else:
            bit_width = self.logic_width * self.n_bits
        if sum_pool:
            assert binary_io, "sum_pool requires binary_io"
            assert bit_width % C == 0, (
                f"sum_pool requires bit_width ({bit_width}) divisible by n_embd ({C}).")
            self.group_size = bit_width // C
            if learn_pool:
                # Init to match fixed centering: (pooled - g/2)/(g/2) = pooled*(2/g) - 1
                self.pool_scale = nn.Parameter(torch.full((C,), 2.0 / self.group_size))
                self.pool_shift = nn.Parameter(torch.full((C,), -1.0))
        self.norm = nn.LayerNorm(C)
        if not no_in_proj:
            self.in_proj = nn.Linear(eff_C, self.logic_width)
        self.logic = nn.ModuleList([
            LearnedLogicLayer(bit_width, bit_width, k=k, seed=seed + layer_idx * 100 + i,
                              conn_init_scale=conn_init_scale, gate_init_scale=gate_init_scale,
                              identity=identity_logic)
            for i in range(depth)
        ])
        if not sum_pool:
            self.out_proj = nn.Linear(bit_width, C)
        self.dropout = nn.Dropout(gpt_cfg.dropout)
        self.use_ste = False  # STE toggle (set during fine-tune / CAGE)

    def set_temperature(self, t):
        for l in self.logic: l.set_temperature(t)

    def set_backward_temp(self, t):
        """CAGE: propagate independent backward STE temperature to all sublayers."""
        for l in self.logic: l.set_backward_temp(t)

    @torch.no_grad()
    def commitment(self):
        """CAGE: average commitment confidence across this block's sublayers."""
        if not self.logic:
            return 1.0
        return sum(l.commitment() for l in self.logic) / len(self.logic)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _aggregate(self, h, B, T):
        """(B*T, bit_width) → (B, T, C). sum_pool: fixed group-sum (+ optional learn_pool
        affine); else trained out_proj Linear."""
        if self.sum_pool:
            grp = h.view(B * T, self.C, self.group_size)
            if self.learn_pool:
                normed = grp.sum(dim=-1) * self.pool_scale + self.pool_shift
            else:
                normed = (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        return self.out_proj(h).view(B, T, self.C)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.eff_C)
        return self.in_proj(normed_btx.reshape(B * T, self.eff_C))

    def forward(self, x, hard=False):
        B, T, C = x.shape
        normed = apply_token_shift(self.norm(x), self._taps)
        h = self._apply_in_proj(normed, B, T)
        h = _apply_activation(h, self.activation)
        if self.binary_io:
            h = _thermometer_ste(h, self.n_bits, self.training) if self.n_bits > 1 else _binarize_ste(h)
        ste = self.use_ste and not hard
        for l in self.logic:
            h = l(h, hard=hard, ste=ste)
        return x + self.dropout(self._aggregate(h, B, T))

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
        self.binary_io  = soft.binary_io
        self.n_bits     = soft.n_bits
        self.sum_pool   = soft.sum_pool
        self.learn_pool = soft.learn_pool
        self.no_in_proj = soft.no_in_proj
        self.C          = soft.C
        self.eff_C      = soft.eff_C
        self.token_shift = soft.token_shift
        self._taps      = soft._taps
        self.group_size = getattr(soft, 'group_size', None)
        if self.sum_pool and self.learn_pool:
            self.register_buffer('pool_scale', soft.pool_scale.detach().clone())
            self.register_buffer('pool_shift', soft.pool_shift.detach().clone())
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
            grp = h.view(B * T, self.C, self.group_size)
            if self.learn_pool:
                normed = grp.sum(dim=-1) * self.pool_scale + self.pool_shift
            else:
                normed = (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        return self.out_proj(h).view(B, T, self.C)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.eff_C)
        return self.in_proj(normed_btx.reshape(B * T, self.eff_C))

    def forward(self, x):
        B, T, C = x.shape
        normed = apply_token_shift(self.norm(x), self._taps)
        h = self._apply_in_proj(normed, B, T)
        h = _apply_activation(h, self.activation)
        if self.binary_io:
            h = _thermometer_ste(h, self.n_bits, training=False) if self.n_bits > 1 else (h > 0.5).to(dtype=h.dtype)
        for l in self.logic:
            h = l(h)
        return x + self.dropout(self._aggregate(h, B, T))


# ---------------------------------------------------------------------------
# Hybrid layer: keep original (trained) attention sublayer, replace MLP only
# ---------------------------------------------------------------------------

class HybridLogicGateGPTLayer(nn.Module):
    """Drop-in replacement for a nanoGPT Block where the attention sublayer is
    copied FROZEN from the trained baseline and the MLP sublayer is replaced
    by a learnable logic circuit. Now supports ALL aggressive flags so the MLP
    side can be truly aggressive (binary_io + no_in_proj + sum_pool + ...)."""

    def __init__(self, gpt_cfg, layer_idx, original_block,
                 logic_width=None, depth=1, k=4, seed=1000,
                 activation='sigmoid', conn_init_scale=0.02, gate_init_scale=0.02,
                 identity_logic=False, binary_io=False, n_bits=1, sum_pool=False,
                 no_in_proj=False, learn_pool=False, token_shift=0):
        super().__init__()
        C = gpt_cfg.n_embd
        self.C = C
        self.layer_idx = layer_idx
        self.logic_width = logic_width or C * 4
        self.activation = activation
        self.binary_io = binary_io
        self.n_bits    = n_bits if binary_io else 1
        self.sum_pool  = sum_pool
        self.learn_pool = learn_pool
        self.token_shift = token_shift
        self._taps = list(range(1, token_shift + 1)) if token_shift > 0 else []
        eff_C = C * (len(self._taps) + 1)
        self.eff_C = eff_C
        self.no_in_proj = no_in_proj
        if no_in_proj:
            assert binary_io, "no_in_proj requires binary_io"
            bit_width = eff_C * self.n_bits
        else:
            bit_width = self.logic_width * self.n_bits
        if sum_pool:
            assert binary_io, "sum_pool requires binary_io"
            assert bit_width % C == 0, (
                f"sum_pool requires bit_width ({bit_width}) divisible by n_embd ({C}).")
            self.group_size = bit_width // C
            if learn_pool:
                self.pool_scale = nn.Parameter(torch.full((C,), 2.0 / self.group_size))
                self.pool_shift = nn.Parameter(torch.full((C,), -1.0))

        # FROZEN: attention sublayer copied verbatim from the trained baseline.
        self.ln_1 = copy.deepcopy(original_block.ln_1)
        self.attn = copy.deepcopy(original_block.attn)
        for p in self.ln_1.parameters(): p.requires_grad = False
        for p in self.attn.parameters(): p.requires_grad = False
        self.ln_1.eval(); self.attn.eval()

        # TRAINABLE: LGN MLP replacement (same shape as LogicGateGPTLayer)
        self.ln_2 = nn.LayerNorm(C)
        if not no_in_proj:
            self.in_proj = nn.Linear(eff_C, self.logic_width)
        self.logic = nn.ModuleList([
            LearnedLogicLayer(bit_width, bit_width, k=k, seed=seed + layer_idx * 100 + i,
                              conn_init_scale=conn_init_scale, gate_init_scale=gate_init_scale,
                              identity=identity_logic)
            for i in range(depth)
        ])
        if not sum_pool:
            self.out_proj = nn.Linear(bit_width, C)
        self.dropout = nn.Dropout(gpt_cfg.dropout)
        self.use_ste = False

    def train(self, mode=True):
        super().train(mode)
        self.ln_1.eval(); self.attn.eval()
        return self

    def set_temperature(self, t):
        for l in self.logic: l.set_temperature(t)

    def set_backward_temp(self, t):
        """CAGE: propagate independent backward STE temperature to all sublayers."""
        for l in self.logic: l.set_backward_temp(t)

    @torch.no_grad()
    def commitment(self):
        if not self.logic:
            return 1.0
        return sum(l.commitment() for l in self.logic) / len(self.logic)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _aggregate(self, h, B, T):
        if self.sum_pool:
            grp = h.view(B * T, self.C, self.group_size)
            if self.learn_pool:
                normed = grp.sum(dim=-1) * self.pool_scale + self.pool_shift
            else:
                normed = (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        return self.out_proj(h).view(B, T, self.C)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.eff_C)
        return self.in_proj(normed_btx.reshape(B * T, self.eff_C))

    def forward(self, x, hard=False):
        x = x + self.attn(self.ln_1(x))          # FROZEN attention
        B, T, C = x.shape
        normed = apply_token_shift(self.ln_2(x), self._taps)
        h = self._apply_in_proj(normed, B, T)
        h = _apply_activation(h, self.activation)
        if self.binary_io:
            h = _thermometer_ste(h, self.n_bits, self.training) if self.n_bits > 1 else _binarize_ste(h)
        ste = self.use_ste and not hard
        for l in self.logic:
            h = l(h, hard=hard, ste=ste)
        return x + self.dropout(self._aggregate(h, B, T))


class HardHybridLogicGateGPTLayer(nn.Module):
    """Hard-snapped HybridLogicGateGPTLayer. Attention stays continuous and identical
    to the original; only the LGN MLP is discretised. Supports all aggressive flags."""

    def __init__(self, soft: HybridLogicGateGPTLayer):
        super().__init__()
        self.layer_idx  = soft.layer_idx
        self.activation = soft.activation
        self.binary_io  = soft.binary_io
        self.n_bits     = soft.n_bits
        self.sum_pool   = soft.sum_pool
        self.learn_pool = soft.learn_pool
        self.no_in_proj = soft.no_in_proj
        self.C          = soft.C
        self.eff_C      = soft.eff_C
        self.token_shift = soft.token_shift
        self._taps      = soft._taps
        self.group_size = getattr(soft, 'group_size', None)
        if self.sum_pool and self.learn_pool:
            self.register_buffer('pool_scale', soft.pool_scale.detach().clone())
            self.register_buffer('pool_shift', soft.pool_shift.detach().clone())
        self.ln_1 = copy.deepcopy(soft.ln_1)
        self.attn = copy.deepcopy(soft.attn)
        self.ln_2 = copy.deepcopy(soft.ln_2)
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
            grp = h.view(B * T, self.C, self.group_size)
            if self.learn_pool:
                normed = grp.sum(dim=-1) * self.pool_scale + self.pool_shift
            else:
                normed = (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        return self.out_proj(h).view(B, T, self.C)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.eff_C)
        return self.in_proj(normed_btx.reshape(B * T, self.eff_C))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        B, T, C = x.shape
        normed = apply_token_shift(self.ln_2(x), self._taps)
        h = self._apply_in_proj(normed, B, T)
        h = _apply_activation(h, self.activation)
        if self.binary_io:
            h = _thermometer_ste(h, self.n_bits, training=False) if self.n_bits > 1 else (h > 0.5).to(dtype=h.dtype)
        for l in self.logic:
            h = l(h)
        return x + self.dropout(self._aggregate(h, B, T))

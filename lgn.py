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
    # Input-Wise Parametrization (Light DLGN 2025): 4 weights per gate instead of 16 logits.
    # Solves vanishing gradients (no self-cancelling negation pairs) and 4x fewer params.
    iwp: bool = False
    # Fixed causal token shift: before each LGN block, concatenate K past positions.
    # Gives pointwise LGN a local cross-token receptive field. K=0 disables; K=2 means each
    # position sees [x[t-2], x[t-1], x[t]] (3x channel width into LGN). MLP-Mixer style.
    token_shift: int = 0
    # Dilated token shift (Idea E): explicit tap list overrides contiguous token_shift.
    # e.g. [1,2,4,8,16] gives an exponential look-back span of 16 tokens with 5 taps.
    shift_taps: list = None
    # Learnable per-channel affine on the sum_pool output (cheap compatibility:
    # lets each LGN layer match residual-stream statistics). Init = fixed centering.
    learn_pool: bool = False
    # Learnable per-BIT weights in the group-sum (richer output aggregation than
    # learn_pool's per-channel affine). Init = uniform (matches fixed centering).
    pool_weighted: bool = False
    # ------------------------------------------------------------------
    # Random interconnect after layer N (depth must be > 1).
    # 999 = all layers learnable; 1 = only first LGN sublayer learns connections,
    # remaining (depth-1) sublayers have FIXED random connections (gates still
    # learnable). Reservoir-computing analogy: 1st layer = encoder, rest = reservoir.
    # ------------------------------------------------------------------
    random_from: int = 999
    # ------------------------------------------------------------------
    # Temporal Conv1d projections (causal, kernel size K, stride=1) instead of
    # Linear in_proj / out_proj. Requires --no-no_in_proj / --no-sum_pool.
    # 0 = use Linear (or skip if no_in_proj/sum_pool). K>0 = causal Conv1d.
    # Allows cross-token mixing via the conv kernel.
    # ------------------------------------------------------------------
    conv_in_k:  int = 0
    conv_out_k: int = 0
    # ------------------------------------------------------------------
    # RDDLGN binary regularization (2025): adds λ·h(1-h) to the loss on
    # post-sigmoid pre-binarization activations. Encourages bits to commit
    # to {0,1} smoothly during training, reducing hard-snap gap.
    # ------------------------------------------------------------------
    bin_reg: bool = False

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
    gumbel_ste: bool = False        # Gumbel-STE training (Mind the Gap 2025)
    # CAGE — Align Forward Adapt Backward (2026, arxiv 2603.14157).
    # Implies STE (hard forward) and adapts the BACKWARD-pass softmax temperature τ_b
    # based on an EMA of average commitment confidence. Closes the discretization gap
    # by construction (forward = inference). Schedule: τ_b in [tau_min, tau_max] linearly
    # interpolated by EMA confidence c_ema (1/K-1.0 -> tau_max-tau_min).
    cage: bool = False
    cage_tau_max: float = 3.0
    cage_tau_min: float = 0.5
    cage_ema:     float = 0.99
    # Binary regularization (RDDLGN, 2025): loss term  λ · h(1-h) on post-sigmoid pre-binarization.
    bin_reg_weight: float = 0.0
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
    """Channel-aligned causal (dilated) token shift.

    normed: (B, T, C). taps: list of positive ints, e.g. [1,2] (contiguous) or
    [1,2,4,8,16] (dilated). Returns (B, T, C*(len(taps)+1)) where contiguous blocks
    of (len(taps)+1) channels are ONE channel's time history [t, t-tap0, t-tap1, ...].
    Wrap-around (first tap positions) zeroed to preserve causality.
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


def resolve_taps(token_shift, shift_taps):
    """Resolve the effective tap list: explicit shift_taps overrides contiguous token_shift."""
    if shift_taps:
        return list(shift_taps)
    if token_shift and token_shift > 0:
        return list(range(1, token_shift + 1))
    return []


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
                 conn_init_scale=0.02, gate_init_scale=0.02, identity=False, iwp=False,
                 freeze_conn=False):
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
        self.iwp = iwp  # Input-Wise Parametrization (Light DLGN 2025)
        # freeze_conn=True: connections are FIXED random (no learnable conn_logits).
        # Used for reservoir-style depth-stacked LGN: 1st layer learns connections,
        # subsequent layers have random fixed connections (gates still learnable).
        self.freeze_conn = freeze_conn
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        # When freeze_conn: only one candidate per output (k_eff=1) is used; we still
        # store k for shape compatibility but only column 0 ever gets read.
        self.register_buffer('cand_a', torch.randint(0, in_dim, (out_dim, k), generator=g))
        self.register_buffer('cand_b', torch.randint(0, in_dim, (out_dim, k), generator=g))
        if not freeze_conn:
            self.conn_logits_a = nn.Parameter(torch.randn(out_dim, k) * conn_init_scale)
            self.conn_logits_b = nn.Parameter(torch.randn(out_dim, k) * conn_init_scale)
        if iwp:
            # 4 raw weights per gate: omega_ij = sigmoid(Omega_ij) is the output for input (i,j).
            # Heavy-tail init per Light DLGN: |Omega| ~ N(mu=1.2, sigma=0.25), committed but
            # NOT saturated (sigmoid'(1.2) ~ 0.177, vs 0.007 at |Omega|=5). Sign chosen to give
            # residual pass-through-A: w(0,*)=0 (negative), w(1,*)=1 (positive).
            sign = torch.tensor([-1.0, -1.0, 1.0, 1.0]).expand(out_dim, 4)
            mag  = 1.2 + torch.randn(out_dim, 4) * 0.25
            self.gate_omega = nn.Parameter(sign * mag)
        else:
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

        Returns commitment of gate logits (and connection logits if learnable),
        averaged together. Higher = more committed to a single choice."""
        if self.iwp:
            w = torch.sigmoid(self.gate_omega)
            gate_c = float(torch.maximum(w, 1 - w).mean())
        else:
            gate_c = float(self._sm(self.gate_logits).max(dim=-1).values.mean())
        if self.freeze_conn:
            return gate_c  # connections fixed, only gate matters
        conn_c = 0.5 * (float(self._sm(self.conn_logits_a).max(dim=-1).values.mean()) +
                        float(self._sm(self.conn_logits_b).max(dim=-1).values.mean()))
        return 0.5 * (gate_c + conn_c)

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        def ent(logits):
            p = self._sm(logits)
            return -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        # Connection entropy only when connections are learnable.
        if self.freeze_conn:
            conn_term = torch.zeros((), device=self.cand_a.device)
        else:
            conn_term = conn_w * (ent(self.conn_logits_a) + ent(self.conn_logits_b))
        if self.iwp:
            # IWP: encourage each sigmoid(omega) to commit to 0 or 1 (binary cross-entropy with itself).
            w = torch.sigmoid(self.gate_omega)
            iwp_ent = -(w * (w + 1e-8).log() + (1 - w) * (1 - w + 1e-8).log()).mean()
            return conn_term + gate_w * iwp_ent
        return conn_term + gate_w * ent(self.gate_logits)

    @torch.no_grad()
    def sharpness(self):
        # Frozen-connection layers report conn=1.0 by convention (one-hot fixed).
        conn_a_sharp = 1.0 if self.freeze_conn else float(self._sm(self.conn_logits_a).max(dim=-1).values.mean())
        conn_b_sharp = 1.0 if self.freeze_conn else float(self._sm(self.conn_logits_b).max(dim=-1).values.mean())
        if self.iwp:
            w = torch.sigmoid(self.gate_omega)
            commitment = torch.maximum(w, 1 - w).mean()  # 0.5 = uncertain, 1.0 = fully committed
            return {
                'conn_a': conn_a_sharp,
                'conn_b': conn_b_sharp,
                'gate':   float(commitment),
            }
        return {
            'conn_a': conn_a_sharp,
            'conn_b': conn_b_sharp,
            'gate':   float(self._sm(self.gate_logits).max(dim=-1).values.mean()),
        }

    def _gumbel(self, shape, device, dtype):
        # Standard Gumbel sample with numerical-safety clamp on BOTH ends of u in (0,1)
        eps = 1e-6
        u = torch.rand(shape, device=device, dtype=dtype).clamp(eps, 1.0 - eps)
        return -torch.log(-torch.log(u))

    def _select_frozen(self, x, cand):
        """Connection-frozen path: gather a single fixed input per output gate (column 0)."""
        return x[:, cand[:, 0]]

    def _select(self, x, cand, logits, hard, ste=False, gumbel=False):
        gathered = x[:, cand]
        if gumbel and not hard:
            # Gumbel-STE: stochastic discrete sample with soft gradient (Mind the Gap, 2025)
            noisy = logits + self._gumbel(logits.shape, logits.device, logits.dtype)
            soft = F.softmax(noisy / self.temperature, dim=-1)
            hard_w = F.one_hot(noisy.argmax(dim=-1), self.k).to(dtype=x.dtype)
            w = soft + (hard_w - soft).detach()
        elif ste:
            # STE (vanilla or CAGE if backward_temp set): forward hard, backward soft.
            # CAGE (2026): _sm_back uses τ_b (adapted externally based on commitment).
            soft = self._sm_back(logits)
            hard_w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
            w = soft + (hard_w - soft).detach()  # forward=hard, backward=soft
        elif hard:
            w = F.one_hot(logits.argmax(dim=-1), self.k).to(dtype=x.dtype)
        else:
            w = self._sm(logits)
        return (gathered * w).sum(dim=-1)

    def forward(self, x, hard=False, ste=False, gumbel=False):
        if self.identity:
            return x  # ablation: bypass all logic computation
        if self.freeze_conn:
            a = self._select_frozen(x, self.cand_a)
            b = self._select_frozen(x, self.cand_b)
        else:
            a = self._select(x, self.cand_a, self.conn_logits_a, hard, ste=ste, gumbel=gumbel)
            b = self._select(x, self.cand_b, self.conn_logits_b, hard, ste=ste, gumbel=gumbel)
        if self.iwp:
            # Input-Wise Parametrization: g(a,b) = (1-a)(1-b)w00 + (1-a)b*w01 + a(1-b)*w10 + ab*w11
            # Hard mode: round sigmoid(Omega) at 0.5 (Omega>0 -> 1, else 0).
            if hard:
                w = (self.gate_omega > 0).to(dtype=a.dtype)
            elif ste:
                w_soft = torch.sigmoid(self.gate_omega)
                w_hard = (self.gate_omega > 0).to(dtype=a.dtype)
                w = w_soft + (w_hard - w_soft).detach()  # forward hard, backward soft
            else:
                w = torch.sigmoid(self.gate_omega)
            w00, w01, w10, w11 = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
            return (1 - a) * (1 - b) * w00 + (1 - a) * b * w01 + a * (1 - b) * w10 + a * b * w11
        gates = diff_logic_gates(a, b)
        if gumbel and not hard:
            noisy = self.gate_logits + self._gumbel(self.gate_logits.shape, self.gate_logits.device, self.gate_logits.dtype)
            soft = F.softmax(noisy / self.temperature, dim=-1)
            hard_gp = F.one_hot(noisy.argmax(dim=-1), 16).to(dtype=x.dtype)
            gp = soft + (hard_gp - soft).detach()
        elif ste:
            # CAGE-aware: backward uses τ_b if set, else self.temperature
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
                 no_in_proj=False, skip_gate=False, learn_pool=False, pool_weighted=False,
                 iwp=False, token_shift=0, random_from=999, conv_in_k=0, conv_out_k=0,
                 bin_reg=False, shift_taps=None):
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
        self.pool_weighted = pool_weighted
        self.iwp = iwp
        self.token_shift = token_shift
        # Dilated token shift (Idea E): explicit tap list overrides contiguous token_shift.
        # taps=[1,2] is identical to token_shift=2; taps=[1,2,4,8,16] is a wide dilated span.
        self._taps = resolve_taps(token_shift, shift_taps)
        self.shift_taps = shift_taps
        # Effective per-position width into the LGN body: (n_taps + 1) * C.
        eff_C = C * (len(self._taps) + 1)
        self.eff_C = eff_C
        self.no_in_proj = no_in_proj
        self.conv_in_k  = conv_in_k
        self.conv_out_k = conv_out_k
        self.random_from = random_from
        self.bin_reg    = bin_reg  # RDDLGN-style binary commitment regularization
        self._bin_reg_buf = None
        # When no_in_proj: LGN operates directly on n_embd binarized features (× n_bits).
        # With token_shift>0, the per-position channel width is eff_C = C*(K+1).
        # ──────────────────────────────────────────────────────────────────────
        # WARNING: in aggressive default (no_in_proj=True), width_mult / edge_width_mult
        # are NO-OPs. logic_width = n_embd * width_mult is computed but unused; the LGN
        # width is determined ONLY by n_embd, n_bits, token_shift, depth and k. To control
        # LGN capacity in aggressive mode use --n_bits, --token_shift, --depth, --k.
        # ──────────────────────────────────────────────────────────────────────
        if conv_in_k > 0:
            assert not no_in_proj, "conv_in_k requires no_in_proj=False"
        if conv_out_k > 0:
            assert not sum_pool,   "conv_out_k requires sum_pool=False"
        if no_in_proj:
            assert binary_io, "no_in_proj requires binary_io"
            bit_width = eff_C * self.n_bits
        else:
            bit_width = self.logic_width * self.n_bits
        if sum_pool:
            assert binary_io, "sum_pool centering assumes binary {0,1} bits; requires binary_io"
            assert bit_width % C == 0, (
                f"sum_pool requires bit_width ({bit_width}) divisible by n_embd ({C}). "
                f"Increase --width_mult or --n_bits.")
            self.group_size = bit_width // C
            if pool_weighted:
                # Per-bit learnable weights + per-channel shift. Init matches fixed centering.
                self.pool_w     = nn.Parameter(torch.full((C, self.group_size), 2.0 / self.group_size))
                self.pool_shift = nn.Parameter(torch.full((C,), -1.0))
            elif learn_pool:
                # Init to match the fixed centering: (pooled - g/2)/(g/2) = pooled*(2/g) - 1
                self.pool_scale = nn.Parameter(torch.full((C,), 2.0 / self.group_size))
                self.pool_shift = nn.Parameter(torch.full((C,), -1.0))
        self.norm     = nn.LayerNorm(C)
        if not no_in_proj:
            if conv_in_k > 0:
                # Causal Conv1d (T dim): each output position depends on inputs [t-K+1..t].
                # Channels: eff_C -> logic_width (same role as Linear in_proj).
                self.in_proj = nn.Conv1d(eff_C, self.logic_width, kernel_size=conv_in_k, padding=0)
            else:
                self.in_proj = nn.Linear(eff_C, self.logic_width)
        self.logic    = nn.ModuleList([
            LearnedLogicLayer(bit_width, bit_width, k=k,
                              seed=seed + layer_idx * 100 + i,
                              conn_init_scale=conn_init_scale,
                              gate_init_scale=gate_init_scale,
                              identity=identity_logic, iwp=iwp,
                              freeze_conn=(i >= random_from))
            for i in range(depth)
        ])
        if not sum_pool:
            if conv_out_k > 0:
                # Causal Conv1d: bit_width -> C with cross-token kernel.
                self.out_proj = nn.Conv1d(bit_width, C, kernel_size=conv_out_k, padding=0)
            else:
                self.out_proj = nn.Linear(bit_width, C)
        self.dropout  = nn.Dropout(gpt_cfg.dropout)
        self.use_ste  = False  # straight-through estimator toggle (set during fine-tune)
        self.use_gumbel = False  # Gumbel-STE toggle (Mind the Gap 2025); set during training
        # Learnable scalar gating the LGN contribution to residual
        self.skip_gate = skip_gate
        if skip_gate:
            self.skip_alpha = nn.Parameter(torch.ones(1))

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

    def bin_reg_loss(self):
        """RDDLGN-style binary regularization on the pre-binarization activations.

        Encourages each scalar h ∈ [0,1] (post-sigmoid, pre-thermometer) to commit
        to 0 or 1 via the term h(1-h). Buffer set during forward when bin_reg is
        enabled. Returns 0 if not enabled or no forward yet this step."""
        buf = getattr(self, '_bin_reg_buf', None)
        if buf is None:
            return torch.zeros((), device=self.norm.weight.device)
        return buf

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _aggregate(self, h, B, T):
        """Convert (B*T, bit_width) → (B, T, C) for residual addition.

        With sum_pool: fixed group-sum, no trained parameters between LGN and residual.
        Without sum_pool: trained out_proj (Linear or causal Conv1d)."""
        if self.sum_pool:
            grp = h.view(B * T, self.C, self.group_size)
            if self.pool_weighted:
                normed = (grp * self.pool_w).sum(dim=-1) + self.pool_shift
            elif self.learn_pool:
                normed = grp.sum(dim=-1) * self.pool_scale + self.pool_shift
            else:
                # Center & scale so output ~ [-1, 1] (assumes uniform bit usage).
                normed = (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        if self.conv_out_k > 0:
            # h: (B*T, bit_width) -> (B, bit_width, T), left-pad K-1 (causal), conv, slice
            h2 = h.view(B, T, -1).transpose(1, 2)
            h2 = F.pad(h2, (self.conv_out_k - 1, 0))
            out = self.out_proj(h2)  # (B, C, T)
            return out.transpose(1, 2).contiguous()
        return self.out_proj(h).view(B, T, self.C)

    def _apply_in_proj(self, normed_btx, B, T):
        """normed_btx: (B, T, eff_C). Returns (B*T, logic_width or eff_C)."""
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.eff_C)
        if self.conv_in_k > 0:
            # Causal Conv1d on T: (B, eff_C, T) -> (B, logic_width, T) -> flatten
            inp = normed_btx.transpose(1, 2)
            inp = F.pad(inp, (self.conv_in_k - 1, 0))
            out = self.in_proj(inp)
            return out.transpose(1, 2).reshape(B * T, self.logic_width)
        return self.in_proj(normed_btx.reshape(B * T, self.eff_C))

    def forward(self, x, hard=False):
        B, T, C = x.shape
        normed = self.norm(x)  # (B, T, C)
        # Causal (dilated) token shift via channel-aligned layout. See apply_token_shift.
        normed = apply_token_shift(normed, self._taps)
        # in_proj: Linear OR causal Conv1d OR no-op (no_in_proj)
        h = self._apply_in_proj(normed, B, T)
        h = _apply_activation(h, self.activation)
        # Binary regularization (RDDLGN, 2025): encourage post-sigmoid h to commit to {0,1}
        # via λ·h(1-h). Stored as buffer; trainer reads and adds to loss. Training-time only.
        if self.bin_reg and self.training:
            self._bin_reg_buf = (h * (1.0 - h)).mean()
        else:
            self._bin_reg_buf = None
        if self.binary_io:
            if self.n_bits > 1:
                h = _thermometer_ste(h, self.n_bits, self.training)
            else:
                h = _binarize_ste(h)
        ste = self.use_ste and not hard
        gumbel = self.use_gumbel and self.training and not hard
        for l in self.logic:
            h = l(h, hard=hard, ste=ste, gumbel=gumbel)
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
        self.iwp      = getattr(soft, 'iwp', False)
        self.freeze_conn = getattr(soft, 'freeze_conn', False)
        with torch.no_grad():
            if self.freeze_conn:
                # Frozen connections: use column 0 of cand_a / cand_b directly (the fixed input).
                idx_a = soft.cand_a[:, 0]
                idx_b = soft.cand_b[:, 0]
            else:
                choice_a = soft.conn_logits_a.argmax(dim=-1)
                choice_b = soft.conn_logits_b.argmax(dim=-1)
                idx_a = soft.cand_a.gather(1, choice_a.unsqueeze(1)).squeeze(1)
                idx_b = soft.cand_b.gather(1, choice_b.unsqueeze(1)).squeeze(1)
            self.register_buffer('idx_a', idx_a.clone())
            self.register_buffer('idx_b', idx_b.clone())
            if self.iwp:
                # Round sigmoid(Omega) at 0.5 (i.e., Omega>0 -> 1, else 0). Gives the 4-bit truth table.
                w_hard = (soft.gate_omega > 0).to(torch.float32)
                self.register_buffer('w_iwp', w_hard.clone())
            else:
                self.register_buffer('coeffs', LOGIC_GATE_MATRIX[soft.gate_logits.argmax(dim=-1).cpu()].clone())

    def forward(self, x):
        if self.identity:
            return x
        a, b = x[:, self.idx_a], x[:, self.idx_b]
        if self.iwp:
            w = self.w_iwp.to(device=x.device, dtype=x.dtype)
            return (1-a)*(1-b)*w[:, 0] + (1-a)*b*w[:, 1] + a*(1-b)*w[:, 2] + a*b*w[:, 3]
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
        self.learn_pool = getattr(soft, 'learn_pool', False)
        self.pool_weighted = getattr(soft, 'pool_weighted', False)
        self.no_in_proj = getattr(soft, 'no_in_proj', False)
        self.skip_gate  = getattr(soft, 'skip_gate', False)
        self.C          = getattr(soft, 'C', None)
        self.eff_C      = getattr(soft, 'eff_C', self.C)
        self.token_shift = getattr(soft, 'token_shift', 0)
        self._taps      = getattr(soft, '_taps', resolve_taps(self.token_shift, None))
        self.group_size = getattr(soft, 'group_size', None)
        self.conv_in_k  = getattr(soft, 'conv_in_k', 0)
        self.conv_out_k = getattr(soft, 'conv_out_k', 0)
        self.logic_width = getattr(soft, 'logic_width', None)
        if self.skip_gate:
            self.register_buffer('skip_alpha', soft.skip_alpha.detach().clone())
        # pool_w / pool_scale only exist when sum_pool=True (group-sum aggregation).
        if self.sum_pool and self.pool_weighted:
            self.register_buffer('pool_w', soft.pool_w.detach().clone())
            self.register_buffer('pool_shift', soft.pool_shift.detach().clone())
        elif self.sum_pool and self.learn_pool:
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
            if self.pool_weighted:
                normed = (grp * self.pool_w).sum(dim=-1) + self.pool_shift
            elif self.learn_pool:
                normed = grp.sum(dim=-1) * self.pool_scale + self.pool_shift
            else:
                normed = (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        if self.conv_out_k > 0:
            h2 = h.view(B, T, -1).transpose(1, 2)
            h2 = F.pad(h2, (self.conv_out_k - 1, 0))
            out = self.out_proj(h2)
            return out.transpose(1, 2).contiguous()
        return self.out_proj(h).view(B, T, self.C)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.eff_C)
        if self.conv_in_k > 0:
            inp = normed_btx.transpose(1, 2)
            inp = F.pad(inp, (self.conv_in_k - 1, 0))
            out = self.in_proj(inp)
            return out.transpose(1, 2).reshape(B * T, self.logic_width)
        return self.in_proj(normed_btx.reshape(B * T, self.eff_C))

    def forward(self, x):
        B, T, C = x.shape
        normed = self.norm(x)
        normed = apply_token_shift(normed, self._taps)
        h = self._apply_in_proj(normed, B, T)
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
    copied FROZEN from the trained baseline and the MLP sublayer is replaced
    by a learnable logic circuit. Now supports ALL aggressive flags so the MLP
    side can be truly aggressive (binary_io + no_in_proj + sum_pool + ...)."""

    def __init__(self, gpt_cfg, layer_idx, original_block,
                 logic_width=None, depth=1, k=4, seed=1000,
                 activation='sigmoid', conn_init_scale=0.02, gate_init_scale=0.02,
                 identity_logic=False, binary_io=False, n_bits=1, sum_pool=False,
                 no_in_proj=False, skip_gate=False,
                 learn_pool=False, pool_weighted=False, iwp=False, token_shift=0,
                 random_from=999, conv_in_k=0, conv_out_k=0, bin_reg=False,
                 shift_taps=None):
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
        self.pool_weighted = pool_weighted
        self.iwp = iwp
        self.token_shift = token_shift
        self._taps = resolve_taps(token_shift, shift_taps)
        self.shift_taps = shift_taps
        self.no_in_proj = no_in_proj
        self.conv_in_k  = conv_in_k
        self.conv_out_k = conv_out_k
        self.random_from = random_from
        self.bin_reg    = bin_reg
        self._bin_reg_buf = None
        # eff_C = per-position channel width into the LGN body: (n_taps + 1) * C
        eff_C = C * (len(self._taps) + 1)
        self.eff_C = eff_C
        if conv_in_k > 0:
            assert not no_in_proj, "conv_in_k requires no_in_proj=False"
        if conv_out_k > 0:
            assert not sum_pool,   "conv_out_k requires sum_pool=False"
        if no_in_proj:
            assert binary_io, "no_in_proj requires binary_io"
            bit_width = eff_C * self.n_bits
        else:
            bit_width = self.logic_width * self.n_bits
        if sum_pool:
            assert binary_io, "sum_pool centering assumes binary {0,1} bits; requires binary_io"
            assert bit_width % C == 0, (
                f"sum_pool requires bit_width ({bit_width}) divisible by n_embd ({C}).")
            self.group_size = bit_width // C
            if pool_weighted:
                self.pool_w     = nn.Parameter(torch.full((C, self.group_size), 2.0 / self.group_size))
                self.pool_shift = nn.Parameter(torch.full((C,), -1.0))
            elif learn_pool:
                self.pool_scale = nn.Parameter(torch.full((C,), 2.0 / self.group_size))
                self.pool_shift = nn.Parameter(torch.full((C,), -1.0))

        # FROZEN: attention sublayer copied verbatim from trained baseline.
        # eval() override below ensures attention dropout stays off regardless of .train().
        self.ln_1 = copy.deepcopy(original_block.ln_1)
        self.attn = copy.deepcopy(original_block.attn)
        for p in self.ln_1.parameters(): p.requires_grad = False
        for p in self.attn.parameters(): p.requires_grad = False
        self.ln_1.eval(); self.attn.eval()

        # TRAINABLE: LGN MLP replacement (same shape as LogicGateGPTLayer)
        self.ln_2 = nn.LayerNorm(C)
        if not no_in_proj:
            if conv_in_k > 0:
                self.in_proj = nn.Conv1d(eff_C, self.logic_width, kernel_size=conv_in_k, padding=0)
            else:
                self.in_proj = nn.Linear(eff_C, self.logic_width)
        self.logic    = nn.ModuleList([
            LearnedLogicLayer(bit_width, bit_width, k=k,
                              seed=seed + layer_idx * 100 + i,
                              conn_init_scale=conn_init_scale,
                              gate_init_scale=gate_init_scale,
                              identity=identity_logic, iwp=iwp,
                              freeze_conn=(i >= random_from))
            for i in range(depth)
        ])
        if not sum_pool:
            if conv_out_k > 0:
                self.out_proj = nn.Conv1d(bit_width, C, kernel_size=conv_out_k, padding=0)
            else:
                self.out_proj = nn.Linear(bit_width, C)
        self.dropout  = nn.Dropout(gpt_cfg.dropout)
        self.use_ste  = False
        self.use_gumbel = False
        self.skip_gate = skip_gate
        if skip_gate:
            self.skip_alpha = nn.Parameter(torch.ones(1))

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
        """CAGE: average commitment confidence across this block's sublayers."""
        if not self.logic:
            return 1.0
        return sum(l.commitment() for l in self.logic) / len(self.logic)

    def bin_reg_loss(self):
        """RDDLGN-style binary regularization buffer; 0 if not enabled."""
        buf = getattr(self, '_bin_reg_buf', None)
        if buf is None:
            return torch.zeros((), device=self.ln_2.weight.device)
        return buf

    def entropy_loss(self, conn_w=0.001, gate_w=0.005):
        return sum(l.entropy_loss(conn_w, gate_w) for l in self.logic)

    @torch.no_grad()
    def sharpness(self):
        stats = [l.sharpness() for l in self.logic]
        return {k: sum(s[k] for s in stats) / len(stats) for k in ['conn_a', 'conn_b', 'gate']}

    def _aggregate(self, h, B, T):
        if self.sum_pool:
            grp = h.view(B * T, self.C, self.group_size)
            if self.pool_weighted:
                normed = (grp * self.pool_w).sum(dim=-1) + self.pool_shift
            elif self.learn_pool:
                normed = grp.sum(dim=-1) * self.pool_scale + self.pool_shift
            else:
                normed = (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        if self.conv_out_k > 0:
            h2 = h.view(B, T, -1).transpose(1, 2)
            h2 = F.pad(h2, (self.conv_out_k - 1, 0))
            out = self.out_proj(h2)
            return out.transpose(1, 2).contiguous()
        return self.out_proj(h).view(B, T, self.C)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.eff_C)
        if self.conv_in_k > 0:
            inp = normed_btx.transpose(1, 2)
            inp = F.pad(inp, (self.conv_in_k - 1, 0))
            out = self.in_proj(inp)
            return out.transpose(1, 2).reshape(B * T, self.logic_width)
        return self.in_proj(normed_btx.reshape(B * T, self.eff_C))

    def forward(self, x, hard=False):
        # FROZEN attention sublayer (identical to original)
        x = x + self.attn(self.ln_1(x))
        # LGN MLP replacement (full aggressive-flag support)
        B, T, C = x.shape
        normed = self.ln_2(x)
        normed = apply_token_shift(normed, self._taps)
        h = self._apply_in_proj(normed, B, T)
        h = _apply_activation(h, self.activation)
        # Binary regularization (RDDLGN, 2025): h(1-h) on post-sigmoid pre-binarization values.
        if self.bin_reg and self.training:
            self._bin_reg_buf = (h * (1.0 - h)).mean()
        else:
            self._bin_reg_buf = None
        if self.binary_io:
            if self.n_bits > 1:
                h = _thermometer_ste(h, self.n_bits, self.training)
            else:
                h = _binarize_ste(h)
        ste = self.use_ste and not hard
        gumbel = self.use_gumbel and self.training and not hard
        for l in self.logic:
            h = l(h, hard=hard, ste=ste, gumbel=gumbel)
        contrib = self.dropout(self._aggregate(h, B, T))
        if self.skip_gate:
            contrib = self.skip_alpha * contrib
        return x + contrib


class HardHybridLogicGateGPTLayer(nn.Module):
    """Hard-snapped HybridLogicGateGPTLayer. Attention stays continuous and identical
    to the original; only the LGN MLP is discretised. Supports all aggressive flags."""

    def __init__(self, soft: HybridLogicGateGPTLayer):
        super().__init__()
        self.layer_idx  = soft.layer_idx
        self.activation = soft.activation
        self.binary_io  = getattr(soft, 'binary_io', False)
        self.n_bits     = getattr(soft, 'n_bits', 1)
        self.sum_pool   = getattr(soft, 'sum_pool', False)
        self.learn_pool = getattr(soft, 'learn_pool', False)
        self.pool_weighted = getattr(soft, 'pool_weighted', False)
        self.no_in_proj = getattr(soft, 'no_in_proj', False)
        self.skip_gate  = getattr(soft, 'skip_gate', False)
        self.C          = getattr(soft, 'C', None)
        self.eff_C      = getattr(soft, 'eff_C', self.C)
        self.token_shift = getattr(soft, 'token_shift', 0)
        self._taps      = getattr(soft, '_taps', resolve_taps(self.token_shift, None))
        self.group_size = getattr(soft, 'group_size', None)
        self.conv_in_k  = getattr(soft, 'conv_in_k', 0)
        self.conv_out_k = getattr(soft, 'conv_out_k', 0)
        self.logic_width = getattr(soft, 'logic_width', None)
        if self.skip_gate:
            self.register_buffer('skip_alpha', soft.skip_alpha.detach().clone())
        # pool_w / pool_scale only exist when sum_pool=True (group-sum aggregation).
        if self.sum_pool and self.pool_weighted:
            self.register_buffer('pool_w', soft.pool_w.detach().clone())
            self.register_buffer('pool_shift', soft.pool_shift.detach().clone())
        elif self.sum_pool and self.learn_pool:
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
            if self.pool_weighted:
                normed = (grp * self.pool_w).sum(dim=-1) + self.pool_shift
            elif self.learn_pool:
                normed = grp.sum(dim=-1) * self.pool_scale + self.pool_shift
            else:
                normed = (grp.sum(dim=-1) - self.group_size / 2) / (self.group_size / 2)
            return normed.view(B, T, self.C)
        if self.conv_out_k > 0:
            h2 = h.view(B, T, -1).transpose(1, 2)
            h2 = F.pad(h2, (self.conv_out_k - 1, 0))
            out = self.out_proj(h2)
            return out.transpose(1, 2).contiguous()
        return self.out_proj(h).view(B, T, self.C)

    def _apply_in_proj(self, normed_btx, B, T):
        if self.no_in_proj:
            return normed_btx.reshape(B * T, self.eff_C)
        if self.conv_in_k > 0:
            inp = normed_btx.transpose(1, 2)
            inp = F.pad(inp, (self.conv_in_k - 1, 0))
            out = self.in_proj(inp)
            return out.transpose(1, 2).reshape(B * T, self.logic_width)
        return self.in_proj(normed_btx.reshape(B * T, self.eff_C))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        B, T, C = x.shape
        normed = self.ln_2(x)
        normed = apply_token_shift(normed, self._taps)
        h = self._apply_in_proj(normed, B, T)
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

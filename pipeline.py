import copy
import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

from lgn import (
    LogicGateGPTLayer, HardLogicGateGPTLayer,
    HybridLogicGateGPTLayer, HardHybridLogicGateGPTLayer,
    annealed_temperature
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _ensure_datasets():
    try:
        import datasets  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets'], check=True)


class WikiText2:
    def __init__(self, data_cfg, device='cuda'):
        _ensure_datasets()
        from datasets import load_dataset
        self.block_size = data_cfg.block_size
        self.device = device
        print('Loading WikiText-2...')
        wt = load_dataset('wikitext', 'wikitext-2-raw-v1')
        self.train = self._encode(wt['train'],      data_cfg.train_chars, device)
        self.val   = self._encode(wt['validation'], data_cfg.val_chars,   device)
        print(f'  train: {len(self.train):,}  val: {len(self.val):,} bytes')

    @staticmethod
    def _encode(rows, max_chars, device):
        pieces, total = [], 0
        for row in rows:
            line = row['text'].strip()
            if not line:
                continue
            pieces.append(line + '\n')
            total += len(pieces[-1])
            if total >= max_chars:
                break
        text = ''.join(pieces)[:max_chars]
        return torch.tensor(list(text.encode('utf-8', errors='ignore')), dtype=torch.long, device=device)

    def get_batch(self, split='train', batch_size=32):
        src = self.train if split == 'train' else self.val
        ix = torch.randint(len(src) - self.block_size - 1, (batch_size,), device=self.device)
        x = torch.stack([src[i:i + self.block_size]         for i in ix])
        y = torch.stack([src[i + 1:i + 1 + self.block_size] for i in ix])
        return x, y

# ---------------------------------------------------------------------------
# Core training helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, data, eval_iters=30, batch_size=32):
    model.eval()
    losses = [float(model(*data.get_batch('val', batch_size))[1]) for _ in range(eval_iters)]
    return sum(losses) / len(losses)


@torch.no_grad()
def estimate_metrics(model, data, eval_iters=30, batch_size=32):
    """Loss (nats), perplexity, and next-token top-1 accuracy on the val set."""
    import math
    model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    for _ in range(eval_iters):
        xb, yb = data.get_batch('val', batch_size)
        logits, loss = model(xb, yb)
        tot_loss += float(loss)
        pred = logits.argmax(dim=-1)
        correct += int((pred == yb).sum())
        total   += yb.numel()
    avg_loss = tot_loss / eval_iters
    return {
        'loss':       round(avg_loss, 5),
        'perplexity': round(math.exp(avg_loss), 4),
        'accuracy':   round(correct / total, 5),
    }


def train_baseline(model, data, cfg):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.baseline_lr)
    model.train()
    for step in range(cfg.baseline_steps):
        _, loss = model(*data.get_batch('train', cfg.batch_size))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % cfg.log_every == 0 or step == cfg.baseline_steps - 1:
            val = estimate_loss(model, data, cfg.eval_iters, cfg.batch_size)
            print(f'  step {step:5d} | train {loss.item():.4f} | val {val:.4f}')
            model.train()


@torch.no_grad()
def get_layer_io(trained_model, layer_idx, data, batch_size=32, input_model=None):
    """Return (layer_in, layer_tgt) for imitation.

    - layer_in: input distribution seen at layer_idx — uses `input_model` if given,
      else `trained_model`. In cumulative scaling, pass live_model so the new LGN
      layer trains against the ACTUAL upstream distribution (which differs from the
      original transformer's after prior layers have been replaced).
    - layer_tgt: original transformer's response at that layer, on the SAME input.
      This is what we want the LGN to approximate.
    """
    src = input_model if input_model is not None else trained_model
    device = next(trained_model.parameters()).device
    # Snapshot per-module training flags and restore them afterwards. Without this,
    # eval() here would leave the (shared) live_model — including the LGN layer we are
    # about to train — in eval mode, silently disabling training-only paths such as the
    # thermometer STE branch and Gumbel-STE on the very forward that computes the loss.
    models = []
    for m in (trained_model, src):
        if m not in models:
            models.append(m)
    saved = [(sub, sub.training) for m in models for sub in m.modules()]
    try:
        for m in models:
            m.eval()
        xb, _ = data.get_batch('train', batch_size)
        pos = torch.arange(xb.size(1), device=device)
        x = src.transformer.drop(src.transformer.wte(xb) + src.transformer.wpe(pos))
        for i, block in enumerate(src.transformer.h):
            if i == layer_idx:
                # Target = original layer applied to LIVE input.
                tgt = trained_model.transformer.h[layer_idx](x)
                return x.detach(), tgt.detach()
            x = block(x)
        raise RuntimeError(f'layer {layer_idx} not found')
    finally:
        for sub, flag in saved:
            sub.training = flag


def imitate_layer(logic_model, layer_idx, trained_model, data, cfg, step_mult=1.0,
                  input_model=None):
    """Train one LGN layer to imitate the corresponding transformer MLP.

    input_model: if provided (cumulative scaling), the upstream forward uses LIVE
    state (with prior LGN layers in place) so the new layer sees the ACTUAL input
    distribution it will face, not the original transformer's distribution. The
    imitation target is still computed by the ORIGINAL layer on that live input."""
    logic_layer = logic_model.transformer.h[layer_idx]
    logic_layer.set_temperature(cfg.temp_start)
    if getattr(cfg, 'gumbel_ste', False):
        logic_layer.use_gumbel = True
    trainable = [p for p in logic_layer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.imitation_lr)
    total_steps = max(1, int(cfg.imitation_steps * step_mult))
    loss_type = getattr(cfg, 'imit_loss', 'mse')
    # If annealing is deferred to fine-tune, keep the gates SOFT here (warm start only).
    soft_hold = getattr(cfg, 'anneal_in_finetune', False)
    logic_layer.train()  # train mode for the whole loop; get_layer_io no longer flips it to eval
    for step in range(total_steps):
        temp = cfg.temp_start if soft_hold else annealed_temperature(step, total_steps, cfg.temp_start, cfg.temp_end)
        logic_layer.set_temperature(temp)

        if loss_type == 'mse':
            # Activation-matching imitation. In scaling, input_model=live_model so the
            # input reflects the actual upstream LGN-modified distribution.
            layer_in, layer_tgt = get_layer_io(
                trained_model, layer_idx, data, cfg.batch_size, input_model=input_model)
            primary = F.mse_loss(logic_layer(layer_in), layer_tgt)
        elif loss_type == 'kl':
            # End-to-end task-loss imitation via output distribution distillation
            xb, yb = data.get_batch('train', cfg.batch_size)
            with torch.no_grad():
                trained_model.eval()
                teacher_logits, _ = trained_model(xb, yb)
            student_logits, _ = logic_model(xb, yb)
            V = student_logits.size(-1)
            log_p_student = F.log_softmax(student_logits.reshape(-1, V), dim=-1)
            p_teacher = F.softmax(teacher_logits.reshape(-1, V), dim=-1).detach()
            primary = F.kl_div(log_p_student, p_teacher, reduction='batchmean')
        else:
            raise ValueError(f"unknown imit_loss '{loss_type}'")

        ent = logic_layer.entropy_loss(cfg.ent_conn, cfg.ent_gate)
        (primary + ent).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == total_steps - 1:
            s = logic_layer.sharpness()
            print(f'    imit-{loss_type} {step:4d}/{total_steps} | temp {temp:.3f} | '
                  f'{loss_type} {primary.detach().item():.5f} | gate_sharp {s["gate"]:.3f}')
    # Leave gates soft if fine-tune will do the annealing; otherwise snap to temp_end.
    if not soft_hold:
        logic_layer.set_temperature(cfg.temp_end)


def _trainable_params(model, target_indices, cfg):
    """Collect trainable params for the optimizer.

    NOTE: the base model is already globally frozen by _make_logic_model /
    _add_logic_layer (which set requires_grad=False on everything, then
    re-enable only the LGN layers via _enable_lgn_grads). So this just returns
    whatever currently has requires_grad=True. The historical 'freeze_unreplaced'
    flag was a no-op (the base is ALWAYS frozen by design) and has been removed."""
    return [p for p in model.parameters() if p.requires_grad]


def finetune_layers(logic_model, target_indices, data, cfg, trained_model=None, anneal_indices=None):
    device = next(logic_model.parameters()).device
    trainable = _trainable_params(logic_model, target_indices, cfg)
    opt = torch.optim.AdamW(trainable, lr=cfg.finetune_lr)
    anneal = getattr(cfg, 'anneal_in_finetune', False)
    imit_w = getattr(cfg, 'ft_imit_weight', 0.0)
    # Cross-layer coordination: only the newly-added layer(s) anneal from soft; layers that
    # were already settled in earlier scaling steps stay SHARP (temp_end) so their committed
    # solutions are not re-melted every step. Default = anneal all (single-layer heatmap).
    if anneal_indices is None:
        anneal_indices = target_indices
    anneal_set = set(anneal_indices)
    if anneal:
        n_sharp = len(target_indices) - len(anneal_set)
        print(f'    [anneal_in_finetune] anneal {sorted(anneal_set)} temp {cfg.temp_start}->{cfg.temp_end}; '
              f'keep {n_sharp} settled layer(s) sharp')
        for idx in target_indices:
            logic_model.transformer.h[idx].set_temperature(
                cfg.temp_start if idx in anneal_set else cfg.temp_end)
    else:
        for idx in target_indices:
            logic_model.transformer.h[idx].set_temperature(cfg.temp_end)
    if imit_w > 0 and trained_model is not None and len(target_indices) == 1:
        print(f'    [curriculum] MSE-to-MLP weight {imit_w} decaying to 0')
    # Enable straight-through estimator: forward=hard, backward=soft
    use_ste = getattr(cfg, 'ste', False)
    use_cage = getattr(cfg, 'cage', False)
    if use_cage:
        use_ste = True  # CAGE implies hard forward (STE forward path)
    if use_ste:
        for idx in target_indices:
            logic_model.transformer.h[idx].use_ste = True
        print('    [STE enabled] forward=hard argmax, backward=soft softmax')
    if use_cage:
        tau_max = getattr(cfg, 'cage_tau_max', 3.0)
        tau_min = getattr(cfg, 'cage_tau_min', 0.5)
        ema_decay = getattr(cfg, 'cage_ema', 0.99)
        cage_K = 16  # gate logits dimension (worst case; conn=4 also fits in [1/16, 1])
        c_ema = 1.0 / cage_K  # initial confidence = uniform
        print(f'    [CAGE enabled] tau_max={tau_max} tau_min={tau_min} ema={ema_decay}')
    # Enable Gumbel-STE (Mind the Gap 2025): stochastic discrete training, reduces soft-hard gap
    if getattr(cfg, 'gumbel_ste', False):
        for idx in target_indices:
            logic_model.transformer.h[idx].use_gumbel = True
        print('    [Gumbel-STE enabled] stochastic discrete sample + soft gradient')
    bin_reg_w = float(getattr(cfg, 'bin_reg_weight', 0.0))
    if bin_reg_w > 0:
        print(f'    [bin_reg enabled] weight={bin_reg_w}')
    logic_model.train()
    for step in range(cfg.finetune_steps):
        if anneal:
            temp = annealed_temperature(step, cfg.finetune_steps, cfg.temp_start, cfg.temp_end)
            for idx in anneal_set:
                logic_model.transformer.h[idx].set_temperature(temp)
            # settled layers stay at temp_end (left untouched)
        # CAGE: update commitment EMA + adaptive backward temperature τ_b each step.
        if use_cage:
            with torch.no_grad():
                conf = sum(logic_model.transformer.h[idx].commitment()
                           for idx in target_indices) / max(len(target_indices), 1)
            c_ema = ema_decay * c_ema + (1.0 - ema_decay) * conf
            # τ_b interpolates from tau_max (uniform conf) → tau_min (fully committed)
            frac = (c_ema - 1.0 / cage_K) / max(1.0 - 1.0 / cage_K, 1e-6)
            tau_b = tau_max - (tau_max - tau_min) * max(0.0, min(1.0, frac))
            for idx in target_indices:
                logic_model.transformer.h[idx].set_backward_temp(tau_b)
        _, lm_loss = logic_model(*data.get_batch('train', cfg.batch_size))
        ent = sum(logic_model.transformer.h[idx].entropy_loss(cfg.ft_ent_conn, cfg.ft_ent_gate)
                  for idx in target_indices)
        loss = lm_loss + ent
        # Binary regularization (RDDLGN 2025): pushes pre-binarization sigmoid output to {0,1}.
        if bin_reg_w > 0:
            bin_terms = [logic_model.transformer.h[idx].bin_reg_loss() for idx in target_indices]
            loss = loss + bin_reg_w * sum(bin_terms)
        # Curriculum: decaying MSE-to-MLP guidance (single-layer fine-tune only).
        # input_model=logic_model so the curriculum sees the LIVE upstream distribution
        # (mirrors the imitation fix; otherwise the target is anchored to the wrong inputs).
        if imit_w > 0 and trained_model is not None and len(target_indices) == 1:
            decay = 1.0 - step / max(cfg.finetune_steps - 1, 1)
            idx = target_indices[0]
            layer_in, layer_tgt = get_layer_io(
                trained_model, idx, data, cfg.batch_size, input_model=logic_model)
            loss = loss + imit_w * decay * F.mse_loss(logic_model.transformer.h[idx](layer_in), layer_tgt)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == cfg.finetune_steps - 1:
            print(f'    ft   {step:4d} | lm {float(lm_loss.detach()):.4f} | ent {float(ent.detach()):.5f}')
            if cfg.ft_log_sharpness:
                for idx in target_indices:
                    s = logic_model.transformer.h[idx].sharpness()
                    layer = logic_model.transformer.h[idx]
                    extra = ''
                    if getattr(layer, 'skip_gate', False):
                        extra = f'  skip_alpha={float(layer.skip_alpha):.4f}'
                    print(f'      L{idx}: gate={s["gate"]:.3f}  conn_a={s["conn_a"]:.3f}  conn_b={s["conn_b"]:.3f}{extra}')
            if cfg.ft_eval_hard:
                hard = make_hard_model(logic_model, target_indices, device)
                hard_val = estimate_loss(hard, data, cfg.eval_iters, cfg.batch_size)
                print(f'      hard_val={hard_val:.4f}')
                logic_model.train()
    # Restore toggles so subsequent eval / joint polish doesn't get unintended noise.
    if use_ste:
        for idx in target_indices:
            logic_model.transformer.h[idx].use_ste = False
    if use_cage:
        for idx in target_indices:
            logic_model.transformer.h[idx].set_backward_temp(None)
    if getattr(cfg, 'gumbel_ste', False):
        for idx in target_indices:
            logic_model.transformer.h[idx].use_gumbel = False


def make_hard_model(soft_model, target_indices, device):
    hard = copy.deepcopy(soft_model)
    for idx in target_indices:
        soft_layer = soft_model.transformer.h[idx]
        if isinstance(soft_layer, HybridLogicGateGPTLayer):
            hard_layer = HardHybridLogicGateGPTLayer(soft_layer).to(device)
        else:
            hard_layer = HardLogicGateGPTLayer(soft_layer).to(device)
        hard.replace_layer(idx, hard_layer)
    hard.eval()
    return hard

# ---------------------------------------------------------------------------
# Experiment: per-layer heatmap
# ---------------------------------------------------------------------------

def _is_edge(layer_idx, n_layer):
    return layer_idx == 0 or layer_idx == n_layer - 1


def _layer_geometry(layer_idx, n_layer, logic_cfg):
    """Return (depth, width_mult) for this layer index, honoring edge overrides."""
    if _is_edge(layer_idx, n_layer):
        depth = logic_cfg.edge_depth if logic_cfg.edge_depth > 0 else logic_cfg.depth
        wmult = logic_cfg.edge_width_mult if logic_cfg.edge_width_mult > 0 else logic_cfg.width_mult
    else:
        depth = logic_cfg.depth
        wmult = logic_cfg.width_mult
    return depth, wmult


def _make_logic_model(trained_default, layer_idx, gpt_cfg, logic_cfg, device):
    model = copy.deepcopy(trained_default)
    depth, wmult = _layer_geometry(layer_idx, gpt_cfg.n_layer, logic_cfg)
    if _is_edge(layer_idx, gpt_cfg.n_layer) and (logic_cfg.edge_depth > 0 or logic_cfg.edge_width_mult > 0):
        print(f'  [edge layer override] depth={depth}, width_mult={wmult}')
    if layer_idx in logic_cfg.hybrid_layers:
        print(f'  [hybrid] keeping original attention sublayer for L{layer_idx}, logic replaces MLP only')
        # Hybrid layer now propagates ALL aggressive flags: the MLP path is identical to
        # LogicGateGPTLayer (binary_io, no_in_proj, sum_pool, learn_pool, pool_weighted,
        # token_shift, iwp, skip_gate). Only the attention sublayer is frozen.
        logic_layer = HybridLogicGateGPTLayer(
            gpt_cfg, layer_idx, trained_default.transformer.h[layer_idx],
            logic_width=gpt_cfg.n_embd * wmult,
            depth=depth, k=logic_cfg.k,
            activation=logic_cfg.activation,
            conn_init_scale=logic_cfg.conn_init_scale,
            gate_init_scale=logic_cfg.gate_init_scale,
            identity_logic=logic_cfg.identity_logic,
            binary_io=logic_cfg.binary_io,
            n_bits=logic_cfg.n_bits,
            sum_pool=logic_cfg.sum_pool,
            no_in_proj=logic_cfg.no_in_proj,
            skip_gate=logic_cfg.skip_gate,
            learn_pool=logic_cfg.learn_pool,
            pool_weighted=logic_cfg.pool_weighted,
            iwp=logic_cfg.iwp,
            token_shift=logic_cfg.token_shift,
            random_from=getattr(logic_cfg, 'random_from', 999),
            conv_in_k=getattr(logic_cfg,  'conv_in_k',  0),
            conv_out_k=getattr(logic_cfg, 'conv_out_k', 0),
            bin_reg=getattr(logic_cfg,    'bin_reg',    False),
            shift_taps=getattr(logic_cfg, 'shift_taps', None),
        ).to(device)
    else:
        logic_layer = LogicGateGPTLayer(
            gpt_cfg, layer_idx,
            logic_width=gpt_cfg.n_embd * wmult,
            depth=depth, k=logic_cfg.k,
            activation=logic_cfg.activation,
            conn_init_scale=logic_cfg.conn_init_scale,
            gate_init_scale=logic_cfg.gate_init_scale,
            identity_logic=logic_cfg.identity_logic,
            binary_io=logic_cfg.binary_io,
            n_bits=logic_cfg.n_bits,
            sum_pool=logic_cfg.sum_pool,
            no_in_proj=logic_cfg.no_in_proj,
            skip_gate=logic_cfg.skip_gate,
            learn_pool=logic_cfg.learn_pool,
            pool_weighted=logic_cfg.pool_weighted,
            iwp=logic_cfg.iwp,
            token_shift=logic_cfg.token_shift,
            random_from=getattr(logic_cfg, 'random_from', 999),
            conv_in_k=getattr(logic_cfg,  'conv_in_k',  0),
            conv_out_k=getattr(logic_cfg, 'conv_out_k', 0),
            bin_reg=getattr(logic_cfg,    'bin_reg',    False),
            shift_taps=getattr(logic_cfg, 'shift_taps', None),
        ).to(device)
    model.replace_layer(layer_idx, logic_layer)
    for p in model.parameters(): p.requires_grad = False
    _enable_lgn_grads(model.transformer.h[layer_idx])
    return model


def run_heatmap(trained_default, gpt_cfg, data, exp_cfg, save_path=None, layers=None):
    device = next(trained_default.parameters()).device
    baseline_val = estimate_loss(trained_default, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
    print(f'Baseline val loss: {baseline_val:.4f}\n')
    results = []

    n = gpt_cfg.n_layer
    layer_list = layers if layers else list(range(n))
    for layer_idx in layer_list:
        print(f'\n{"="*55}\nLayer {layer_idx} / {n - 1}\n{"="*55}')
        logic_model = _make_logic_model(trained_default, layer_idx, gpt_cfg, exp_cfg.logic, device)
        logic_layer = logic_model.transformer.h[layer_idx]

        # per-layer annealing: edge layers (first/last 20%) get 2x imitation steps
        step_mult = 1.0
        if exp_cfg.train.per_layer_anneal:
            edge_thresh = max(1, round(n * 0.2))
            is_edge = layer_idx < edge_thresh or layer_idx >= n - edge_thresh
            step_mult = 2.0 if is_edge else 1.0
            if step_mult > 1.0:
                print(f'  [per_layer_anneal] edge layer -> {step_mult}x steps')

        if exp_cfg.train.imitation_steps > 0:
            print('  [Imitation]')
            imitate_layer(logic_model, layer_idx, trained_default, data, exp_cfg.train, step_mult=step_mult)
        else:
            print('  [Imitation skipped: imitation_steps=0 -> direct training]')
            logic_layer.set_temperature(exp_cfg.train.temp_start)
        sharp_imit = logic_layer.sharpness()
        val_imit   = estimate_loss(logic_model, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)

        print('  [Fine-tune]')
        finetune_layers(logic_model, [layer_idx], data, exp_cfg.train, trained_model=trained_default)
        sharp_ft = logic_layer.sharpness()

        soft_val   = estimate_loss(logic_model, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
        hard_model = make_hard_model(logic_model, [layer_idx], device)
        hard_val   = estimate_loss(hard_model, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)

        result = {
            'layer_idx':        layer_idx,
            'baseline_val':     round(baseline_val, 5),
            'val_after_imit':   round(val_imit, 5),
            'soft_val':         round(soft_val, 5),
            'hard_val':         round(hard_val, 5),
            'soft_degradation': round(soft_val - baseline_val, 5),
            'hard_degradation': round(hard_val - baseline_val, 5),
            'sharpness_imit':   {k: round(v, 4) for k, v in sharp_imit.items()},
            'sharpness_ft':     {k: round(v, 4) for k, v in sharp_ft.items()},
        }
        results.append(result)
        print(f'\n  Layer {layer_idx:2d} | soft d {result["soft_degradation"]:+.4f} | '
              f'hard d {result["hard_degradation"]:+.4f} | gate_sharp {sharp_ft["gate"]:.3f}')

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, 'w') as f: json.dump(results, f, indent=2)

    return results

# ---------------------------------------------------------------------------
# Experiment: scaling (greedy / uniform)
# ---------------------------------------------------------------------------

def _enable_lgn_grads(layer):
    """Enable gradients on a logic layer's trainable params.

    BUG FIX: HybridLogicGateGPTLayer.__init__ freezes attn/ln_1 (copied from trained
    baseline), but the surrounding pipeline used to re-enable ALL params via
    `for p in layer.parameters(): p.requires_grad = True`. That silently re-trained
    the attention sublayer, defeating the "keep original attention" intent.
    Now: skip attn.* and ln_1.* names for hybrid layers; standard layers unchanged."""
    is_hybrid = isinstance(layer, HybridLogicGateGPTLayer)
    for name, p in layer.named_parameters():
        if is_hybrid and (name.startswith('attn.') or name.startswith('ln_1.')):
            p.requires_grad = False
        else:
            p.requires_grad = True


def _logic_indices(model):
    return [i for i, b in enumerate(model.transformer.h)
            if isinstance(b, (LogicGateGPTLayer, HybridLogicGateGPTLayer))]


def _add_logic_layer(model, layer_idx, gpt_cfg, logic_cfg, device, trained_default=None):
    depth, wmult = _layer_geometry(layer_idx, gpt_cfg.n_layer, logic_cfg)
    if _is_edge(layer_idx, gpt_cfg.n_layer) and (logic_cfg.edge_depth > 0 or logic_cfg.edge_width_mult > 0):
        print(f'  [edge layer override] depth={depth}, width_mult={wmult}')
    if layer_idx in logic_cfg.hybrid_layers:
        if trained_default is None:
            raise ValueError("hybrid layer requires trained_default to copy attention from")
        print(f'  [hybrid] keeping original attention sublayer for L{layer_idx}, logic replaces MLP only')
        new_layer = HybridLogicGateGPTLayer(
            gpt_cfg, layer_idx, trained_default.transformer.h[layer_idx],
            logic_width=gpt_cfg.n_embd * wmult,
            depth=depth, k=logic_cfg.k,
            activation=logic_cfg.activation,
            conn_init_scale=logic_cfg.conn_init_scale,
            gate_init_scale=logic_cfg.gate_init_scale,
            identity_logic=logic_cfg.identity_logic,
            binary_io=logic_cfg.binary_io,
            n_bits=logic_cfg.n_bits,
            sum_pool=logic_cfg.sum_pool,
            no_in_proj=logic_cfg.no_in_proj,
            skip_gate=logic_cfg.skip_gate,
            learn_pool=logic_cfg.learn_pool,
            pool_weighted=logic_cfg.pool_weighted,
            iwp=logic_cfg.iwp,
            token_shift=logic_cfg.token_shift,
            random_from=getattr(logic_cfg, 'random_from', 999),
            conv_in_k=getattr(logic_cfg,  'conv_in_k',  0),
            conv_out_k=getattr(logic_cfg, 'conv_out_k', 0),
            bin_reg=getattr(logic_cfg,    'bin_reg',    False),
            shift_taps=getattr(logic_cfg, 'shift_taps', None),
        ).to(device)
    else:
        new_layer = LogicGateGPTLayer(
            gpt_cfg, layer_idx,
            logic_width=gpt_cfg.n_embd * wmult,
            depth=depth, k=logic_cfg.k,
            activation=logic_cfg.activation,
            conn_init_scale=logic_cfg.conn_init_scale,
            gate_init_scale=logic_cfg.gate_init_scale,
            identity_logic=logic_cfg.identity_logic,
            binary_io=logic_cfg.binary_io,
            n_bits=logic_cfg.n_bits,
            sum_pool=logic_cfg.sum_pool,
            no_in_proj=logic_cfg.no_in_proj,
            skip_gate=logic_cfg.skip_gate,
            learn_pool=logic_cfg.learn_pool,
            pool_weighted=logic_cfg.pool_weighted,
            iwp=logic_cfg.iwp,
            token_shift=logic_cfg.token_shift,
            random_from=getattr(logic_cfg, 'random_from', 999),
            conv_in_k=getattr(logic_cfg,  'conv_in_k',  0),
            conv_out_k=getattr(logic_cfg, 'conv_out_k', 0),
            bin_reg=getattr(logic_cfg,    'bin_reg',    False),
            shift_taps=getattr(logic_cfg, 'shift_taps', None),
        ).to(device)
    model.replace_layer(layer_idx, new_layer)
    for p in model.parameters(): p.requires_grad = False
    for idx in _logic_indices(model):
        _enable_lgn_grads(model.transformer.h[idx])


def joint_polish(live_model, target_indices, data, cfg, trained_model=None):
    """Coordinate all LGN layers together after sequential scaling: a final joint
    fine-tune of every LGN layer (all kept sharp at temp_end) on the LM loss, with
    an optional system-level KL term to the original transformer's logits."""
    steps = getattr(cfg, 'joint_polish_steps', 0)
    kl_w  = getattr(cfg, 'joint_polish_kl_weight', 0.0)
    if steps <= 0:
        return
    for idx in target_indices:
        live_model.transformer.h[idx].set_temperature(cfg.temp_end)
    # If per-layer training used hard forward (STE/CAGE), polish in the same regime so
    # we don't re-open the discretization gap with a soft-forward polish.
    polish_ste = getattr(cfg, 'ste', False) or getattr(cfg, 'cage', False)
    if polish_ste:
        for idx in target_indices:
            live_model.transformer.h[idx].use_ste = True
    trainable = _trainable_params(live_model, target_indices, cfg)
    opt = torch.optim.AdamW(trainable, lr=cfg.finetune_lr)
    print(f'  [JOINT POLISH] {len(target_indices)} layers together | {steps} steps | '
          f'kl_weight={kl_w} | ste={polish_ste}')
    live_model.train()
    for step in range(steps):
        xb, yb = data.get_batch('train', cfg.batch_size)
        student_logits, lm_loss = live_model(xb, yb)
        ent = sum(live_model.transformer.h[idx].entropy_loss(cfg.ft_ent_conn, cfg.ft_ent_gate)
                  for idx in target_indices)
        loss = lm_loss + ent
        if kl_w > 0 and trained_model is not None:
            with torch.no_grad():
                trained_model.eval()
                teacher_logits, _ = trained_model(xb, yb)
            V = student_logits.size(-1)
            log_p = F.log_softmax(student_logits.reshape(-1, V), dim=-1)
            p_t   = F.softmax(teacher_logits.reshape(-1, V), dim=-1).detach()
            loss = loss + kl_w * F.kl_div(log_p, p_t, reduction='batchmean')
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == steps - 1:
            print(f'    polish {step:4d}/{steps} | lm {float(lm_loss.detach()):.4f}')
    if polish_ste:
        for idx in target_indices:
            live_model.transformer.h[idx].use_ste = False


def run_scaling(trained_default, gpt_cfg, data, exp_cfg,
                strategy='greedy', heatmap_results=None, save_path=None,
                protected_layers=None):
    device = next(trained_default.parameters()).device
    protected = set(protected_layers or [])
    n = gpt_cfg.n_layer

    if strategy == 'greedy':
        if heatmap_results is None:
            raise ValueError("strategy='greedy' requires heatmap_results")
        layer_order = [r['layer_idx'] for r in sorted(heatmap_results, key=lambda r: r['hard_degradation'])
                       if r['layer_idx'] not in protected]
    elif strategy == 'reverse_greedy':
        if heatmap_results is None:
            raise ValueError("strategy='reverse_greedy' requires heatmap_results")
        # Hardest layer first: lets the still-transformer downstream compensate while LGN learns L0.
        layer_order = [r['layer_idx'] for r in sorted(heatmap_results, key=lambda r: -r['hard_degradation'])
                       if r['layer_idx'] not in protected]
    else:
        layer_order = [i for i in range(0, n, max(1, n // 8)) if i not in protected]

    if protected:
        print(f'Protected layers (never replaced): {sorted(protected)}')

    baseline_val = estimate_loss(trained_default, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
    print(f'Baseline val loss: {baseline_val:.4f}')
    print(f'Replacement order ({strategy}): {layer_order}\n')

    results = [{'n_replaced': 0, 'replaced_layers': [],
                'soft_val': round(baseline_val, 5), 'hard_val': round(baseline_val, 5),
                'soft_degradation': 0.0, 'hard_degradation': 0.0}]

    live_model = copy.deepcopy(trained_default)

    for new_layer_idx in layer_order:
        current = sorted(_logic_indices(live_model) + [new_layer_idx])
        print(f'\n{"="*55}\nAdding layer {new_layer_idx}  (total: {len(current)} -> {current})\n{"="*55}')

        _add_logic_layer(live_model, new_layer_idx, gpt_cfg, exp_cfg.logic, device,
                         trained_default=trained_default)

        # per-layer annealing: scale steps by normalised difficulty from heatmap
        step_mult = 1.0
        if exp_cfg.train.per_layer_anneal and heatmap_results:
            scores = {r['layer_idx']: r['hard_degradation'] for r in heatmap_results}
            all_scores = list(scores.values())
            lo, hi = min(all_scores), max(all_scores)
            difficulty = (scores.get(new_layer_idx, 0) - lo) / max(hi - lo, 1e-8)
            step_mult = 1.0 + difficulty * 2.0  # 1x (easiest) to 3x (hardest)
            print(f'  [per_layer_anneal] difficulty={difficulty:.2f} -> {step_mult:.2f}x steps')

        if exp_cfg.train.imitation_steps > 0:
            print(f'  [Imitation layer {new_layer_idx}]')
            # Pass live_model so imitation uses the ACTUAL upstream distribution (with
            # prior LGN replacements), not the original transformer's distribution.
            imitate_layer(live_model, new_layer_idx, trained_default, data, exp_cfg.train,
                          step_mult=step_mult, input_model=live_model)
        else:
            # Direct LM training: no imitation warm-start. Start the new layer soft so
            # fine-tune annealing can shape it from scratch (matches run_heatmap behaviour).
            print(f'  [Imitation skipped: imitation_steps=0 -> direct training] layer {new_layer_idx}')
            live_model.transformer.h[new_layer_idx].set_temperature(exp_cfg.train.temp_start)

        print(f'  [Fine-tune {len(current)} layers]')
        # Only the newly-added layer anneals; previously-settled layers stay sharp (coordination).
        finetune_layers(live_model, current, data, exp_cfg.train,
                        trained_model=trained_default, anneal_indices=[new_layer_idx])

        soft_val   = estimate_loss(live_model, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
        hard_model = make_hard_model(live_model, current, device)
        hard_val   = estimate_loss(hard_model, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)

        result = {
            'n_replaced':       len(current),
            'replaced_layers':  current,
            'soft_val':         round(soft_val, 5),
            'hard_val':         round(hard_val, 5),
            'soft_degradation': round(soft_val - baseline_val, 5),
            'hard_degradation': round(hard_val - baseline_val, 5),
        }
        results.append(result)
        print(f'\n  {len(current):2d} replaced | soft d {result["soft_degradation"]:+.4f} | '
              f'hard d {result["hard_degradation"]:+.4f}')

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, 'w') as f: json.dump(results, f, indent=2)

    # Optional final joint coordination of all LGN layers together.
    if getattr(exp_cfg.train, 'joint_polish_steps', 0) > 0:
        current = _logic_indices(live_model)
        print(f'\n{"="*55}\nJOINT POLISH (all {len(current)} LGN layers)\n{"="*55}')
        joint_polish(live_model, current, data, exp_cfg.train, trained_model=trained_default)
        soft_val = estimate_loss(live_model, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
        hard_model = make_hard_model(live_model, current, device)
        hard_val   = estimate_loss(hard_model, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
        results.append({
            'n_replaced':       len(current),
            'replaced_layers':  current,
            'polished':         True,
            'soft_val':         round(soft_val, 5),
            'hard_val':         round(hard_val, 5),
            'soft_degradation': round(soft_val - baseline_val, 5),
            'hard_degradation': round(hard_val - baseline_val, 5),
        })
        print(f'\n  [POLISHED] soft d {results[-1]["soft_degradation"]:+.4f} | '
              f'hard d {results[-1]["hard_degradation"]:+.4f}')
        if save_path:
            with open(save_path, 'w') as f: json.dump(results, f, indent=2)

    # Full metric comparison: original transformer vs final hard LGN model.
    final_idx = _logic_indices(live_model)
    final_hard = make_hard_model(live_model, final_idx, device)
    base_m = estimate_metrics(trained_default, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
    lgn_m  = estimate_metrics(final_hard, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
    print(f'\n{"="*55}\nFINAL METRIC COMPARISON (all {len(final_idx)} layers -> LGN, hard)\n{"="*55}')
    print(f'{"":12} | {"loss":>8} | {"perplexity":>11} | {"accuracy":>9}')
    print(f'{"transformer":12} | {base_m["loss"]:>8.4f} | {base_m["perplexity"]:>11.4f} | {base_m["accuracy"]:>9.4f}')
    print(f'{"LGN (hard)":12} | {lgn_m["loss"]:>8.4f} | {lgn_m["perplexity"]:>11.4f} | {lgn_m["accuracy"]:>9.4f}')
    if save_path:
        mp = os.path.join(os.path.dirname(os.path.abspath(save_path)), 'metrics.json')
        with open(mp, 'w') as f:
            json.dump({'transformer': base_m, 'lgn_hard': lgn_m, 'n_lgn_layers': len(final_idx)}, f, indent=2)

    return results

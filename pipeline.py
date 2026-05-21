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
def get_layer_io(trained_model, layer_idx, data, batch_size=32):
    device = next(trained_model.parameters()).device
    trained_model.eval()
    xb, _ = data.get_batch('train', batch_size)
    pos = torch.arange(xb.size(1), device=device)
    x = trained_model.transformer.drop(
        trained_model.transformer.wte(xb) + trained_model.transformer.wpe(pos)
    )
    for i, block in enumerate(trained_model.transformer.h):
        if i == layer_idx:
            return x.detach(), block(x).detach()
        x = block(x)
    raise RuntimeError(f'layer {layer_idx} not found')


def imitate_layer(logic_model, layer_idx, trained_model, data, cfg, step_mult=1.0):
    logic_layer = logic_model.transformer.h[layer_idx]
    logic_layer.set_temperature(cfg.temp_start)
    trainable = [p for p in logic_layer.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.imitation_lr)
    total_steps = max(1, int(cfg.imitation_steps * step_mult))
    loss_type = getattr(cfg, 'imit_loss', 'mse')
    for step in range(total_steps):
        temp = annealed_temperature(step, total_steps, cfg.temp_start, cfg.temp_end)
        logic_layer.set_temperature(temp)
        logic_layer.train()

        if loss_type == 'mse':
            # Activation-matching imitation
            layer_in, layer_tgt = get_layer_io(trained_model, layer_idx, data, cfg.batch_size)
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
    logic_layer.set_temperature(cfg.temp_end)


def finetune_layers(logic_model, target_indices, data, cfg):
    device = next(logic_model.parameters()).device
    trainable = [p for p in logic_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.finetune_lr)
    for idx in target_indices:
        logic_model.transformer.h[idx].set_temperature(cfg.temp_end)
    # Enable straight-through estimator: forward=hard, backward=soft
    use_ste = getattr(cfg, 'ste', False)
    if use_ste:
        for idx in target_indices:
            logic_model.transformer.h[idx].use_ste = True
        print('    [STE enabled] forward=hard argmax, backward=soft softmax')
    logic_model.train()
    for step in range(cfg.finetune_steps):
        _, lm_loss = logic_model(*data.get_batch('train', cfg.batch_size))
        ent = sum(logic_model.transformer.h[idx].entropy_loss(cfg.ft_ent_conn, cfg.ft_ent_gate)
                  for idx in target_indices)
        (lm_loss + ent).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == cfg.finetune_steps - 1:
            print(f'    ft   {step:4d} | lm {float(lm_loss):.4f} | ent {float(ent):.5f}')
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
    # Restore soft forward so subsequent eval measures the soft model
    if use_ste:
        for idx in target_indices:
            logic_model.transformer.h[idx].use_ste = False


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
            sum_pool=False,      # hybrid uses its own out_proj (no sum-pool support)
            no_in_proj=False,    # hybrid always keeps its own in_proj
            skip_gate=False,     # hybrid does not support skip_gate
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
        ).to(device)
    model.replace_layer(layer_idx, logic_layer)
    for p in model.parameters(): p.requires_grad = False
    for p in model.transformer.h[layer_idx].parameters(): p.requires_grad = True
    return model


def run_heatmap(trained_default, gpt_cfg, data, exp_cfg, save_path=None):
    device = next(trained_default.parameters()).device
    baseline_val = estimate_loss(trained_default, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)
    print(f'Baseline val loss: {baseline_val:.4f}\n')
    results = []

    n = gpt_cfg.n_layer
    for layer_idx in range(n):
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

        print('  [Imitation]')
        imitate_layer(logic_model, layer_idx, trained_default, data, exp_cfg.train, step_mult=step_mult)
        sharp_imit = logic_layer.sharpness()
        val_imit   = estimate_loss(logic_model, data, exp_cfg.train.eval_iters, exp_cfg.train.batch_size)

        print('  [Fine-tune]')
        finetune_layers(logic_model, [layer_idx], data, exp_cfg.train)
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
            sum_pool=False,
            no_in_proj=False,
            skip_gate=False,
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
        ).to(device)
    model.replace_layer(layer_idx, new_layer)
    for p in model.parameters(): p.requires_grad = False
    for idx in _logic_indices(model):
        for p in model.transformer.h[idx].parameters(): p.requires_grad = True


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

        print(f'  [Imitation layer {new_layer_idx}]')
        imitate_layer(live_model, new_layer_idx, trained_default, data, exp_cfg.train, step_mult=step_mult)

        print(f'  [Fine-tune {len(current)} layers]')
        finetune_layers(live_model, current, data, exp_cfg.train)

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

    return results

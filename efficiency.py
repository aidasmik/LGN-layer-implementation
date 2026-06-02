"""Compute per-config efficiency metrics for the LGN report.

LGN's primary value is hardware efficiency (discrete logic gates instead of
float matmuls). We quantify, for each configuration:
- Trainable parameters per block / total
- Number of Boolean gates after hard snap
- Float multiply-add operations (FLOPs) per token forward
- Boolean gate operations per token forward (post-snap inference)
- Memory footprint (float vs hard-discrete)
- Wall-clock inference throughput (tokens/sec on GPU)
- FPGA-relevant: gate count and estimated LUT area
- Effective cost ratios vs the original transformer (the speedup story)
"""

import time
import torch

from lgn import (
    ExperimentConfig, make_gpt,
    LogicGateGPTLayer, HybridLogicGateGPTLayer,
)


def count_params(module, only_trainable=True):
    """Return total parameter count of a module."""
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def count_lgn_gates(model):
    """Count discrete logic gates (= bit_width per LGN layer, summed over LGN blocks/layers).

    Each LearnedLogicLayer produces `out_dim` gates (one Boolean function per output bit).
    A LogicGateGPTLayer holds `depth` such layers in series. Hybrid has the same structure
    for its MLP half (attention is float, doesn't count toward gate count)."""
    total = 0
    for block in model.transformer.h:
        if isinstance(block, (LogicGateGPTLayer, HybridLogicGateGPTLayer)):
            for ll in block.logic:
                total += ll.out_dim
    return total


def block_flops(block, block_size=64, n_embd=128):
    """Approximate float multiply-add count for ONE block, PER TOKEN.

    block_size = sequence length T, used only for the attention term (each token
    attends to T positions). All returned counts are per-token, consistent with
    block_bool_ops and block_io_ops, so model_summary's 'flops_per_token' is honest."""
    T = block_size
    C = n_embd
    flops = 0
    if isinstance(block, (LogicGateGPTLayer, HybridLogicGateGPTLayer)):
        # Hybrid: count frozen attention (Q,K,V,O proj + softmax + value aggregation)
        if isinstance(block, HybridLogicGateGPTLayer):
            # 4 Linear matmuls of shape C x C: 4*C^2 per token
            flops += 4 * C * C
            # Attention: T tokens look at T tokens, dot product C dims: T*C
            flops += T * C
            # Softmax: ~5*T per row
            flops += 5 * T
            # Attention output: T*C (weighted sum)
            flops += T * C
        # LGN MLP path
        if not block.no_in_proj:
            # in_proj: eff_C -> logic_width
            flops += block.eff_C * block.logic_width
        # LGN body: each gate computes f(a,b) which is ~5 flops (1+a+b+ab combination)
        # Number of gates per layer = bit_width = block.logic[0].out_dim
        # In SOFT mode, each gate has 16 logic options summed, so ~16x; in HARD only 1.
        # We're estimating HARD inference cost here.
        for ll in block.logic:
            flops += ll.out_dim * 5  # forward through one gate: ~5 ops
        # Aggregation
        if block.sum_pool:
            # sum_pool: just additions, negligible vs above
            flops += block.logic[0].out_dim  # ~bit_width adds
            if block.learn_pool or block.pool_weighted:
                flops += block.C * (block.group_size if hasattr(block, 'group_size') else 1)
        else:
            # out_proj: bit_width -> C
            flops += block.logic[0].out_dim * C
    else:
        # Original transformer Block: attention + GELU MLP
        # Attention: 4 * C^2 per token + ~T*C
        flops += 4 * C * C + 2 * T * C
        # MLP: typically 4*C hidden, so 2 * (C * 4C) = 8 * C^2 per token
        flops += 8 * C * C
    return flops  # per-token (attention's T-dependence is already inside the sum)


def block_bool_ops(block):
    """Approximate Boolean gate operations per token forward through ONE block.
    Only counts the LGN logic (the truly discrete part). Returns 0 for original blocks."""
    if not isinstance(block, (LogicGateGPTLayer, HybridLogicGateGPTLayer)):
        return 0
    # Each gate evaluates: 1 multiplication of selected bits + truth-table lookup
    # Practically: 2 input lookups + 1 lookup into 4-bit truth table = ~3 ops
    total = 0
    for ll in block.logic:
        total += ll.out_dim * 3  # 3 ops per gate (lookup A, lookup B, eval truth table)
    return total


def block_io_ops(block, block_size=64, n_embd=128):
    """Pre/post processing ops (norm, binarize) per token. Per-token to stay consistent
    with block_flops / block_bool_ops."""
    C = n_embd
    ops = 0
    if isinstance(block, (LogicGateGPTLayer, HybridLogicGateGPTLayer)):
        # LayerNorm: ~5*C per token
        ops += 5 * C
        # If hybrid: ln_1 + ln_2 (two norms)
        if isinstance(block, HybridLogicGateGPTLayer):
            ops += 5 * C
        # Token shift: just index copies, ~0 compute
        # Binarize: 1 compare per scalar; thermometer: n_bits comparisons
        if block.binary_io:
            ops += block.eff_C * block.n_bits  # n_bits threshold checks per position
    else:
        # Original block: just ln_1 + ln_2
        ops += 10 * C
    return ops


def model_memory_bytes(model, dtype_bytes=4):
    """Total model memory: params + buffers in current dtype.

    For LGN HARD models, buffers store discrete connections/coefficients (still
    fp32 in our impl). True deployment can pack to <=1 byte per gate (idx_a/idx_b
    in int16 = 2 B each + 4-bit truth table = 0.5 B), but here we report PyTorch's
    actual footprint."""
    pb = sum(p.numel() * p.element_size() for p in model.parameters())
    bb = sum(b.numel() * b.element_size() for b in model.buffers())
    return {'params_bytes': pb, 'buffers_bytes': bb, 'total_bytes': pb + bb}


def hard_lgn_packed_bytes(gates_total, in_dim=1024):
    """Theoretical packed memory for hard-snapped LGN body (FPGA deployment view):
    each gate stores idx_a (log2(in_dim) bits), idx_b (log2(in_dim) bits), and a
    4-bit truth table = 16 possible gates. With in_dim=1024 -> 10+10+4 = 24 bits/gate.
    Plus per-channel pool affine (cheap), norms (cheap). LGN body only."""
    import math
    bits_per_gate = 2 * math.ceil(math.log2(max(in_dim, 2))) + 4
    return int(gates_total * bits_per_gate / 8)  # bytes


def benchmark_inference(model, gpt_cfg, batch_size=32, n_warmup=20, n_iter=200, device='cuda'):
    """Wall-clock throughput: tokens per second on the current device.

    Note: this measures the SOFT-model inference (float). For HARD inference,
    a separate hard model must be built (we time both when available). The hard
    model is what a real deployment would use — it's strictly smaller / faster."""
    model.eval().to(device)
    xb = torch.randint(0, gpt_cfg.vocab_size, (batch_size, gpt_cfg.block_size), device=device)
    yb = xb.clone()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(xb, yb)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(xb, yb)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    total_tokens = n_iter * batch_size * gpt_cfg.block_size
    return {
        'ms_per_batch':   round(elapsed / n_iter * 1000, 3),
        'tokens_per_sec': int(total_tokens / elapsed),
        'batches_per_sec': round(n_iter / elapsed, 2),
        'batch_size':     batch_size,
        'seq_len':        gpt_cfg.block_size,
    }


def build_hard_version(soft_model, device='cuda'):
    """Wrap soft LGN model in HardLogicGateGPTLayer(s) to get the deployment model."""
    from pipeline import make_hard_model
    lgn_idx = [i for i, b in enumerate(soft_model.transformer.h)
               if isinstance(b, (LogicGateGPTLayer, HybridLogicGateGPTLayer))]
    if not lgn_idx:
        return soft_model  # no LGN to convert (transformer baseline)
    return make_hard_model(soft_model, lgn_idx, device)


def model_summary(model, gpt_cfg):
    """Comprehensive summary: per-block and total."""
    n_layer = gpt_cfg.n_layer
    rows = []
    tot_params = 0
    tot_flops = 0
    tot_gates = 0
    tot_bool_ops = 0
    tot_io_ops = 0
    for i, block in enumerate(model.transformer.h):
        kind = type(block).__name__
        p = count_params(block, only_trainable=False)
        tp = count_params(block, only_trainable=True)
        gates = 0
        if isinstance(block, (LogicGateGPTLayer, HybridLogicGateGPTLayer)):
            for ll in block.logic:
                gates += ll.out_dim
        fl = block_flops(block, gpt_cfg.block_size, gpt_cfg.n_embd)
        bo = block_bool_ops(block)
        io = block_io_ops(block, gpt_cfg.block_size, gpt_cfg.n_embd)
        rows.append({
            'layer': i, 'kind': kind,
            'params_total': p, 'params_trainable': tp,
            'gates': gates, 'flops_per_token': fl,
            'bool_ops_per_token': bo, 'io_ops_per_token': io,
        })
        tot_params += p
        tot_flops += fl
        tot_gates += gates
        tot_bool_ops += bo
        tot_io_ops += io
    # Add embedding + lm_head
    emb_params = count_params(model.transformer.wte, only_trainable=False) + count_params(model.transformer.wpe, only_trainable=False)
    head_params = count_params(model.lm_head, only_trainable=False)
    return {
        'per_block': rows,
        'totals': {
            'block_params': tot_params,
            'embedding_params': emb_params,
            'head_params': head_params,
            'total_params': tot_params + emb_params + head_params,
            'lgn_gates': tot_gates,
            'flops_per_token': tot_flops,
            'bool_ops_per_token': tot_bool_ops,
            'io_ops_per_token': tot_io_ops,
        }
    }


def make_model_for_config(name, device='cuda'):
    """Build a model for a given config name (no training, just for inventory)."""
    cfg = ExperimentConfig()  # aggressive defaults
    if name == 'transformer':
        # Original — no LGN
        model, gpt_cfg = make_gpt(cfg.model, cfg.data, device)
        return model, gpt_cfg, cfg
    # All LGN configs: build model with LGN at all layers
    model, gpt_cfg = make_gpt(cfg.model, cfg.data, device)
    from pipeline import _add_logic_layer
    if name == 'identity':
        cfg.logic.identity_logic = True
        cfg.logic.learn_pool = True  # match the report's identity run (scaling base uses --learn_pool)
        for i in range(gpt_cfg.n_layer):
            _add_logic_layer(model, i, gpt_cfg, cfg.logic, device, trained_default=model)
    elif name == 'aggressive':
        cfg.logic.learn_pool = True
        for i in range(gpt_cfg.n_layer):
            _add_logic_layer(model, i, gpt_cfg, cfg.logic, device, trained_default=model)
    elif name == 'token_shift':
        cfg.logic.learn_pool = True
        cfg.logic.token_shift = 2
        for i in range(gpt_cfg.n_layer):
            _add_logic_layer(model, i, gpt_cfg, cfg.logic, device, trained_default=model)
    elif name == 'hybrid_L0':
        cfg.logic.learn_pool = True
        cfg.logic.hybrid_layers = [0]
        for i in range(gpt_cfg.n_layer):
            _add_logic_layer(model, i, gpt_cfg, cfg.logic, device, trained_default=model)
    elif name == 'combo':
        cfg.logic.learn_pool = True
        cfg.logic.hybrid_layers = [0]
        cfg.logic.token_shift = 2
        for i in range(gpt_cfg.n_layer):
            _add_logic_layer(model, i, gpt_cfg, cfg.logic, device, trained_default=model)
    else:
        raise ValueError(f"unknown config: {name}")
    return model, gpt_cfg, cfg


def main():
    import json
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    configs = ['transformer', 'identity', 'aggressive', 'token_shift', 'hybrid_L0', 'combo']
    results = {}
    for c in configs:
        print(f'\n=== {c} ===')
        try:
            model, gpt_cfg, _ = make_model_for_config(c, device=device)
            s = model_summary(model, gpt_cfg)
            mem_soft = model_memory_bytes(model)
            results[c] = s
            t = s['totals']
            results[c]['memory_soft'] = mem_soft
            results[c]['memory_lgn_packed_bytes'] = hard_lgn_packed_bytes(t['lgn_gates']) if t['lgn_gates'] > 0 else 0

            # Benchmark SOFT inference (training-mode forward, still measures topology cost)
            bench_soft = benchmark_inference(model, gpt_cfg, device=device)
            results[c]['bench_soft'] = bench_soft

            # Benchmark HARD inference where applicable (LGN configs only)
            try:
                hard = build_hard_version(model, device=device)
                bench_hard = benchmark_inference(hard, gpt_cfg, device=device)
                mem_hard = model_memory_bytes(hard)
                results[c]['bench_hard'] = bench_hard
                results[c]['memory_hard'] = mem_hard
            except Exception as e:
                results[c]['bench_hard_error'] = str(e)

            print(f"  trainable params:   {sum(b['params_trainable'] for b in s['per_block']):,}")
            print(f"  total params:       {t['total_params']:,}")
            print(f"  LGN gates:          {t['lgn_gates']:,}  (FPGA LUTs ~ same)")
            print(f"  FLOPs/token:        {t['flops_per_token']:,}")
            print(f"  Bool ops/token:     {t['bool_ops_per_token']:,}")
            print(f"  Memory (soft fp32): {mem_soft['total_bytes']/1024:.0f} KB")
            print(f"  Memory (LGN packed est): {results[c]['memory_lgn_packed_bytes']/1024:.0f} KB")
            print(f"  Soft throughput:    {bench_soft['tokens_per_sec']:,} tok/sec  ({bench_soft['ms_per_batch']:.2f} ms/batch)")
            if 'bench_hard' in results[c]:
                print(f"  Hard throughput:    {results[c]['bench_hard']['tokens_per_sec']:,} tok/sec  ({results[c]['bench_hard']['ms_per_batch']:.2f} ms/batch)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results[c] = {'error': str(e)}

    # Compute speedup table vs transformer (the LGN value story)
    if 'transformer' in results and 'totals' in results['transformer']:
        tfr = results['transformer']
        speedups = {}
        for c, r in results.items():
            if 'error' in r or c == 'transformer':
                continue
            speedups[c] = {
                'params_savings_x':    round(tfr['totals']['total_params'] / max(r['totals']['total_params'], 1), 2),
                'flops_savings_x':     round(tfr['totals']['flops_per_token'] / max(r['totals']['flops_per_token'], 1), 2),
                'memory_savings_x':    round(tfr['memory_soft']['total_bytes'] / max(r['memory_soft']['total_bytes'], 1), 2),
            }
            if 'bench_soft' in r:
                speedups[c]['soft_speedup_x'] = round(r['bench_soft']['tokens_per_sec'] / max(tfr['bench_soft']['tokens_per_sec'], 1), 2)
            if 'bench_hard' in r:
                speedups[c]['hard_speedup_x'] = round(r['bench_hard']['tokens_per_sec'] / max(tfr['bench_soft']['tokens_per_sec'], 1), 2)
        results['_speedup_vs_transformer'] = speedups
        print('\n=== SPEEDUP TABLE (vs transformer) ===')
        for c, s in speedups.items():
            print(f"  {c:20} params {s.get('params_savings_x','?')}x | flops {s.get('flops_savings_x','?')}x | mem {s.get('memory_savings_x','?')}x | hard_speed {s.get('hard_speedup_x','?')}x")

    with open('results/efficiency_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved -> results/efficiency_summary.json')


if __name__ == '__main__':
    main()

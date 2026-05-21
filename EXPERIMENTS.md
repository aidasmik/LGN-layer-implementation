# Experiment log

All runs use the same baseline: 12-layer nanoGPT, n_embd=128, n_head=4, trained on
WikiText-2 byte-level. Baseline validation loss ≈ 1.522–1.537 (varies ±0.01 across
eval runs due to random batches).

Key metric: **hard_degradation** = `hard_val − baseline_val`. Negative = improvement,
positive = worse than baseline. Reported for layers 0, 5 (representative middle),
and 11 (last). Full per-layer JSON in `results/<experiment>/heatmap.json`.

## Summary table

| Experiment            | L0      | L5      | L11     | Note                                       |
|-----------------------|:-------:|:-------:|:-------:|--------------------------------------------|
| Baseline (sigmoid)    | +0.122  | −0.023  | +0.077  | reference                                  |
| tanh activation       | +2.707  | +0.065  | +2.208  | catastrophic; tanh outputs [-1,1] break Boolean polynomial |
| relu activation       | +1.608  | +0.019  | +0.424  | also bad; unbounded positive               |
| init_scale 0.1        | +0.122  | −0.038  | +0.080  | no effect; temperature annealing dominates |
| dropout 0.05          | +0.157  | +0.014  | +0.138  | worst of both worlds                       |
| dropout 0.1           | −0.018  | −0.022  | +0.059  | helps; baseline is also weaker             |
| per_layer_anneal      | +0.156  | −0.044  | +0.089  | extra steps for edges don't help           |
| edge_d3w4 (3-deep, 4× wide) | +0.277 | −0.020 | +0.155 | bigger is WORSE — snap errors compound through depth |
| protected (0,11 excluded) | n/a | n/a | n/a    | greedy scaling caps cleanly at 10 layers   |
| KL imitation loss     | +0.079  | −0.021  | +0.079  | helps L0 partially; no effect on L11       |
| STE during fine-tune  | +0.085  | −0.022  | +0.079  | eliminates soft-to-hard gap (gap goes ≤ 0) |
| **Hybrid attention L0** | **−0.014** | n/a | n/a  | **completely fixes L0**                    |
| **Identity ablation (hybrid L0)** | **−0.011** | −0.025 | **+0.058** | LearnedLogicLayer = identity → sigmoid-bottleneck MLP. **Beats learned logic at every layer.** |

## Key findings

1. **L0 problem is structural, not capacity.** Adding more depth or width to the
   L0 logic stack makes things worse (edge_d3w4: +0.277 vs +0.122). The issue is
   that pointwise logic cannot do cross-token mixing.

2. **Hybrid attention is the right fix for L0.** Copy the trained attention
   sublayer (frozen) and replace only the MLP with logic. L0 hard_degradation
   goes from +0.122 to −0.014.

3. **L11 problem is different — output sensitivity.** Errors at L11 propagate
   directly to logits via the small `ln_f + lm_head` head. KL imitation didn't
   help; STE didn't help; bigger layers didn't help. L11 has a floor of about
   +0.077 with pointwise logic.

4. **STE removed the soft-to-hard gap.** With STE during fine-tuning, the hard
   model became as good or better than the soft model across most layers
   (gap went from +0.01..+0.06 to −0.03..0.0). But this didn't translate to
   beating baseline at L11 — confirming that L11's residual degradation is
   expressivity-limited, not snap-related.

5. **Sigmoid is uniquely suited to Boolean polynomial.** Tanh and relu
   catastrophically break the hard snap because the gate polynomial
   `c₀ + c₁A + c₂B + c₃AB` assumes A,B ∈ [0,1]. Tanh gives [-1,1], relu gives
   unbounded positive — both shatter the discrete behavior.

6. **Greedy ordering matters.** Replacing middle layers first lets the model
   adapt; edges last because they hurt most when forced.

7. **Logic gates contribute negatively in most layers (identity ablation).**
   Replacing `LearnedLogicLayer.forward` with identity (so the layer is just
   `norm → in_proj → sigmoid → out_proj → +residual`) gives equal or better
   results than learned logic at every layer except L0. At L11 the improvement
   is dramatic: hard_degradation drops from +0.077 (baseline) and +0.093 (full
   hybrid_L0) to **+0.058** with identity. The two linear projections plus
   sigmoid bottleneck are doing 90%+ of the work; the actual Boolean gates
   constrain each neuron to one of 16 functions on 2 random wires, which is
   strictly less expressive than what the sigmoid bottleneck carries naturally.
   The logic story works conceptually only because the architecture surrounding
   the gates is doing the real work.

## Implementation knobs added

The codebase now exposes these via CLI:

| Flag                  | Effect                                                          |
|-----------------------|-----------------------------------------------------------------|
| `--activation`        | sigmoid / tanh / relu / hardsigmoid / none                      |
| `--conn_init_scale`   | initialization stddev for connection logits                     |
| `--gate_init_scale`   | initialization stddev for gate logits                           |
| `--dropout`           | model dropout (applied during baseline training and fine-tune)  |
| `--edge_depth`        | depth override for edge layers (0 and n_layer-1)                |
| `--edge_width_mult`   | width override for edge layers                                  |
| `--per_layer_anneal`  | scale imitation steps by layer position (edges get 2x)          |
| `--hybrid_layers`     | layer indices where attention is kept frozen, only MLP replaced |
| `--protected_layers`  | (scale only) layers excluded from replacement                   |
| `--imit_loss`         | mse (activation match) or kl (output distribution match)        |
| `--ste`               | straight-through estimator during fine-tuning                   |
| `--ft_eval_hard`      | periodically eval hard model during fine-tune                   |
| `--ft_log_sharpness`  | print per-layer sharpness during fine-tune                      |

## Architecture variants in `lgn.py`

- `LogicGateGPTLayer` — original: replaces full transformer block
- `HybridLogicGateGPTLayer` — keeps frozen attention from baseline, replaces only MLP
- `HardLogicGateGPTLayer` / `HardHybridLogicGateGPTLayer` — discrete versions for inference

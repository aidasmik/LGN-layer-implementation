# LGN Final Report — Outline

## 0. Status

**Code:** all 10 bug fixes applied (8 from review + B10 curriculum + Hybrid extended to true aggressive). Syntax OK, smoke-tested.

**Experiments running:** 5 cumulative-scaling configs with fixed code (`results/report/<name>/`):
- `identity` — control (LGN = identity)
- `aggressive` — pure LGN baseline
- `token_shift_k2` — pure LGN + local cross-token
- `hybrid_L0_agg` — **TRUE aggressive hybrid** (frozen attention + aggressive LGN MLP, NOT linear sandwich)
- `combo` — hybrid L0 + token shift K=2

ETA ~2.5–3 hours.

---

## 1. LGN — what it is and how we use it

### 1.1 The single gate
- 2 binary inputs A, B → 1 binary output
- Universal Boolean polynomial: `g(A,B) = c₀ + c₁·A + c₂·B + c₃·A·B`
- 16 Boolean gates expressible via 4 coefficients (see `LOGIC_GATE_MATRIX`)
- Soft training: `gate_logits` (16-dim softmax) → which gate to use
- Hard inference: argmax → one Boolean gate per position

### 1.2 The LGN block (`LogicGateGPTLayer`)
Architecture (aggressive default):
```
x → LayerNorm → [token_shift] → activation → binarize/thermometer
   → LearnedLogicLayer × depth
   → group-sum (sum_pool) → +residual
```

**Trainable parameters per LGN layer (LearnedLogicLayer):**
| Parameter | Shape | Purpose |
|---|---|---|
| `conn_logits_a` | `(out_dim, k)` | Which of k=4 candidates → input A |
| `conn_logits_b` | `(out_dim, k)` | Which of k=4 candidates → input B |
| `gate_logits`   | `(out_dim, 16)` | Which Boolean gate (16-way softmax) |

**With aggressive flags (defaults):** `out_dim = bit_width = C·n_bits·(K+1)` where C=128, n_bits=8, K=token_shift.
- Without token shift: 1024 gates per LGN layer
- With token_shift K=2: 3072 gates per LGN layer

### 1.3 Training pipeline (`pipeline.py`)
Per layer replacement:
1. **Imitation** (200 steps): MSE on the LIVE upstream distribution against the ORIGINAL layer's output. (Live distribution = uses LGN-replaced layers if they exist — fixed via `input_model=live_model`.)
2. **Fine-tune** (3000 steps): LM cross-entropy + entropy regularizer on gate/connection logits.
3. **Temperature annealing**: softmax temperature 2.0 → 0.1 over training. Only the newly-added layer anneals; previously-settled layers stay sharp at 0.1.
4. **Hard snap**: argmax connections, argmax gates → discrete Boolean circuit.
5. **Frozen base**: attention sublayers, embeddings, lm_head are NEVER trained during LGN fine-tuning — all `hard_degradation` numbers are pure LGN cost.

### 1.4 Key flags / variants we test
| Flag | Effect |
|---|---|
| `--binary_io --n_bits 8` | Thermometer encoding: each scalar → 8 bits via STE thresholds |
| `--no_in_proj` | Skip the trained Linear before LGN |
| `--sum_pool` | Replace `out_proj` Linear with fixed group-sum aggregation |
| `--learn_pool` | Add learnable per-channel affine after group-sum (cheap) |
| `--token_shift K` | Causal channel-aligned shift: position t sees [t-K..t] |
| `--hybrid_layers 0` | At L0, keep frozen baseline attention (only MLP becomes LGN) |
| `--iwp` | Light DLGN 2025 Input-Wise Parametrization (4 weights instead of 16-gate softmax) |
| `--gumbel_ste` | Mind the Gap 2025 stochastic discrete training |

---

## 2. Efficiency: where LGN earns its name

| Config | Trainable params | FLOPs/token | LGN gates | Bool ops/token | vs transformer |
|---|---:|---:|---:|---:|---|
| Transformer (baseline) | 2.38 M | **163.6 M** | 0 | 0 | 1.0× |
| Identity (control) | 0.30 M | 4.7 M | 12,288 | 36,864 | **34.8× fewer FLOPs** |
| Aggressive | 0.30 M | 5.5 M | 12,288 | 36,864 | **29.7× fewer FLOPs** |
| Token shift K=2 | 0.89 M | 16.5 M | 36,864 | 110,592 | 9.9× fewer FLOPs |
| Hybrid L0 (aggressive) | 0.30 M | 10.8 M | 12,288 | 36,864 | 15.1× fewer FLOPs |
| Combo (Hybrid + tshift) | 0.89 M | 21.8 M | 36,864 | 110,592 | 7.5× fewer FLOPs |

**Key takeaway:** the LGN trade is **30× FLOPs reduction for ~half the accuracy**, with aggressive setup. Hybrid configurations soften the FLOP gain (attention is float-heavy) but recover accuracy. The **gate count** (12K–37K Boolean ops/token) is the actual hardware-relevant metric — these are 2-input look-up table operations, ideal for FPGA / ASIC inference.

---

## 3. Results — per-layer difficulty

**Method:** replace ONE layer at a time, measure `hard_degradation` vs original transformer.

Per-layer aggressive results (`fig 01_per_layer_difficulty.png`):

| Layer | hard_degradation | Difficulty |
|---|---:|---|
| L0 | **+1.058** | Severe (cross-token mixing required) |
| L1 | +0.046 | Easy |
| L2 | +0.016 | Easy |
| L3 | +0.012 | Easy |
| L4–L9 | +0.030 — +0.150 | Easy to Moderate |
| L10 | +0.181 | Hard |
| L11 | +0.367 | Hard (lm_head precision) |

**Pattern:** boundary layers (L0, L11) dominate the difficulty. Middle layers are nearly trivial — most have hd < 0.1 nat.

---

## 4. Results — cumulative scaling (replacing all layers)

**Method:** greedy ordering (easiest first per heatmap); replace one by one; finetune after each.

Refer to `fig 02_cumulative_scaling.png`.

Final n=12 accuracy (frozen base, fixed code, all configs):

| Config | Accuracy | Loss | Perplexity |
|---|---:|---:|---:|
| Transformer (ceiling) | **54.9%** | 1.54 | 4.67 |
| Identity (no LGN) | **TBD** (running) | — | — |
| Aggressive (pure LGN) | **TBD** | — | — |
| Token shift K=2 | **TBD** | — | — |
| Hybrid L0 (true aggressive) | **TBD** | — | — |
| Combo | **TBD** | — | — |

Numbers will be filled in once `results/report/<config>/metrics.json` is available.

---

## 5. Why L0 cannot be replaced by pointwise LGN

### 5.1 Empirical evidence (ours)
- L0 single-layer hard_degradation: **+1.06 nat** (vs 0.01–0.37 for any other layer)
- Hybrid L0 (with frozen attention restored): reduces overall degradation by ~0.5 nat
- Token shift K=2 alone: helps but does not fully recover L0

### 5.2 Mechanistic reason
- **Attention is cross-token, content-based mixing**: token i looks at all other tokens via Q·Kᵀ, weighted by softmax, then aggregates Values
- **LGN is pointwise**: each output position depends ONLY on its own input position (via 2 selected bits)
- L0 sees raw embeddings; without attention, no token has "seen" any other → no contextual information has been mixed in yet
- A pointwise function on raw embeddings cannot produce contextualized representations

### 5.3 Literature support
| Paper | Finding |
|---|---|
| Half the Nonlinearity is Wasted (arxiv 2603.03459, 2026) | In 162M–2.8B transformers, **boundary layers consume the majority of MLP nonlinearity budget**; middle layers are near-linear and ~40% can be replaced by a single Linear with <3% perplexity cost |
| Discrete Charm of MLP (arxiv 2603.10985, 2026) | MLPs do **binary routing of continuous signals**; L0 specifically uses gateway neurons that route exceptions — fundamentally different from middle layers |
| Recurrent DDLGN (arxiv 2508.06097, 2025) | Logic gates with **internal state** can do sequence modeling — closest known route to "cross-token via LGN" (replaces attention with stateful gates, not parallel attention) |
| BitNet b1.58 (Microsoft, 2024) | Industry uses **ternary weights** (not Boolean gates) for transformer compression — keeps full attention + LM head as float |

### 5.4 Three solutions we tested
1. **Hybrid L0** (frozen attention + aggressive LGN MLP) — accuracy bump
2. **Selective LGN** (keep transformer at L0, L11) — efficiency-quality tradeoff
3. **Token shift K=2** — gives LGN local cross-token receptive field

---

## 6. What we can replace, and what we gain

### 6.1 Utilization map — where the LGN actually does work
Refer to `fig 04_lgn_utilization.png`. Identity ablation per layer reveals only **4 of 12 layers** carry meaningful LGN load:
- **Active (>0.02 nat)**: L0 (+0.240), L7 (+0.020), L10 (+0.024), L11 (+0.025)
- **Inactive (<0.02 nat)**: L1, L2, L3, L4, L5, L6, L8, L9 — could be replaced by identity for negligible cost

### 6.2 Selective LGN curve
Refer to `fig 07_selective_curve.png`.

| Transformer layers kept | LGN layers | Accuracy |
|---:|---:|---:|
| 0 | 12 | 27.2% |
| 1 (L0) | 11 | 34.9% |
| 2 (L0, L11) | 10 | 36.4% |
| 4 (L0, L1, L10, L11) | 8 | 38.9% |
| 12 (all) | 0 | 54.9% |

**Diminishing returns:** first transformer layer kept (L0) buys +7.7 ppt; each additional buys less. Sweet spot near 2–4 layers kept.

### 6.3 Recommendation
- **For maximum efficiency** (30× FLOP reduction, ~50% accuracy retained): use the **aggressive + hybrid L0** configuration
- **For balanced quality** (~70% of transformer accuracy at ~3× speedup): use **selective LGN** keeping L0 and L11 as transformer
- **For maximum quality at modest savings**: keep most layers transformer; use LGN only in 4–6 middle layers

---

## 7. Code-level summary (key files)
| File | Purpose |
|---|---|
| `lgn.py` | Core LGN classes: LearnedLogicLayer, LogicGateGPTLayer, HybridLogicGateGPTLayer + Hard variants |
| `pipeline.py` | Training: imitation, fine-tune, scaling, hard-snap evaluation |
| `run.py` | CLI for heatmap and scale commands |
| `efficiency.py` | Parameter/FLOP/gate counter for each config |
| `run_report_experiments.py` | Re-runs the 5 report configurations with fixed code |
| `plot_report_final.py` | Generates the 8 English-language report figures |

---

## 8. Open questions / future work

1. **Stateful gates** (RDDLGN-style) — replace attention with logic-gate flip-flops to make a TRULY pure-LGN transformer-replacement
2. **Light DLGN reparametrization** properly implemented (avoiding sigmoid saturation) — could close vanishing-gradient gap at depth 12+
3. **Co-training the base** (different goal — for deployment, not pure-LGN measurement)
4. **FPGA inference benchmark** — verify the ~30× FLOP reduction translates to actual hardware speedup

# LGN as a Transformer-Sublayer Replacement — Complete Report

**Everything done, every result, and the literature that justifies the ceiling.**

Project: `LGN_Nano` — replacing transformer sublayers with Differentiable Logic Gate
Networks (LGN) in a byte-level autoregressive language model (nanoGPT, WikiText-2),
under a strict honesty protocol.

Repo: https://github.com/aidasmik/LGN-layer-implementation
Date: 2026-06-02. Two experiments (`dilated_124816`, `tshift2_polish_kl`) still running → marked `[pending]`.

---

## 0. TL;DR

- **Transformer ceiling:** 54.87 % next-byte accuracy.
- **Best pure-LGN (all 12 layers Boolean):** **36.45 %** = 66 % of transformer (`combo` = hybrid-L0 + token-shift K=2).
- **Best with a few transformer layers kept:** **39.01 %** = 71 % (`sel_4edges`).
- **Honest floor (no cross-token aid):** 27.22 % (`aggressive`).
- **The ceiling is set by ONE thing: the L0 cross-token bottleneck.** A pointwise Boolean
  function cannot do attention's content-based all-to-all token mixing. Every technique
  that helped attacked that; every gate-level trick (depth, width, conv, linear, IWP,
  Gumbel) failed or was a measurement artifact.
- **The literature confirms this** (see §11): the only prior LGN sequence model (RDDLGN,
  2025) had to introduce *stateful/recurrent* gates precisely because pointwise gates
  cannot mix sequence positions; every strong LGN result (MNIST 98 %, CIFAR 86 %) is on
  *images*, where there is no cross-token problem.

---

## 1. Setup

- **Model:** nanoGPT, 12 layers × 128 dim × 4 heads, byte-level (vocab 256), block 64, WikiText-2.
- **Transformer baseline (frozen reference):** loss 1.541, perplexity 4.67, **accuracy 54.87 %**.
- **LGN gate:** 2-input Boolean polynomial `g(A,B) = c₀ + c₁A + c₂B + c₃AB`, 16-way gate
  softmax + connection softmax over k=4 candidates. Soft training → argmax hard snap.
- **"Aggressive" setup (default, the honest one):** `--binary_io --n_bits 8` (thermometer),
  `--no_in_proj` (no trained Linear before), `--sum_pool` (fixed group-sum after),
  `--learn_pool` (cheap per-channel affine). No float matmul inside the block.

## 2. Training pipeline

Per-layer cumulative replacement, greedy easiest-first by per-layer difficulty:
1. **Imitation** (200 steps): MSE to the original layer's output on the *live* upstream distribution.
2. **Fine-tune** (3000 steps): LM cross-entropy + entropy reg, temperature annealed 2.0 → 0.1 (only the new layer anneals; settled layers stay sharp).
3. **Hard snap:** argmax → discrete Boolean circuit.
4. **Frozen base throughout:** attention, embeddings, lm_head never trained → every number is pure LGN cost.

## 3. Methodology — three honesty guards

1. **Frozen base** — the LGN cannot cheat by retraining the surrounding network.
2. **Identity ablation** (`--identity_logic`) — the **fake-LGN detector**. For each
   architecture, a twin with the LGN body replaced by pass-through. `LGN contribution =
   hd(identity) − hd(real)`. If equal → the LGN is decorative and the surrounding
   machinery does the work. This caught every conv/linear "win" as fake (§7).
3. **Two-phase screening** — cheap 4-layer screen (L0,L5,L10,L11; 100 imit + 500 ft;
   ~5 min) before committing a config to full 12-layer scaling (~3 h). Caveat:
   per-layer screening is *insensitive to token-mixing* (token-shift's benefit is
   system-level), so cross-token ideas are confirmed at full scale.

---

## 4. The difficulty map (per-layer, single-layer replacement, aggressive)

| Layer | hard_degradation (nat) | Role |
|---|---:|---|
| **L0** | **+0.90 … +1.06** | cross-token mixing — severe |
| L1–L9 | +0.00 … +0.15 | near-trivial (some ≈ identity) |
| L10 | +0.16 | moderate |
| **L11** | **+0.30 … +0.37** | lm_head precision — hard |

The entire gap lives at the boundaries. **L0 alone costs more than all eight middle
layers combined.** This single fact explains every result below.

---

## 5. FULL production results — all 24 cumulative-scaling configs (12 layers each)

Next-byte accuracy, frozen base, fixed code. Sorted best → worst.

| # | Config | Accuracy | Loss | PPL | What it is |
|---:|---|---:|---:|---:|---|
| — | **Transformer** | **54.87 %** | 1.541 | 4.67 | ceiling (frozen reference) |
| 1 | sel_4edges | 39.01 % | 2.101 | 8.17 | 8 LGN + 4 transformer (L0,L1,L10,L11) |
| 2 | sel_edges | 37.03 % | 2.196 | 8.98 | 10 LGN + 2 transformer (L0,L11) |
| 3 | **combo** | **36.45 %** | 2.236 | 9.35 | hybrid-L0 + token-shift K=2 (best pure-LGN) |
| 4 | token_shift_k2 | 36.22 % | 2.251 | 9.49 | contiguous shift [1,2] |
| 5 | token_shift_k3 | 36.13 % | 2.259 | 9.58 | shift [1,2,3] |
| 6 | tshift2_cage | 35.91 % | 2.251 | 9.49 | token-shift K=2 + CAGE |
| 7 | dilated_124 | 35.62 % | 2.258 | 9.57 | dilated shift [1,2,4] |
| 8 | token_shift_k1 | 35.16 % | 2.279 | 9.76 | shift [1] |
| 9 | token_shift_k2_reverse | 35.07 % | 2.277 | 9.74 | K=2, reverse-greedy order |
| 10 | sel_L0 | 34.70 % | 2.259 | 9.57 | 11 LGN + 1 transformer (L0) |
| 11 | hybrid_L0_agg | 33.46 % | 2.341 | 10.39 | frozen attention at L0 |
| 12 | hybrid_edges | 33.44 % | 2.336 | 10.33 | frozen attention at L0 & L11 |
| 13 | aggressive_n16 | 27.57 % | 2.514 | 12.35 | 16 bits/scalar |
| 14 | aggressive_s42 | 27.26 % | 2.545 | 12.74 | seed 42 |
| 15 | **aggressive** | **27.22 %** | 2.536 | 12.63 | honest pure-LGN floor |
| 16 | aggressive_cage | 27.03 % | 2.535 | 12.61 | CAGE (gap closes, acc flat) |
| 17 | aggressive_s7 | 26.90 % | 2.546 | 12.76 | seed 7 |
| 18 | aggressive_n4 | 26.38 % | 2.575 | 13.13 | 4 bits/scalar |
| 19 | depth2_rand | 26.32 % | 2.574 | 13.11 | depth 2, 1 learn + 1 random |
| 20 | depth2_learn | 25.86 % | 2.573 | 13.11 | depth 2, all learnable |
| 21 | depth4_learn | 25.52 % | 2.627 | 13.83 | depth 4, all learnable |
| 22 | depth4_rand | 25.31 % | 2.633 | 13.92 | depth 4, reservoir |
| 23 | identity | 23.25 % | 2.784 | 16.18 | control (LGN = pass-through) |
| 24 | iwp_fixed | 22.17 % | 2.691 | 14.75 | Input-Wise Parametrization (Light DLGN) |
| — | dilated_124816 | `[pending]` | — | — | dilated shift [1,2,4,8,16] |
| — | tshift2_polish_kl | `[pending]` | — | — | token-shift K=2 + joint polish + KL distill |

**Variance (3 seeds of aggressive):** 27.13 % ± 0.19 pp — results are stable, not lucky seeds.

### 5.1 Token-shift K sweep (the cross-token breakthrough)
| K | Accuracy | Δ vs aggressive |
|---:|---:|---:|
| 0 | 27.22 % | — |
| 1 | 35.16 % | +7.9 |
| **2** | **36.22 %** | **+9.0** |
| 3 | 36.13 % | +8.9 |

### 5.2 Selective-LGN curve
| Transformer layers kept | LGN layers | Accuracy |
|---:|---:|---:|
| 0 | 12 | 27.22 % |
| 1 (L0) | 11 | 34.70 % |
| 2 (L0,L11) | 10 | 37.03 % |
| 4 (L0,L1,L10,L11) | 8 | **39.01 %** |
| 12 | 0 | 54.87 % |

### 5.3 n_bits sweep
4 → 26.38 %, 8 → 27.22 %, 16 → 27.57 %. Within noise; granularity barely matters.

---

## 6. Screening + identity ablation (per-layer hard_degradation, Σ over L0,L5,L10,L11)

Lower Σ = better. `aggressive` Σ = 1.348 is the reference.

| Config | L0 | L5 | L10 | L11 | Σhd |
|---|---:|---:|---:|---:|---:|
| conv3_hybridL0_identity | 0.008 | -0.038 | 0.029 | 0.074 | 0.073 |
| conv3_identity | 0.014 | -0.039 | 0.030 | 0.074 | 0.079 |
| linear_proj_identity | 0.107 | -0.036 | 0.032 | 0.095 | 0.198 |
| linear_proj | 0.242 | -0.027 | 0.074 | 0.152 | 0.441 |
| conv3_hybridL0 | 0.432 | -0.006 | 0.116 | 0.206 | 0.749 |
| conv3 | 0.452 | -0.007 | 0.117 | 0.205 | 0.767 |
| conv3_in_only | 0.608 | -0.028 | 0.067 | 0.150 | 0.797 |
| combo_cage_binreg | 0.441 | -0.011 | 0.137 | 0.272 | 0.839 |
| tshift2_cage | 0.455 | -0.012 | 0.135 | 0.272 | 0.850 |
| dilated_1248_cage | 0.478 | -0.014 | 0.135 | 0.269 | 0.868 |
| conv5 | 0.518 | 0.009 | 0.157 | 0.305 | 0.989 |
| hybridL0_cage | 0.546 | 0.000 | 0.157 | 0.300 | 1.002 |
| aggressive_cage | 0.630 | 0.000 | 0.157 | 0.300 | 1.086 |
| conv7 | 0.614 | 0.006 | 0.176 | 0.315 | 1.111 |
| dilated_124 | 0.853 | -0.013 | 0.137 | 0.279 | 1.257 |
| **aggressive** | **0.897** | 0.000 | 0.155 | 0.296 | **1.348** |
| aggressive_binreg020 | 0.957 | 0.003 | 0.159 | 0.310 | 1.429 |
| aggressive_gumbel | 1.473 | 0.035 | 0.196 | 0.392 | **2.096** (worst) |

### 6.1 Identity ablation — the fake-LGN detector
`LGN contribution = hd(identity twin) − hd(real)`. Large positive = the Boolean gates do real work.

| Architecture | LGN contribution | Verdict |
|---|---:|---|
| aggressive | **+0.48** | ✅ REAL |
| aggressive_cage | **+0.75** | ✅ REAL (CAGE *increases* real LGN work) |
| token_shift K=2 + CAGE | **+1.14** | ✅ REAL (strongest) |
| conv3 | **−0.69** | ❌ FAKE — the conv kernel does the work |
| conv3_in_only | −0.69 | ❌ FAKE |
| conv3_hybridL0 | −0.68 | ❌ FAKE |
| linear_proj | −0.24 | ❌ FAKE |

`conv3` with a real LGN: L0 hd = 0.452. `conv3` with an **identity** LGN: L0 hd = **0.014**.
The Conv1d kernel solves L0 *by itself*; the LGN only adds noise. Conv/linear re-introduce
the trained float transformation we deliberately removed → they are not honest LGN.

---

## 7. What works vs what doesn't

### Works (honest, verified by identity ablation)
| Technique | Effect | Mechanism |
|---|---:|---|
| Selective LGN (keep L0/L11) | up to **+11.8 pp** | sidesteps the boundary bottleneck |
| Token-shift K=2 | **+9.0 pp** | local cross-token window for pointwise LGN |
| Hybrid L0 | **+6.2 pp** | restores real attention exactly where needed |
| CAGE | gap −50 % (acc flat) | closes soft→hard discretization gap by construction |

### Does not work
| Technique | Result | Why |
|---|---|---|
| Conv / Linear projections | "best" hd but **FAKE** | projection does the work, LGN decorative |
| Depth + random interconnect | −1 … −2 pp | hard-snap error accumulates across Boolean layers |
| Gumbel-STE (Mind the Gap) | **worst** (Σhd 2.10) | gap needs hard forward, not Gumbel noise |
| IWP (Light DLGN) | −5 pp | image conv-LGN technique, hurts byte-LM |
| Binary regularization (RDDLGN) | flat / −0.1 | tuned to their sigmoid encoder, not ours |
| Reverse greedy (hard-first) | −1.15 pp | easy-first lets the net adapt to LGN noise gradually |
| Hash embedding L0 | removed (buggy) | train/eval forward mismatch → invalid measurement |
| Dilated token shift | ~flat / −0.6 pp | contiguous K=2 already the sweet spot |

**The one structural lesson:** every win came from **cross-token receptive field**, never
from making the pointwise Boolean computation richer.

---

## 8. CAGE — discretization gap (Align Forward, Adapt Backward, 2026)

Hard forward (argmax = inference) + adaptive backward temperature based on commitment.

| Config (12 layers) | soft_deg | hard_deg | gap |
|---|---:|---:|---:|
| aggressive | +0.990 | +1.016 | +0.027 |
| **aggressive_cage** | +0.998 | +1.012 | **+0.014** (−48 %) |
| token_shift_k2 | +0.641 | +0.744 | +0.103 |
| **tshift2_cage** | +0.692 | +0.739 | **+0.047** (−54 %) |

CAGE does exactly what its paper claims — halves the gap — but our baseline gap was
already small, so final accuracy is unchanged. It is an honest-training guarantee, not
an accuracy lever, in this regime.

---

## 9. Efficiency (FPGA-relevant payoff)

Per-token, from `efficiency.py`.

| Config | Params | FLOPs/token | LGN gates | Bool ops/token | FLOPs vs transformer |
|---|---:|---:|---:|---:|---:|
| Transformer | 2.45 M | 2.56 M | 0 | 0 | 1.0× |
| **Aggressive** | 0.37 M | 0.086 M | 12,288 | 36,864 | **29.7× fewer** |
| Hybrid L0 | 0.44 M | 0.168 M | 12,288 | 36,864 | 15.2× fewer |
| Token-shift K=2 | 0.96 M | 0.258 M | 36,864 | 110,592 | 9.9× fewer |
| Combo | 1.03 M | 0.340 M | 36,864 | 110,592 | 7.5× fewer |

- Aggressive: **6.5× fewer params, ~30× fewer FLOPs/token.**
- Hardware metric = gate count (12K–37K 2-input Boolean LUTs/token → one FPGA LUT each).
- **GPU caveat:** on GPU the LGN is *slower* (no optimized kernel for discrete gather +
  gate eval). The advantage is theoretical / FPGA-realized, matching Petersen 2024's
  1900× being specialized-hardware inference, not stock GPU.

---

## 10. The ceiling — explicit statement

| Question | Answer |
|---|---|
| Best pure LGN (all 12 Boolean)? | **36.45 %** = 66 % of transformer |
| Best with a few transformer layers? | **39.01 %** = 71 % |
| Honest floor (no cross-token aid)? | 27.22 % = 50 % |
| Absolute floor (residual + frozen base only)? | 23.25 % = 42 % |
| What sets the ceiling? | the **L0 cross-token bottleneck** — pointwise Boolean gates cannot do content-based all-to-all token mixing |
| Can more/better gates raise it? | **No** — depth, width, conv, linear, IWP all failed or were fake; only cross-token mechanisms help |
| Does SOTA discretization (CAGE) help? | halves the gap honestly, but does **not** raise the ceiling |

**Theory:** attention computes, for token *i*, a content-weighted aggregate over *all*
positions (softmax(QKᵀ)·V). A pointwise function of the residual stream at position *i*
depends only on position *i*. No depth or width of pointwise Boolean gates can reconstruct
all-to-all mixing — it is the wrong function class. Token-shift/hybrid work precisely
because they *add* a (fixed or attention-based) cross-token channel.

---

## 11. Literature — and how it confirms / justifies the ceiling

### 11.1 LGNs are an image technique; sequences are different
| Work | Result | Bearing on our ceiling |
|---|---|---|
| **Petersen et al. 2022, Deep DLGN** (NeurIPS) — arXiv 2210.08277 | MNIST ~97–99 % w/ 8K–64K gates | All on **images** — fixed grid, **no cross-token problem**. Establishes LGNs as a classification tool, not a sequence-mixing one. |
| **Petersen et al. 2024, Convolutional DLGN** (NeurIPS) — arXiv 2411.04732 | CIFAR-10 **86.29 %** w/ 61M gates; 1900× faster inference | Conv *spatial* structure on images. We tested conv in LM → fake-LGN (§6.1). Their speedup is specialized hardware, justifying our GPU caveat (§9). |
| **LILogicNet 2025** — arXiv 2511.12340 | MNIST 98.45 % w/ 8K gates | Again images; compact learnable connectivity. No sequence result exists in this line. |

> **Justification:** every strong LGN number in the literature is on images, where the
> input is a fixed-size grid and there is *no* token-mixing requirement. Our byte-LM
> setting introduces exactly the operation pointwise LGNs lack — content-based cross-token
> mixing — which is why a hard ceiling appears that image benchmarks never expose.

### 11.2 The only prior LGN sequence model needed STATEFUL gates — directly confirming the wall
| Work | Result | Bearing on our ceiling |
|---|---|---|
| **Recurrent Deep DLGN (RDDLGN) 2025** — arXiv 2508.06097 | WMT'14 EN→DE: 5.00 BLEU train / **4.39 BLEU** hard (GRU 5.41); discretization 30.9 % → 27.7 % | **The only prior LGN applied to sequences.** Crucially, they had to make the gates **stateful (flip-flops/latches)** — i.e. recurrent — because pointwise/parallel logic gates *cannot* mix across sequence positions. This is the same wall we hit at L0, from the opposite direction: they added state to break it; we showed parallel pointwise gates cannot. Their hard-snap gap (−3.2 pp) matches ours in magnitude. |

> **This is the strongest external justification of our ceiling:** an independent group,
> to make LGNs work on sequences at all, had to abandon the pointwise/parallel form and
> introduce recurrence. Our negative results on depth/conv/width are the parallel-form
> confirmation that no amount of pointwise capacity substitutes for cross-token state.

### 11.3 Discretization-gap literature — explains why gate-level tricks can't raise the ceiling
| Work | Claim | What we found |
|---|---|---|
| **Mind the Gap 2025** — arXiv 2506.07500 | Gumbel-STE closes the soft→hard gap on images | We tested → **worst result** (Σhd 2.10). |
| **Align Forward, Adapt Backward (CAGE) 2026** — arXiv 2603.14157 | The gap is closed by a **hard forward**, not by Gumbel noise; adaptive backward temp; 30–76× faster than Gumbel-ST | We confirmed: CAGE halves the gap honestly (§8); Gumbel-STE alone fails — exactly CAGE's thesis. But closing the gap does **not** raise accuracy when the gap is already small, so discretization is *not* what caps us — cross-token capacity is. |
| **Light DLGN (IWP) 2025** — arXiv 2510.03250 | 4-weight reparam, 8.5× faster convergence at depth | We tested → −5 pp; helps deep image conv-LGNs, not our byte setup. |

> **Justification:** the most recent gap-closing methods (CAGE) provably remove the
> discretization penalty in our model, yet accuracy does not move. This isolates the
> ceiling as an **expressivity/architecture** limit (cross-token), not a training or
> discretization artifact.

### 11.4 MLP-structure literature — why the cost concentrates at the boundaries
- **"Half the Nonlinearity is Wasted"** (2026) — in large transformers, boundary layers
  consume most of the MLP nonlinearity budget; ~40 % of middle layers can be replaced by a
  single Linear with <3 % perplexity cost. → Matches our difficulty map exactly: L0/L11
  hard, L1–L9 near-trivial.
- **"Discrete Charm of the MLP"** (2026) — L0 uses gateway neurons routing exceptions,
  structurally different from middle layers. → Explains why L0 is uniquely hard for a
  pointwise replacement.

---

## 12. Recommended configurations

| Goal | Config | Accuracy | Cost |
|---|---|---:|---|
| Max efficiency, pure LGN | aggressive | 27.2 % | ~30× fewer FLOPs |
| Best pure-LGN quality | combo (hybrid-L0 + token-shift K=2) | 36.5 % | ~7.5× fewer FLOPs |
| Best quality/efficiency balance | sel_edges (keep L0, L11) | 37.0 % | a few × fewer FLOPs |
| Highest quality | sel_4edges (keep 4 layers) | 39.0 % | modest savings |

## 13. Future work (the only routes left)
1. **Stateful / recurrent LGN gates** (RDDLGN-style) — the literature-backed route to
   genuine sequence mixing with logic gates; the only way to a truly pure-LGN replacement.
2. **Fixed, honest, wider cross-token mixers** — token-shift works; the open question is a
   non-trained mixer reaching further than K=2 without becoming fake-LGN (dilated did not help).
3. **Accept the hybrid** for deployment — selective LGN (keep L0,L11) gives 71 % of
   transformer accuracy at a few× FLOP savings.

---

## 14. Code map & reproduction

| Path | Purpose |
|---|---|
| `lgn.py` | LGN classes + all techniques + hard-snap mirrors + efficiency helpers |
| `pipeline.py` | imitation, fine-tune, cumulative scaling, joint polish, CAGE schedule |
| `run.py` | CLI (`heatmap` / `scale`) |
| `efficiency.py` | params / FLOPs(per-token) / gates / memory + GPU benchmark |
| `experiments/run_report_experiments.py` | main 18-config scaling batch |
| `experiments/run_screen*.py` | two-phase screening + identity / CAGE / dilated ablations |
| `experiments/run_cage_scaling.py`, `run_depth_conv_experiments.py`, `run_phase_E_H.py`, `run_reverse_greedy.py` | targeted scaling batches |
| `experiments/plot_report_final.py` | figure generation |
| `tests/test_lgn_integrity.py` | 6 integrity tests (all pass) |

```bash
# baseline heatmap (per-layer difficulty)
python run.py heatmap --learn_pool --checkpoint results/baseline.pt --results_dir results/aggressive
# main scaling batch
python experiments/run_report_experiments.py
# fast screening of a new idea
python experiments/run_screen.py
```

## 15. Sources
- Petersen et al., *Deep Differentiable Logic Gate Networks*, NeurIPS 2022 — arXiv 2210.08277
- Petersen et al., *Convolutional Differentiable Logic Gate Networks*, NeurIPS 2024 — arXiv 2411.04732
- *Recurrent Deep Differentiable Logic Gate Networks* (RDDLGN), 2025 — arXiv 2508.06097
- *Mind the Gap* (Gumbel-STE), 2025 — arXiv 2506.07500
- *Light Differentiable Logic Gate Networks* (IWP), 2025 — arXiv 2510.03250
- *Align Forward, Adapt Backward* (CAGE), 2026 — arXiv 2603.14157
- *LILogicNet: Compact Logic Gate Networks with Learnable Connectivity*, 2025 — arXiv 2511.12340
- *Half the Nonlinearity is Wasted* (2026); *The Discrete Charm of the MLP* (2026) — MLP boundary-layer structure

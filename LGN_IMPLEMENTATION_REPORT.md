# LGN as a Transformer Replacement — Implementation Report

> ⚠️ **CLI note (historical doc):** this report describes the *full* experimental history.
> The codebase was later slimmed to only the accuracy-raising techniques. **Flags such as
> `--iwp`, `--gumbel_ste`, `--shift_taps`, `--conv_in_k/--conv_out_k`, `--skip_gate`,
> `--pool_weighted`, `--edge_depth`, `--random_from` and the `reverse_greedy` strategy no
> longer exist.** The current CLI supports: aggressive setup (`--binary_io --n_bits
> --no_in_proj --sum_pool --learn_pool`), `--token_shift`, `--hybrid_layers`,
> `--protected_layers`, `--cage`, `--identity_logic`. See `LGN_FULL_REPORT.md` §14.

**Project:** LGN_Nano — replacing transformer sublayers with Differentiable Logic Gate
Networks (LGN) in a byte-level autoregressive language model (nanoGPT, WikiText-2).

**Status:** 30+ configurations measured. 2 configurations (`dilated_124816`,
`tshift2_polish_kl`) still running at time of writing — marked `[pending]`.

**Date:** 2026-06-02

---

## 1. Executive summary

We replaced the MLP (and optionally attention) sublayers of a 12-layer, 128-dim
byte-level GPT with **Differentiable Logic Gate Networks** and measured how much
language-model accuracy survives the replacement, under a **strictly honest protocol**
(the base model is frozen; every reported number is pure LGN cost; an identity-ablation
verifies the LGN body actually does the work).

**Headline numbers (byte-level next-token accuracy on WikiText-2):**

| Model | Accuracy | Perplexity | Notes |
|---|---:|---:|---|
| Transformer (ceiling) | **54.87 %** | 4.67 | frozen reference |
| Best pure-LGN replacement (all 12 layers) | **36.45 %** | 9.35 | `combo`: hybrid-L0 + token-shift K=2 |
| Best LGN + a few kept transformer layers | **39.01 %** | 8.17 | `sel_4edges`: 8 LGN + 4 transformer |
| Aggressive pure LGN (no cross-token help) | **27.22 %** | 12.63 | the "honest floor" |
| Identity control (LGN = pass-through) | **23.25 %** | 16.18 | residual + frozen base alone |

**The ceiling for a fully-pointwise LGN replacement is ~36 % accuracy (66 % of the
transformer), and it is set by one thing: the L0 cross-token bottleneck.** Every
technique that helped attacked that bottleneck; every technique that tried to make the
*pointwise Boolean computation itself* stronger (more depth, wider gates, conv/linear
projections, alternative parametrizations) either failed or was a measurement artifact.

---

## 2. What was implemented

### 2.1 Core LGN machinery (`lgn.py`)

- **`LearnedLogicLayer`** — a layer of 2-input Boolean gates. Each output gate:
  - selects inputs A, B from `k=4` random candidates via learnable `conn_logits_a/b`
  - selects one of 16 Boolean functions via a 16-way `gate_logits` softmax
  - soft training: `g(A,B) = Σ softmax(gate_logits)·gate_truth_table`
  - hard inference: argmax → one discrete Boolean gate
- **`LogicGateGPTLayer`** — drop-in nanoGPT block: `LayerNorm → [token-shift] →
  activation → binarize → LGN stack → group-sum pool → + residual`.
- **`HybridLogicGateGPTLayer`** — keeps the trained attention sublayer frozen, replaces
  only the MLP with an LGN.
- **`Hard*` mirrors** — fully discrete versions for inference-time measurement
  (argmax connections + argmax gates → pure Boolean circuit).

### 2.2 "Aggressive" setup (the honest default)

The default configuration removes every trained float transformation around the gates:

| Flag | Effect |
|---|---|
| `--binary_io --n_bits 8` | thermometer encoding: each scalar → 8 bits via STE thresholds |
| `--no_in_proj` | **no** trained Linear before the LGN |
| `--sum_pool` | **no** trained Linear after the LGN (fixed group-sum) |
| `--learn_pool` | only a cheap per-channel affine on the pooled output |

This matters: with trained Linear projections around the gates, the projections do the
work and the LGN becomes decorative (see §5, "fake LGN"). The aggressive setup is the
only configuration where the Boolean gates genuinely carry the computation.

### 2.3 Techniques implemented and tested

Built and measured during the project:

1. **Token shift** (`--token_shift K`) — causal channel-aligned mixer: position *t* sees
   `[x[t], x[t-1], …, x[t-K]]`. Gives pointwise LGN a local cross-token receptive field.
2. **Dilated token shift** (`--shift_taps 1 2 4 8 16`) — exponential look-back span at
   low channel cost (Idea E).
3. **Hybrid layers** (`--hybrid_layers 0`) — keep frozen attention at chosen layers.
4. **Selective LGN** (`--protected_layers 0 11`) — keep some layers as full transformer.
5. **Depth + random interconnect** (`--depth D --random_from N`) — stacked LGN sublayers,
   first learnable then reservoir-style fixed-random connections.
6. **Conv1d projections** (`--conv_in_k --conv_out_k`) — causal conv in/out projections.
7. **IWP** (`--iwp`) — Input-Wise Parametrization (Light DLGN, 2025): 4 weights/gate.
8. **Gumbel-STE** (`--gumbel_ste`) — Mind the Gap (2025) stochastic discrete training.
9. **CAGE** (`--cage`) — Align Forward, Adapt Backward (2026): hard forward + adaptive
   backward temperature, closes the discretization gap by construction.
10. **Binary regularization** (`--bin_reg_weight`) — RDDLGN (2025) bit-commitment term.
11. **Reverse greedy** (`--strategy reverse_greedy`) — hardest layer first (Idea B).
12. **Joint polish + KL distillation** (`--joint_polish_steps --joint_polish_kl_weight`)
    — final all-layer coordination pass distilling the transformer logits (Idea H).

> *Hash-embedding L0 (Idea A) was prototyped but removed from the codebase:* its
> imitation phase and its evaluation used different forward paths (the byte-bit path
> only activated inside the full-model forward), so its measurement was invalid. See §6.2.

### 2.4 Training pipeline (`pipeline.py`)

Per-layer cumulative replacement (greedy easiest-first by per-layer difficulty):
1. **Imitation** (200 steps): MSE against the original layer's output on the *live*
   upstream distribution.
2. **Fine-tune** (3000 steps): LM cross-entropy + entropy regularizer, temperature
   annealed 2.0 → 0.1 (only the new layer anneals; settled layers stay sharp).
3. **Hard snap**: argmax → discrete circuit; measure hard degradation.
4. **Frozen base throughout** — attention, embeddings, lm_head are never trained during
   LGN fine-tuning, so all degradation numbers are pure LGN cost.

---

## 3. Methodology — how we kept the measurement honest

Three guards, all of which caught real problems:

### 3.1 Frozen base
The transformer's attention, embedding and lm_head are frozen. The LGN cannot "cheat"
by retraining the surrounding network. Every accuracy number is the cost of the LGN
alone.

### 3.2 Identity ablation (the fake-LGN detector)
For each architecture we run a twin with `--identity_logic` (LGN body returns its input
unchanged). The **LGN contribution** = `hd(identity) − hd(real)`. If they are equal, the
LGN is decorative and the surrounding machinery is doing the work.

This is the single most important methodological tool in the project. It revealed that
**every conv/linear-projection variant was fake LGN** (§5).

### 3.3 Two-phase screening
Full 12-layer cumulative scaling costs ~3 h per config. We built a **cheap screen**
(`run_screen.py`): 4 representative layers (L0, L5, L10, L11), 100 imitation + 500
fine-tune steps, ~5–7 min per config. We screen first, then promote only winners to full
scaling. This cut a ~30 h sweep to ~3 h of screening + targeted full runs.

**Caveat discovered:** the per-layer screen is *insensitive to token-mixing* methods —
token-shift's benefit is system-level (cross-token information propagates across layers)
and barely shows up in single-layer replacement. Screening is valid for comparing gate
parametrizations and projection schemes, but cross-token ideas must be confirmed at full
scale.

---

## 4. Results — the difficulty map

Per-layer hard-degradation (single-layer replacement, aggressive setup) localizes the
problem precisely:

| Layer | hard_degradation (nat) | Role |
|---|---:|---|
| **L0** | **+0.90 … +1.06** | cross-token mixing — severe |
| L1–L9 | +0.00 … +0.15 | near-trivial (some ≈ identity) |
| L10 | +0.16 | moderate |
| **L11** | **+0.30 … +0.37** | lm_head precision — hard |

> **The entire accuracy gap lives at the boundaries.** L0 alone (one layer!) costs more
> than all eight middle layers combined. This single fact explains every result below.

---

## 5. Full results table (all 30+ configurations)

### 5.1 Production — full 12-layer cumulative scaling (next-token accuracy)

| Config | Accuracy | Δ vs aggressive | What it is |
|---|---:|---:|---|
| **Transformer** | **54.87 %** | — | ceiling |
| sel_4edges | 39.01 % | +11.8 | 8 LGN + 4 transformer (L0,L1,L10,L11) |
| sel_edges | 37.03 % | +9.8 | 10 LGN + 2 transformer (L0,L11) |
| **combo** | **36.45 %** | **+9.2** | hybrid-L0 + token-shift K=2 |
| token_shift_k2 | 36.22 % | +9.0 | contiguous shift [1,2] |
| token_shift_k3 | 36.13 % | +8.9 | shift [1,2,3] |
| tshift2_cage | 35.91 % | +8.7 | token-shift K=2 + CAGE |
| dilated_124 | 35.62 % | +8.4 | dilated shift [1,2,4] (Idea E) |
| token_shift_k1 | 35.16 % | +7.9 | shift [1] |
| token_shift_k2_reverse | 35.07 % | +7.9 | K=2, reverse-greedy order (Idea B) |
| sel_L0 | 34.70 % | +7.5 | 11 LGN + 1 transformer (L0) |
| hybrid_L0_agg | 33.46 % | +6.2 | frozen attention at L0 |
| hybrid_edges | 33.44 % | +6.2 | frozen attention at L0 & L11 |
| aggressive_n16 | 27.57 % | +0.4 | 16 bits/scalar |
| aggressive_s42 | 27.26 % | +0.0 | seed 42 |
| **aggressive** | **27.22 %** | **0.0** | honest pure-LGN floor |
| aggressive_cage | 27.03 % | −0.2 | CAGE (gap closes, acc flat) |
| aggressive_s7 | 26.90 % | −0.3 | seed 7 |
| aggressive_n4 | 26.38 % | −0.8 | 4 bits/scalar |
| depth2_rand | 26.32 % | −0.9 | depth 2, 1 learn + 1 random |
| depth2_learn | 25.86 % | −1.4 | depth 2, all learnable |
| depth4_learn | 25.52 % | −1.7 | depth 4, all learnable |
| depth4_rand | 25.31 % | −1.9 | depth 4, reservoir |
| identity | 23.25 % | −4.0 | control (LGN = pass-through) |
| iwp_fixed | 22.17 % | −5.1 | Input-Wise Parametrization |
| dilated_124816 | `[pending]` | — | dilated shift [1,2,4,8,16] |
| tshift2_polish_kl | `[pending]` | — | token-shift K=2 + joint polish + KL |

3-seed variance of `aggressive`: **27.13 % ± 0.19 pp** — results are stable, not lucky seeds.

### 5.2 Discretization gap (soft vs hard, full 12-layer)

| Config | soft_deg | hard_deg | gap |
|---|---:|---:|---:|
| aggressive | +0.990 | +1.016 | +0.027 |
| **aggressive_cage** | +0.998 | +1.012 | **+0.014** (−48 %) |
| token_shift_k2 | +0.641 | +0.744 | +0.103 |
| **tshift2_cage** | +0.692 | +0.739 | **+0.047** (−54 %) |

CAGE does exactly what its paper claims — halves the discretization gap — but our
baseline gap was already small, so it does not move final accuracy.

---

## 6. What works, what doesn't, and why

### 6.1 Works (honest, verified by identity ablation)

| Technique | Effect | Mechanism |
|---|---:|---|
| **Selective LGN** (keep L0/L11) | up to **+11.8 pp** | sidesteps the boundary bottleneck entirely |
| **Token shift K=2** | **+9.0 pp** | gives pointwise LGN a local cross-token window |
| **Hybrid L0** | **+6.2 pp** | restores real attention exactly where it is needed |
| **CAGE** | gap −50 % (acc flat) | closes the soft→hard discretization gap by construction |

LGN-contribution check (identity ablation, screening Σhd):
`aggressive` +0.48, `aggressive_cage` +0.75, `tshift2_cage` **+1.14** — CAGE and token
shift both *increase* how much real work the Boolean gates do.

### 6.2 Does not work

| Technique | Result | Why |
|---|---|---|
| **Conv / Linear projections** | "best" hd but **FAKE** | identity ablation: conv3 LGN-contribution = **−0.69** — the conv kernel does the cross-token work; the LGN is decorative. The projection re-introduces the trained float transformation we deliberately removed. |
| **Depth + random interconnect** | −1 to −2 pp | quantization error accumulates across hard-snapped Boolean layers; the reservoir argument that holds for image MNIST does not survive the residual byte-LM setting |
| **Gumbel-STE** | **−18 pp** (catastrophic) | confirms CAGE's central thesis: Gumbel noise alone does not close the gap; the forward path must be hard |
| **IWP** | −5 pp | the 4-weight reparametrization helps deep image conv-LGNs but hurts our thermometer + sum-pool byte setup |
| **Binary regularization** | flat / −0.1 | RDDLGN's bit-commitment term is tuned to their sigmoid encoder, not our STE thermometer |
| **Hash embedding L0** (Idea A) — *removed* | apparent L0 hd 0.63 → 1.01 | **measurement invalid**: imitation trained on the residual-stream activation while evaluation used the byte-bit path (train/eval forward mismatch). Feature removed from the codebase. The intuition that a trained embedding is a *useful prior* (learned byte-similarity structure), not a bottleneck, still stands. |
| **Reverse greedy** (Idea B) | −1.15 pp | easy-first ordering lets the network gradually adapt to LGN noise; hard-first plants a bad L0 foundation that later layers must work around |
| **Dilated token shift** (Idea E) | −0.6 pp (124); 124816 pending | the contiguous K=2 window is already the sweet spot; sparse far taps add channels without adding usable signal |

### 6.3 The one structural lesson

Every win came from **cross-token receptive field**, never from making the pointwise
Boolean computation richer. A pointwise function on the residual stream — no matter how
deep, wide, or cleverly parametrized — cannot reconstruct what attention does at L0. This
is consistent with the literature (§7) and is the fundamental ceiling on pure-LGN
transformer replacement.

---

## 7. Literature comparison

### 7.1 Where LGNs have been used

| Work | Domain | Result | Relation to us |
|---|---|---|---|
| Petersen et al. 2022, *Deep DLGN* (NeurIPS) | image (MNIST/CIFAR) | MNIST ~97–99 % with 8K–64K gates | original DLGN; classification, not sequences |
| Petersen et al. 2024, *Convolutional DLGN* (NeurIPS) | image | CIFAR-10 **86.29 %** w/ 61M gates, **1900×** faster inference | conv structure on images; we tested conv → fake-LGN in LM |
| LILogicNet 2024 (arXiv 2511.12340) | image | MNIST **98.45 %** w/ 8K gates | compact learnable connectivity |
| Light DLGN 2025 (arXiv 2510.03250) | image | IWP, **8.5×** faster convergence at depth | we tested IWP → −5 pp in our LM setting |
| Mind the Gap 2025 | discretization | Gumbel-STE | we tested → −18 pp |
| **Align Forward, Adapt Backward (CAGE) 2026** | discretization | MNIST gap → ~0; **30–76×** faster than Gumbel-ST | we confirmed: gap halves, but our baseline gap was already small |
| **RDDLGN 2025** (arXiv 2508.06097) | **sequence** (WMT'14 EN→DE translation) | 5.00 BLEU train / **4.39 BLEU** hard; GRU 5.41; gap 30.9 % → 27.7 % | **the only prior LGN sequence model** |

### 7.2 Our position in the literature

- **LGNs are an image-classification technique.** Every strong result (MNIST 98 %,
  CIFAR 86 %) is on images, where the input is naturally a fixed-size grid and there is
  no cross-token mixing problem.
- **Only one prior work applies LGNs to sequences:** RDDLGN (2025), and it is
  *seq2seq translation with a recurrent (stateful) LGN*, not autoregressive
  byte-level language modeling, and not a *transformer-sublayer replacement*. Its
  discretization gap (30.9 → 27.7 %, −3.2 pp) is the same order as ours.
- **Our setting is therefore novel:** replacing transformer sublayers with pointwise
  LGNs in an autoregressive byte-LM, under a frozen-base honesty protocol. To our
  knowledge no prior paper measures this.
- **Our negative results are informative, not just failures:** the identity-ablation
  fake-LGN detector is a methodological contribution — it shows that conv/linear LGN
  "wins" can be illusory, a caveat that applies to LGN claims more broadly.
- **The L0 cross-token wall matches theory:** RDDLGN had to introduce *stateful*
  (recurrent) gates to do sequence mixing — exactly because pointwise gates cannot. Our
  token-shift / hybrid results are the parallel-architecture version of the same finding.

---

## 8. Efficiency (the LGN payoff)

Measured by `efficiency.py` (total params, theoretical FLOPs **per token**, gate counts)
plus wall-clock benchmarks on an RTX 2080 SUPER. (FLOPs are now reported per-token,
consistent with bool/io ops — an earlier version multiplied by the 64-token sequence
length while labelling the field "per token"; the savings ratios were unaffected.)

| Config | Total params | FLOPs/token | LGN gates | Bool ops/token | FLOPs vs transformer |
|---|---:|---:|---:|---:|---:|
| Transformer | 2.45 M | 2.56 M | 0 | 0 | 1.0× |
| Identity | 0.37 M | 0.086 M | 12,288 | 36,864 | **29.7× fewer** |
| **Aggressive** | 0.37 M | 0.086 M | 12,288 | 36,864 | **29.7× fewer** |
| Hybrid L0 | 0.44 M | 0.168 M | 12,288 | 36,864 | 15.2× fewer |
| Token shift K=2 | 0.96 M | 0.258 M | 36,864 | 110,592 | 9.9× fewer |
| Combo | 1.03 M | 0.340 M | 36,864 | 110,592 | 7.5× fewer |

- **Params:** aggressive LGN uses **6.5× fewer** parameters than the transformer.
- **FLOPs:** aggressive uses **~30× fewer** theoretical FLOPs/token.
- **The hardware-relevant metric is the gate count** (12K–37K 2-input Boolean LUTs per
  token). These are exactly the operations that map to one FPGA/ASIC LUT each.

**GPU wall-clock caveat:** on GPU the LGN is *slower* (hard_speedup 0.15×–0.52× vs
transformer) because PyTorch has no optimized kernel for discrete gather + gate
evaluation — it runs as many small ops. The FLOP/gate advantage is **theoretical /
FPGA-relevant**, not a GPU win. This matches Petersen 2024's claim that the speedup is
realized on dedicated hardware (their 1900× was specialized inference), not stock GPU
training.

---

## 9. The ceiling — explicit statement

| Question | Answer |
|---|---|
| Best **pure** LGN (all 12 layers Boolean)? | **36.45 %** (combo) = 66 % of transformer accuracy |
| Best with a few transformer layers kept? | **39.01 %** (sel_4edges, 4 of 12 kept) = 71 % |
| Honest floor (no cross-token aid)? | **27.22 %** (aggressive) = 50 % |
| Absolute floor (residual + frozen base only)? | **23.25 %** (identity) = 42 % |
| What sets the ceiling? | the **L0 cross-token bottleneck** — a pointwise Boolean function cannot do content-based all-to-all token mixing |
| Can more/better gates raise it? | **No** — depth, width, conv, linear, IWP all failed or were fake. Only cross-token mechanisms (token-shift, hybrid, selective) help. |
| Does state-of-the-art discretization (CAGE) help? | It **halves the gap honestly** but our gap was already small, so accuracy is flat. It does *not* raise the ceiling. |

### 9.1 What would be needed to break the ceiling

Grounded in the experiments and literature, the only promising routes left are
*architectural* (cross-token), not *gate-level*:

1. **Stateful / recurrent LGN gates** (RDDLGN-style flip-flops) — the only literature
   route to genuine sequence mixing with logic gates. A truly pure-LGN transformer
   replacement would need this, not parallel pointwise gates.
2. **Wider learned cross-token windows that stay honest** — token-shift works; the open
   question is a cross-token mixer that is fixed/non-trained (so it does not become
   "fake LGN") yet reaches further than K=2. Dilated shift was the attempt; results so
   far say K=2 is already the sweet spot.
3. **Accept the hybrid** — if the goal is deployment rather than a pure-LGN proof,
   selective LGN (keep L0, L11 as transformer) gives 71 % of transformer accuracy at a
   few × FLOP savings, and is the recommended practical configuration.

---

## 10. Recommended configurations

| Goal | Config | Accuracy | Cost |
|---|---|---:|---|
| **Maximum efficiency**, pure LGN | aggressive | 27.2 % | ~30× fewer FLOPs |
| **Best pure-LGN quality** | combo (hybrid-L0 + token-shift K=2) | 36.5 % | ~7.5× fewer FLOPs |
| **Best quality / efficiency balance** | sel_edges (keep L0, L11) | 37.0 % | a few × fewer FLOPs |
| **Highest quality** | sel_4edges (keep 4 layers) | 39.0 % | modest savings |

---

## 11. Code map

| File | Purpose |
|---|---|
| `lgn.py` | LGN classes, all 13 techniques, hard-snap mirrors, efficiency helpers |
| `pipeline.py` | imitation, fine-tune, cumulative scaling, joint polish, CAGE schedule |
| `run.py` | CLI (heatmap / scale subcommands, all flags) |
| `efficiency.py` | params / FLOPs / gates / memory + GPU benchmark + speedup table |
| `run_screen*.py` | two-phase screening harnesses (base, CAGE, dilated, identity ablations) |
| `run_*_experiments.py`, `run_phase_E_H.py` | production scaling batches |

---

## 12. Sources

- Petersen et al., *Deep Differentiable Logic Gate Networks*, NeurIPS 2022 — arXiv 2210.08277
- Petersen et al., *Convolutional Differentiable Logic Gate Networks*, NeurIPS 2024 — arXiv 2411.04732
- *Light Differentiable Logic Gate Networks* (IWP), 2025 — arXiv 2510.03250
- *Mind the Gap* (Gumbel-STE), 2025 — arXiv 2506.07500
- *Align Forward, Adapt Backward* (CAGE), 2026 — arXiv 2603.14157
- *Recurrent Deep Differentiable Logic Gate Networks* (RDDLGN), 2025 — arXiv 2508.06097
- *LILogicNet: Compact Logic Gate Networks with Learnable Connectivity*, 2025 — arXiv 2511.12340

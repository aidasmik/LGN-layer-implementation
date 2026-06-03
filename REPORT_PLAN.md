# LGN ataskaitos planas

> ⚠️ **CLI pastaba (istorinis dokumentas):** kodas vėliau sutrumpintas iki tik accuracy
> keliančių technikų. Flag'ai `--pool_weighted`, `--iwp`, `--gumbel_ste`, `--shift_taps`,
> `--conv_in_k/--conv_out_k`, `--skip_gate`, `--edge_depth`, `--random_from` ir
> `reverse_greedy` strategija **nebeegzistuoja**. Dabartinis CLI: aggressive
> (`--binary_io --n_bits --no_in_proj --sum_pool --learn_pool`), `--token_shift`,
> `--hybrid_layers`, `--protected_layers`, `--cage`, `--identity_logic`. Žr. `LGN_FULL_REPORT.md` §14.

> Pilnas dokumentas, kuris atspindi LGN implementaciją, treniravimo pipeline'ą, eksperimentų rezultatus ir literatūros sąsajas. Šis failas — planas su skyrių struktūra, pagrindiniais skaičiais ir trūkstamų testų sąrašu.

---

## Struktūra (8 skyriai)

### 1. Įvadas ir tikslas
- **Tikslas:** ištirti, ar nanoGPT MLP sluoksniai gali būti pakeisti differentiable Logic Gate Networks (LGN) ir kiek tai kainuoja kokybės atžvilgiu.
- **Modelis:** 12 sluoksnių × 128d × 4h transformer, byte-level WikiText-2.
- **Pagrindiniai klausimai:**
  - Kokius sluoksnius galim pakeisti pigiai?
  - Kodėl L0 yra struktūriškai problemiškas?
  - Koks LGN įnašas yra realus (vs tinklo kompensacija)?

### 2. LGN implementacija ir architektūra
**Vienas loginis vartas:**
- 2 įvestys A, B → 1 išvestis
- 16-vartų Bool polinomas: `g(A,B) = c₀ + c₁·A + c₂·B + c₃·A·B`
- Koeficientai paimti iš `LOGIC_GATE_MATRIX` (16 eilučių × 4 koef.)
- **Mokomi parametrai per išvesties poziciją:**
  - `conn_logits_a, conn_logits_b` (shape `out_dim × k`) — kuri iš k=4 kandidato įvestys naudoti A ir B
  - `gate_logits` (shape `out_dim × 16`) — kuris iš 16 Bool vartų

**LGN blokas (`LogicGateGPTLayer`):**
```
x → LayerNorm → [token_shift] → [in_proj] → activation → [binary_io]
   → LearnedLogicLayer(s) [depth times]
   → [out_proj | sum_pool] → +residual
```
- **Aggressive default** (`binary_io + no_in_proj + sum_pool`): LGN yra **vienintelis** mokomas komponentas bloke. Linear sluoksniai pašalinti.
- **Soft → hard snap** per temperatūros annealing'ą (2.0 → 0.1).

**CLI vėliavos ir jų paskirtis** (santrauka, pilna lentelė atskira):
- `--binary_io / --n_bits` — įvesties binarizacija (thermometer encoding)
- `--no_in_proj` — pašalina `in_proj` Linear
- `--sum_pool / --learn_pool / --pool_weighted` — išvesties agregacija
- `--token_shift K` — fixed causal cross-token mixer
- `--hybrid_layers` — kuriuose sluoksniuose laikyti originalų attention
- `--iwp` — Input-Wise Parametrization (Light DLGN, alternatyvi vartų išraiška)
- `--gumbel_ste` — Gumbel-noise STE treniravimas

### 3. Treniravimo pipeline
**Trys etapai per sluoksnį:**
1. **Imitation** (1k žingsnių, MSE į originalų MLP išėjimą)
2. **Fine-tune** (1–3k žingsnių, LM loss, temperatūros annealing)
3. **Hard snap** (argmax → diskretus Boolean grandinė) + val loss/ppl/acc

**Cumulative scaling:** greedy ordering pagal heatmap difficulty; pakeičiama po vieną; tik **naujai pridedamas** sluoksnis annealinamas, jau apsisprendę lieka sharp.

**Užšaldyta bazė:** transformer attention sluoksniai, embeddings, lm_head VISADA užšaldyti per fine-tune. Mokomi tik LGN sluoksnių parametrai → matuojam gryną LGN kainą (ne tinklo adaptaciją).

### 4. Per-layer heatmap — kiek sunku pakeisti kiekvieną sluoksnį

**Eksperimentas:** kiekvienas iš 12 sluoksnių keičiamas ATSKIRAI (kiti 11 lieka transformer). Matuojam `hard_degradation`.

**Pagrindiniai radiniai:**
- **L0, L11 yra gerokai sunkesni** už vidurio sluoksnius
- **L1–L9 hard_d ≈ 0** (kai kurie net neigiami — LGN šiek tiek geriau už transformer'į)
- Aggressive setup'e L0 = +1.06 nat single-layer, L11 = +0.37

**Grafikas:** `01_heatmap_per_layer.png` (jau yra)

### 5. Cumulative scaling — pakeičiant po vieną

**Eksperimentas:** sluoksniai keičiami sekvenciškai greedy tvarka (lengviausi pirma); kiekvieno žingsnio modelis treniruojamas; final hard val loss matuojamas.

**Pagrindiniai skaičiai (jau turim, pilna fixed-code versija ateina):**

| n pakeista | Aggressive | Identity (kontrolė) |
|---:|---:|---:|
| 0 | 0.000 | 0.000 |
| 4 | +0.143 | +0.287 |
| 8 | +0.515 | +0.809 |
| 12 | **+1.033** | **+1.413** |

**Žalias plotas tarp linijų = realus LGN įnašas: +0.38 nat per visus 12 sluoksnių.**

**Grafikas:** `02_cumulative_scaling.png`

### 6. Kodėl L0 nepakeičiamas (struktūrinė priežastis)

**Mūsų radiniai:**
- L0 single-layer aggressive: +1.06 nat (kitų sluoksnių dydis ~0.01–0.37)
- Hybrid L0 (atstato attention) sumažina iki +0.41 nat su 12 LGN sluoksnių vs aggressive +1.03

**Kodėl pointwise LGN negali pakeisti L0:**
1. **L0 mato neapdorotą embedding'ą** — informacija tarp tokenų dar nepasidalinta
2. **Pointwise LGN procesuoja kiekvieną tokeną IZOLIUOTAI** — negali atlikti cross-token mixing'o
3. **Attention's pirmojo sluoksnio darbas** yra kontekstualizuoti embedding'us prieš tolimesnį apdorojimą — LGN to fundamentaliai negali

**Literatūros patvirtinimas:**
- [Half the Nonlinearity is Wasted (2026)](https://arxiv.org/abs/2603.03459) tyrimas 162M–2.8B parametrų modeliuose: **kraštiniai (pirmas, paskutinis) sluoksniai sunaudoja didžiąją dalį nonlinearity budget'o**, viduriniai near-linear.
- Kompresijos literatūra: "compressing L0 MLP up-projection alone increases perplexity by **20,000×**" — sutampa su mūsų L0 jautrumu.
- [Discrete Charm of the MLP (2026)](https://arxiv.org/abs/2603.10985): MLP atlieka binary routing of continuous signals; ankstesni sluoksniai (L1–L3) naudoja "gateway neurons" route'inimui, vėlesni — "consensus architecture". L0 ypatinga rolė.

**Grafikas:** `03_L0_difficulty.png` (lyginant L0 vs kiti sluoksniai)

### 7. Kuriuos sluoksnius galim pakeisti ir kiek laimim

**Utilizacijos žemėlapis (identity ablation):**

| Sluoksnis | LGN įnašas (nat) | Statusas |
|---|---:|---|
| L0 | +0.240 | aktyvus |
| L1–L6 | +0.009 — +0.014 | inertiškas |
| L7 | +0.020 | aktyvus (ribinis) |
| L8–L9 | +0.018 | inertiškas |
| L10 | +0.024 | aktyvus |
| L11 | +0.025 | aktyvus |

**Iš 12 sluoksnių LGN dirba realiai tik 4** (L0, L7, L10, L11). Vidurio sluoksniai (L1–L6, L8–L9) — LGN nieko reikšmingo nedaro (gali būti pakeisti identity be praradimo).

**Selektyvaus LGN rezultatai:**

| Setup | LGN sluoks. | Transf. sluoks. | accuracy |
|---|---:|---:|---:|
| Transformer (ceiling) | 0 | 12 | **54.9%** |
| Aggressive (visi LGN) | 12 | 0 | 27.2% |
| sel_L0 (transf L0) | 11 | 1 | 34.9% |
| sel_edges (transf L0+L11) | 10 | 2 | 36.4% |
| sel_4edges (transf L0,L1,L10,L11) | 8 | 4 | **38.9%** |
| Hybrid L0 (FIXED) | 12 | 0 (attention freeze) | **43.3%** |
| Token shift K=2 (pure LGN, fixed) | 12 | 0 | 35.2% |

**Pagrindinis pasiūlymas:**
- Replacing only MIDDLE layers (L1–L9, except L7) → daugiausia nauda mažiausiam kokybės nuostoliui.
- Keeping L0 + L11 untouched (or hybridized) → didžiausias accuracy išlaikymas.

**Grafikas:** `04_selective_curve.png` (acc vs LGN sluoksnių skaičius)

### 8. Pure-LGN ribos ir kelios kryptys
**Pamatuotos ribos (mūsų budget'e):**
- Visi parametrų sweep'ai aggressive setup'e: saturuoja prie ~27% acc
- Gumbel-STE: mažina soft-hard gap'ą, bet absoliuti accuracy krenta
- IWP (Light DLGN): per saturuotą init nepasiteisino; net su fix'u mūsų budget'e neperviršija baseline'o
- Token shift K=2: **vienintelis pataisymas, kuris reikšmingai padidina pure LGN accuracy** (27% → 35%)

**Tolesnės kryptys:**
1. **Stateful gates** (Recurrent DDLGN) — sprendžia cross-token problemą be attention'o
2. **Selektyvus LGN** — keičiam tik vidurinius, gauname efektyvumo-kokybės kompromisą
3. **Combined hybrid + cross-token** (mūsų eksperimentas dabar bėga)

**Grafikas:** `05_final_comparison.png` + `06_method_evolution.png`

---

## Trūkstami testai (laukiam fixed batch)

Šie runs dabar bėga su pataisytu kodu (channel-aligned token shift, true identity STE, hybrid attn truly frozen, imitation live distribution):

| Run | Statusas | Naudos raportui |
|---|---|---|
| `fix_aggressive` | ⏳ | Švarus aggressive baseline (visi 4 fix'ai įjungti) |
| `fix_tshift_k2` | ⏳ | Channel-aligned token shift — geresnis nei senas? |
| `fix_hybrid_L0` | ⏳ | Hybrid su fix'ais (attn truly frozen + eval mode) |
| `fix_combo` | ⏳ | Hybrid L0 + Token shift K=2 su visais fix'ais |

**Jei rezultatai pasikeis reikšmingai**, atnaujinsim ataskaitos skaičius.

## Papildomi testai (rekomenduoju paleisti po fixed batch)

| Testas | Apimtis | Vertė ataskaitai |
|---|---|---|
| **Per-layer heatmap su fixed code** | ~25 min | Pagrindinis "per sluoksnis sunkumas" grafikas naujam kodui |
| **Selektyvūs (sel_L0/edges/4edges) — bet protected sluoksniai TRENUOJAMI** | ~1 val | Sąžiningas selektyvus matavimas (dabartiniame protected sluoksniai užšaldyti) |
| **Token shift K=1 ir K=4** (be hybrid) | ~1 val | Optimalus K nustatymas — ar daugiau shift padeda |

---

## Reikalingi grafikai

| # | Grafikas | Duomenys | Statusas |
|---|---|---|---|
| 1 | Per-layer heatmap (12 sluoksnių aggressive) | `results/aggressive/heatmap.json` | ✅ turim |
| 2 | Cumulative scaling: aggressive vs identity | `cmp_learnpool` vs `aggressive_identity` scaling | ✅ turim |
| 3 | L0 difficulty across configs | hybrid_L0_fixed vs aggressive | ✅ turim |
| 4 | Selective LGN curve (acc vs n_transformer_layers) | sel_L0, sel_edges, sel_4edges | ✅ turim |
| 5 | Final method comparison bar chart | visi runs | 🔄 papildomas su fix_* |
| 6 | Method evolution timeline (chronological) | visi runs | ✅ galim padaryti |
| 7 | Utilization per layer (LGN contribution) | aggressive vs aggressive_identity per layer | ✅ turim |
| 8 | Architecture diagram (LGN block) | manual draw / ASCII | ⏳ reikia padaryti |

---

## Pagrindiniai skaičiai ataskaitai (santrauka)

```
                                       acc      ppl     loss
Transformer (ceiling)               54.9%    4.67    1.54
─────────────────────────────────────────────────────────
Pure LGN:
  Aggressive (visi 12 LGN, baseline) 27.2%   12.66    2.54
  + Token shift K=2                  35.2%    9.77    2.28  (+8.0)
─────────────────────────────────────────────────────────
Selektyvus (LGN + transformer layers):
  sel_L0   (1 transf, 11 LGN)        34.9%    9.57    2.26
  sel_edges (2 transf, 10 LGN)       36.4%    9.01    2.20
  sel_4edges (4 transf, 8 LGN)       38.9%    8.24    2.11
─────────────────────────────────────────────────────────
Hybrid:
  Hybrid L0 (attn frozen + LGN MLP)  43.3%    7.04    1.95
─────────────────────────────────────────────────────────
LGN įnašas (identity ablation):
  Aggressive vs Identity-LGN gap @ n=12:  +0.38 nat (real LGN contribution)
  Tikrai aktyvūs sluoksniai (>0.02 nat):   L0, L7, L10, L11 (4 iš 12)
```

---

## Statusas ir kiti žingsniai

✅ **Bugs reviewed:** 10 issues found, all fixed (8 from user review + 2 dependent fixes)
✅ **Code: syntax OK, all fixes verified**
🔄 **Fixed batch running:** 4 cumulative scaling runs (~3 hours)
⏳ **Po fixed batch:**
1. Sugeneruoti visus 8 grafikus su atnaujintais skaičiais
2. Suvesti report'ą į `LGN_FINAL_REPORT.md`
3. Įvertinti, ar verta paleisti papildomus testus (per-layer heatmap fixed, K sweep)

Po šių žingsnių ataskaita bus pilna ir publikuotina.

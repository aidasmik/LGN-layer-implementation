# LGN-Nano: Logic Gate Networks transformer'io sluoksniuose

Tikrinu, kiek nanoGPT transformer'io MLP sluoksnių galima pakeisti į Boolean **Learned Logic Gate Networks (LGN)** sluoksnius, ir ar iš to lieka realaus loginio darbo, ar tik aplinkinių Linear sluoksnių kompensacija.

**Modelis:** nanoGPT, 12 sluoksnių × 128d × 4 head'ai, byte-level WikiText-2.
**LGN blokas:** 16-vartų Bool polinomas `c₀ + c₁·A + c₂·B + c₃·A·B` su soft → hard snap ir temperatūros annealing'u.

> ⚠️ **CLI pastaba:** §1–§13 aprašo VISĄ eksperimentų istoriją (įsk. technikas, kurios nepadėjo).
> Kodas vėliau sutrumpintas iki tik accuracy keliančių technikų — tekste minimi
> `pool_weighted`, `iwp`, `gumbel_ste`, `shift_taps`, `conv_*`, `skip_gate`, `edge_depth`,
> `random_from` (random interconnect) ir `reverse_greedy` **nebeegzistuoja kode**.
> Dabartinis CLI: aggressive setup, `--depth` (numatytas 1; >1 vis dar veikia, bet nepadeda),
> `--token_shift`, `--hybrid_layers`, `--protected_layers`, `--cage`, `--identity_logic`.
> Pilnas einamasis aprašas: `LGN_FULL_REPORT.md`.

---

Pradėjau nuo iš anksto pasiūlytų krypčių: aktyvacijos funkcijos, dropout, init scale, per-layer annealing, gilesni kraštiniai sluoksniai. Čia daug nesiplėsiu, nes po identity ablation testo šie rezultatai tapo nelabai aktualūs.

## 1.1 Aktyvacijos funkcija

Pirmiausia dariau parametrų sweep. Išbandžiau tris aktyvacijos funkcijas, siekdamas pagerinti pirmą ir paskutinį sluoksnius: sigmoid (baseline), tanh ir relu. Geriausiai vis tiek liko sigmoid, tanh ir relu ypač blogai veikė kraštiniuose sluoksniuose.

![activations](results/figs/report/01_activations.png)

Tas pats matosi ir sharpness grafike. Sigmoid po fine-tune pasiekia apie 0.96 max-softmax, o tanh ir relu veikia prasčiau.

![sharpness](results/figs/report/02_sharpness.png)

## 1.2 Dropout testai

Dropout pridėjimas tik pablogina rezultatus.

![dropout 0.05](results/figs/report/03_dropout_005.png)
![dropout 0.1](results/figs/report/04_dropout_01.png)

## 1.3 Per-layer annealing

Nepadeda. Sunkesniems sluoksniams (L0, L11) skyriau daugiau imitation žingsnių, bet rezultatas nebuvo geresnis.

![per-layer anneal](results/figs/report/05_per_anneal.png)

## 1.4 Gilesni pradinis ir galinis sluoksnis

L0 ir L11 padidinau iki depth=3, width_mult=4. Nepadėjo, daugiau gatu greičiausiai lėmė kad snap mode klaidos kaupiasi.

![edge d3w4](results/figs/report/06_edge_d3w4.png)

---

## 2. Ablation test

Išbandžiau jūsų siūlytą ablation testą. Pakeičiau LGN vidų į identity  jis tiesiog praleidžia signalą ir neatlieka jokių logikos veiksmų. Jei LGN darytų kažką naudingo, būtų matyti stiprus pablogėjimas, dabar jo nebuvo… Vadinasi LGN iš esmės nieko nedaro.

Manau, jog taip atsitinka dėl linear sluoksniai supančių patį LGN. Prieš ir po LGN yra pilni mokomi Linear sluoksniai, jie galėjo patys išmokti tai, ką turi daryti LGN. Imitation dalyje linear sluoksniai tiesiog perėmė MLP elgesį, o LGN tarp jų tapo beveik kiaurai pereina. Fine-tune metu tai dar labiau sustiprėjo, nes gradientams greičiausiai lengviau eiti per linear sluoksnius nei per LGN su jos temperatūros annealingu.

![ablation pairs](results/figs/report/07_ablation_pairs.png)

---

## 3. LGN panaudojimas

Kadangi, kaip suprantu, projekto tikslas yra išsiaiškinti LGN, o ne Linear sluoksnių pritaikymą transformeriuose, išbandžiau, kaip juos galima priversti veikti. Bandžiau pašalinti linear sluoksnius.

Šalinau po vieną:

- **binary_io** — LGN įvestis binary per Straight-Through Estimator threshold (0/1). Dabar LGN gauna Bool signalą, ne continuous.
- **sum_pool** — out_proj Linear pakeistas fiksuotu group-sum (padalina width į grupes po 16, susumuoja).
- **no_in_proj** — ir in_proj Linear išmestas.

Galutinė konfigūracija (aggressive setup) – LGN, kuriame tik LGN dalis yra mokoma; viskas kita (binarizacija, sum_pool) yra fiksuota. Kuo mažiau Linear pagalbos, tuo modelis labiau blogėjo, bet tuo aiškiau pradėjo matytis tikras LGN. Kai aggressive setup LGN pakeičiama į identity, skirtumas jau aiškiai matomas. Vidutinis LGN įnašas — apie +0.034 nat per sluoksniui, o per visus 12 sluoksnių kaupiamai apie +0.38 nat. Pirmą kartą galim sakyti, kad pati LGN dalis kažką realiai atlieka.

![aggressive progression](results/figs/report/08_aggressive_progression.png)
![scaling](results/figs/report/11_scaling_main.png)

Aggressive setup LGN nedirba visuose sluoksniuose vienodai. Kai kiekvieną sluoksnį pakeitę į LGN palyginu su tuo pačiu sluoksniu pakeistu į identity, skirtumas yra didelis tik keliuose, daugiausia L0 ir paskutiniuose sluoksniuose (L10, L11). L0 vienas duoda didžiąją dalį viso LGN įnašo. Vidurio sluoksniuose (L1–L9) LGN įnašas labai nedidelis.

![utilization](results/figs/report/09_utilization_per_layer.png)

## 3.2 Bandymai pagerinti aggressive setup

Bandžiau pakeisti:

- **K=32** — dvigubai connection capacity. Nieko reikšmingo neduoda.
- **ft2k** — dvigubai fine-tune žingsnių. Šiek tiek pagerina.
- **skipgate** — learnable scalar, kuris padaugina LGN įnašą prieš residual. Nedidelis pagerėjimas.
- **per-layer anneal** — sunkesniems sluoksniams daugiau imitation žingsnių. Nepadėjo, šiek tiek blogina.

Visi pavieniai pataisymai duoda vos kelių procentų naudą. Bandžiau dar stack'inti du geriausius : skipgate + ft2k. Rezultatai pablogėji.

![improvements](results/figs/report/12_improvements.png)
![combo break](results/figs/report/14_combo_break.png)

Dabar kai žinome, jog bent pats LGN kažką daro, bandysiu dar jį optimizuoti. Tradicianiai pakeitimai daug nedavė, tačiau planuoju išbanyti pridėti hibridinius sluoksnius. Jie greičiausiai būtų pritaikyti kraštiniams sluoksniams. Ten paliekamas originalus transformer'io attention, paimtas iš baseline'o (jis jau apmokytas, pre-baked) keičiama tik MLP dalis į LGN.

---

## 4. Pilnas modelis vs transformer (accuracy)

Iki šiol matavom degradaciją tik per loss (nats). Įdomu pamatyti, ką tai reiškia praktiškai — kiek next-byte spėjimų teisingi (accuracy).

Trijų modelių palyginimas, visi 12 MLP sluoksnių pakeisti, bazė užšaldyta:

![three-way comparison](results/figs/report/17_three_way_metrics.png)

| Modelis | loss | perplexity | accuracy |
|---|---:|---:|---:|
| Transformer (originalas) | 1.54 | 4.67 | **54.9%** |
| Identity-classic (Linear MLP, jokios logikos) | 2.48 | 11.98 | 28.2% |
| Aggressive LGN (tikra logika, be Linear) | 2.54 | 12.66 | 27.2% |

Įdomus radinys: **identity-classic** (Linear sluoksniai be loginių vartų) yra **beveik toks pat blogas** kaip aggressive LGN. Pakeitus visus 12 MLP'ų užšaldytoje bazėje, kokybė nukrenta perpus **nepriklausomai nuo metodo**. Tad gap iki transformer'io daugiausia nėra "logika silpna" — tai bendras kainos paliepimas keisti visus MLP'us. Pozityvus kampas: aggressive LGN pasiekia tą pačią accuracy kaip Linear sandwich **be jokio float matmul** bloke.

## 5. Gryno LGN ribos (I/O bottleneck atakos)

Užšaldytos bazės kontekste bandžiau pagerinti patį LGN per I/O kanalus:

- **n_bits=16** — turtingesnė įvestis (finesnė embedding binarizacija)
- **depth=2** — daugiau loginės talpos
- **pool_weighted** — turtingesnė išvestis (mokami per-bit svoriai vietoj uniform sum)

| Konfiguracija | accuracy | vs baseline |
|---|---:|---:|
| Aggressive + learn_pool (baseline) | 27.2% | — |
| + n_bits=16 (input) | 27.6% | +0.4% |
| + depth=2 (capacity) | 25.5% | **−1.7%** (blogiau) |
| + pool_weighted (output) | 27.4% | +0.2% |

Visi pataisymai **per noise floor** (±0.5 ppt). Daugiau loginių vartų net pablogina (snap klaidos kaupiasi).

**Išvada:** grynas LGN saturavęs ties ~27% — struktūrinė riba šitam setup'ui (pointwise LGN, užšaldyta bazė, visi 12 sluoksnių).

## 6. Hybrid L0 — architektūrinis apėjimas

Paliekant **attention L0 sluoksnyje** (jis apdoroja raw embedding'ą, kur cross-token mixing kritinis), accuracy šokteli dramatiškai:

![final comparison](results/figs/report/18_final_comparison.png)

| Modelis | accuracy |
|---|---:|
| Aggressive LGN (visi 12) | 27.2% |
| **Hybrid L0 + aggressive LGN** | **44.4%** |
| Transformer (ceiling) | 54.9% |

**+17 punktų vien iš vieno architektūrinio sprendimo.** Tai parodo, kad gap iki transformer'io koncentruotas **L0** sluoksnyje, kur LGN fundamentaliai negali atlikti reikiamo darbo (pointwise vartai negali maišyti tokenų — tai attention darbas).

Hybrid uždaro **daugiau nei pusę** likusio gap'o (27→44, kelias iki 55). Tai ne pure LGN — bet aiškiai rodo, kur LGN "lubos" ir kur architektūrinė nuolaida turi prasmę.

> **Pastaba (atnaujinta):** šis 44.4% yra iš ANKSTYVOJO classic-setup matavimo. Su
> **pataisytu kodu** (griežtas honest-protokolas, žr. §8–§13) tikslesni skaičiai:
> hybrid L0 + aggressive = **33.5%**, o geriausias pure-LGN (combo: hybrid-L0 +
> token_shift K=2) = **36.5%**. Bendras pasakojimas tas pats (cross-token L0 sprendimas
> uždaro gap'ą), tik tikslūs procentai žemesni dėl švaresnio matavimo.

## 7. Literatūros įžvalgos (2025 darbai)

Naujausi DLGN tyrimai sutinka su mūsų pastebėjimais ir siūlo konkrečius sprendimus:

- **["Mind the Gap"](https://arxiv.org/abs/2506.07500) (NeurIPS 2025)** — siūlo **Gumbel noise + STE** soft-hard gap'ui. Jų image rezultatai geri, BET **mes ištestavom mūsų byte-LM setup'e — žlugo** (blogiausias rezultatas, žr. 7.1). Įpėdinis CAGE (2026) parodė kodėl: svarbu hard forward, ne Gumbel triukšmas.

- **["Light DLGN"](https://arxiv.org/abs/2510.03250) (2025)** — vartų **reparametrizacija (IWP)**. 4× mažiau parametrų, greitesnis training. **Ištestuota → −5 pp** mūsų setup'e (tinka image conv-LGN, ne byte-LM).

- **["Recurrent DDLGN"](https://arxiv.org/abs/2508.06097) (2025)** — *būtent* mūsų cross-token apribojimo sprendimas. Į loginį tinklą įdedami **stateful vartai (flip-flops, latches)**, kurie leidžia logikai dirbti su sekomis. WMT'14 vertimas: 5.0 BLEU (vs GRU 5.4), su **20,000× mažiau loginių operacijų**. Tai realus kelias atsisakyti attention'o **pačiame LGN** lygyje.

### 7.1 Ką iš šių krypčių JAU ištestavom (atnaujinta)

Aukščiau buvusi rekomendacija buvo „pradėti nuo Gumbel-STE". **Jį ištestavom — ir jis pasirodė blogiausias.** Realūs rezultatai:

| Kryptis | Statusas | Rezultatas |
|---|---|---|
| **Gumbel-STE** (Mind the Gap) | ✅ ištestuota | **Blogiausia iš visų** — screening Σhd **2.096** vs aggressive 1.348 (+0.75 blogiau), L0 hd 0.90 → **1.47**. Mūsų byte-LM setup'e Gumbel noise gap'o NEUŽDARO. |
| **CAGE** (Align Forward, Adapt Backward, 2026) | ✅ ištestuota | Mind the Gap *įpėdinis*. **Uždaro discretization gap'ą perpus, sąžiningai** (aggressive 0.027→0.014; tshift 0.103→0.047) — bet accuracy nepasikeičia (mūsų gap'as jau buvo mažas). Identity-ablation patvirtino: CAGE *padidina* tikrą LGN įnašą (+0.48 → +0.75 … +1.14 nat). |
| **IWP** (Light DLGN) | ✅ ištestuota | **−5 pp** (22.17%) — tinka gilioms image conv-LGN, kenkia mūsų thermometer + sum-pool byte setup'ui. |
| **Stateful / recurrent gates** (RDDLGN) | ❌ neištestuota | Vienintelė likusi tikrai perspektyvi kryptis grynam cross-token LGN. Reikalauja didelio architektūrinio pakeitimo. |

**Kodėl Gumbel-STE žlugo, o CAGE ne** (CAGE 2026 centrinė tezė, kurią mūsų duomenys patvirtino): gap'ą uždaro ne Gumbel triukšmas, o **hard forward kelias** (forward = argmax = inference). Gumbel-STE turi soft forward → gap lieka. CAGE: hard forward + adaptyvi backward temperatūra.

### 7.2 Atnaujinta rekomendacija

- **NEdaryti Gumbel-STE** — empiriškai paneigta mūsų task'ui.
- **CAGE** jau įgyvendinta (`--cage`) ir veikia kaip žadėta (gap −50%), bet accuracy nekelia, nes mūsų baseline gap'as jau mažas. Naudinga kaip *honest-training* garantija, ne kaip accuracy boost'as.
- **Vienintelis kelias pakelti lubas** — ne gate-lygmens triukai (depth, conv, IWP, Gumbel — visi žlugo arba fake), o **cross-token mechanizmai**: jau veikia token_shift (+9 pp) ir hybrid L0 (+6 pp); ilgalaikė kryptis — **stateful RDDLGN-stiliaus vartai** (dar neištestuota).

> Pilnas rezultatų rinkinys ir metodologija: `LGN_IMPLEMENTATION_REPORT.md`.

---

# Antras etapas — pilnas eksperimentų rinkinys (fixed-code)

> §1–§7 yra ankstyvojo etapo (classic-setup) žurnalas. Žemiau — pilnas, suderintas
> 30+ konfigūracijų rinkinys su **pataisytu kodu** ir griežtu honest-protokolu
> (užšaldyta bazė, identity-ablation kiekvienam). Pilna versija: `LGN_IMPLEMENTATION_REPORT.md`.

## 8. Cross-token: token shift (pagrindinis proveržis)

Pointwise LGN negali maišyti tokenų — visas gap'as koncentruotas L0 (hd ~+0.9…1.06).
Sprendimas: **token_shift K** — kiekviena pozicija mato `[x[t], x[t-1], …, x[t-K]]`
(channel-aligned, fiksuotas, jokių mokomų mixing parametrų → lieka honest LGN).

| Config | Accuracy | vs aggressive |
|---|---:|---:|
| aggressive (be cross-token) | 27.22% | — |
| token_shift K=1 | 35.16% | +7.9 pp |
| **token_shift K=2** | **36.22%** | **+9.0 pp** |
| token_shift K=3 | 36.13% | +8.9 pp |

**K=2 — sweet spot.** Bandėm ir **dilated taps** (platesnis span tuo pačiu kanalų kiekiu):
`[1,2,4]` davė 35.62% (truputį blogiau nei K=2), platesni span'ai nepadeda — contiguous K=2 jau optimalus.

## 9. Selective LGN — efektyvumo/kokybės kreivė

Paliekam kelis sluoksnius transformer'iu (boundary sluoksniai brangūs):

| Transformer sluoksniai palikti | LGN sluoksnių | Accuracy |
|---|---:|---:|
| 0 (visi LGN) | 12 | 27.22% |
| 1 (L0) | 11 | 34.70% |
| 2 (L0, L11) | 10 | 37.03% |
| **4 (L0,L1,L10,L11)** | 8 | **39.01%** |
| 12 (visi transformer) | 0 | 54.87% |

Pirmas paliktas sluoksnis (L0) duoda +7.5 pp; kiekvienas kitas mažiau. Sweet spot ~2–4.

## 10. Dvifazis screening + fake-LGN atradimas

Pilnas 12-sluoksnių scaling = ~3 h/config. Sukūrėm **pigų screening'ą**
(`experiments/run_screen.py`): 4 reprezentatyvūs sluoksniai (L0,L5,L10,L11), 500 ft
žingsnių, ~5 min/config. Screen'inam pirma, pilną scaling tik laimėtojams.

**Svarbiausias metodologinis įrankis — identity ablation** (`--identity_logic`): LGN
vidus tampa pass-through. `LGN_įnašas = hd(identity) − hd(real)`. Jei lygu — LGN
dekoratyvus, o supantys sluoksniai daro darbą.

Tai **demaskavo visus conv/linear variantus kaip FAKE LGN:**

| Architektūra | LGN įnašas | Verdiktas |
|---|---:|---|
| aggressive | **+0.48** | ✅ REAL |
| token_shift K=2 + CAGE | **+1.14** | ✅ REAL |
| conv3 (Conv1d in/out proj) | **−0.69** | ❌ FAKE — conv kernel daro cross-token darbą, LGN tik triukšmas |
| linear_proj | −0.24 | ❌ FAKE |

`conv3` su LGN turėjo L0 hd 0.45, o `conv3` su identity LGN — **0.014**. T.y. conv
išsprendžia L0 PATS; LGN tik kenkia. Conv pakartoja tą mokomą float transformaciją,
kurią sąmoningai pašalinom aggressive setup'e.

## 11. CAGE — Align Forward, Adapt Backward (2026)

Hard forward (argmax = inference) + adaptyvi backward temperatūra pagal commitment.
**Uždaro discretization gap'ą perpus, sąžiningai:**

| Config | soft→hard gap | su CAGE |
|---|---:|---:|
| aggressive | +0.027 | **+0.014** (−48%) |
| token_shift K=2 | +0.103 | **+0.047** (−54%) |

Bet **accuracy nepasikeičia** (aggressive_cage 27.03% ≈ aggressive 27.22%) — mūsų
baseline gap'as jau buvo mažas, tad nėra ko „gelbėti". Naudinga kaip honest-training
garantija, ne accuracy boost'as. (Skirtingai nei image-LGN, kur gap'ai dideli ir CAGE duoda +20pp.)

## 12. Ką dar bandėm ir NEpadėjo

| Kryptis | Rezultatas | Kodėl |
|---|---|---|
| Depth + random interconnect | 25.3–26.3% (−1…−2 pp) | hard-snap klaidos kaupiasi per gylį |
| Conv/Linear projekcijos | "geriausias" hd, bet **FAKE** | projekcija daro darbą (žr. §10) |
| Gumbel-STE (Mind the Gap) | **blogiausias** (Σhd 2.10) | reikia hard forward, ne Gumbel triukšmo |
| IWP (Light DLGN) | −5 pp (22.17%) | tinka image conv-LGN, ne byte-LM |
| Binary regularization (RDDLGN) | flat/−0.1 | netinka mūsų thermometer encoder'iui |
| Reverse greedy (sunkiausi pirma) | −1.15 pp | greedy easy-first leidžia tinklui palaipsniui prisitaikyti |
| Hash embedding L0 | pašalinta (buggy) | train/eval forward mismatch — matavimas nevalidus |

## 13. Galutinės lubos

| | Accuracy | % nuo transformer |
|---|---:|---:|
| Transformer (ceiling) | 54.87% | 100% |
| Geriausias su keliais transformer sluoksniais (sel_4edges) | **39.01%** | 71% |
| Geriausias PURE LGN (combo: hybrid-L0 + token_shift K=2) | **36.45%** | 66% |
| Honest floor (aggressive, jokio cross-token) | 27.22% | 50% |
| Identity control (LGN = pass-through) | 23.25% | 42% |

**Vienas dėsnis:** kiekvienas laimėjimas — iš **cross-token receptive field**
(token_shift, hybrid, selective). *Niekas* iš gate-lygmens triukų (depth, conv, linear,
IWP, Gumbel) nepakėlė lubų. Pointwise Boolean funkcija ant residual stream'o
fundamentaliai negali atkurti to, ką attention daro L0 — tai ir yra pure-LGN riba.
Prasmingas kelias toliau: **stateful / recurrent LGN vartai** (RDDLGN, dar neištestuota).

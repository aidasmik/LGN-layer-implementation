# LGN-Nano: Logic Gate Networks transformer'io sluoksniuose

Tikrinu, kiek nanoGPT transformer'io MLP sluoksnių galima pakeisti į Boolean **Learned Logic Gate Networks (LGN)** sluoksnius, ir ar iš to lieka realaus loginio darbo, ar tik aplinkinių Linear sluoksnių kompensacija.

**Modelis:** nanoGPT, 12 sluoksnių × 128d × 4 head'ai, byte-level WikiText-2.
**LGN blokas:** 16-vartų Bool polinomas `c₀ + c₁·A + c₂·B + c₃·A·B` su soft → hard snap ir temperatūros annealing'u.

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

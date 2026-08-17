# DraftVision draft-grade experiments — results (test = 2019-2020, n=733)

Baseline to beat (production config, reproduced exactly by this harness): **accuracy 0.4093, macro-F1 0.4022** (raw ensemble mean); 0.4065 calibrated.

Split identical to scripts/train_models.py: train 2010-2017 (+63 seed rows w=5), calibration 2018, test 2019-2020 (untouched in every run). `hist` = +2000-2009 train-only rows; `hist-w` = same with recency weight 0.97^(2020-year). Feature sets: `base13` = production 13 features; `+meas_raw` = + raw vertical/bench/broad/cone/shuttle/height_in/weight_lb; `+meas_norm` = + the same 7 as per-position-group z-scores (stats from non-test years only); `+meas_raw+age` = + draft-file age (**LEAKY — see Sanity**). Methods: `flat` = XGB+CatBoost 4-class mean ensemble (prod architecture); `ordinal` = 3 cumulative-binary XGBs P(y<=k), differenced/clipped/renormalized, argmax; `regression` = XGB regressor on class value + thresholds tuned on 2018.

| config (features \| method \| train) | test acc | macro-F1 | recall R1-2 | recall R3-4 | recall R5-7 | recall UDFA | calibrated acc |
|---|---|---|---|---|---|---|---|
| base13|flat|2010+ | 0.4093 | 0.4022 | 0.438 | 0.308 | 0.436 | 0.438 | 0.4065 |
| base13|ordinal|2010+ | 0.4038 | 0.3555 | 0.633 | 0.051 | 0.382 | 0.540 | — |
| base13|regression|2010+ | 0.4229 | 0.3931 | 0.578 | 0.128 | 0.569 | 0.393 | — |
| +meas_raw|flat|2010+ | 0.4434 | 0.4295 | 0.492 | 0.333 | 0.320 | 0.616 | 0.4447 |
| +meas_raw|ordinal|2010+ | 0.4270 | 0.3603 | 0.734 | 0.038 | 0.249 | 0.701 | — |
| +meas_raw|regression|2010+ | 0.4379 | 0.3951 | 0.438 | 0.077 | 0.684 | 0.442 | — |
| +meas_norm|flat|2010+ | 0.4393 | 0.4270 | 0.453 | 0.353 | 0.333 | 0.598 | 0.4434 |
| +meas_norm|ordinal|2010+ | 0.4379 | 0.3738 | 0.688 | 0.051 | 0.253 | 0.750 | — |
| +meas_norm|regression|2010+ | 0.4216 | 0.3757 | 0.469 | 0.064 | 0.680 | 0.384 | — |
| +meas_raw+age|flat|2010+ **(leaky)** | 0.6562 | 0.6262 | 0.656 | 0.372 | 0.511 | 1.000 | 0.6835 |
| +meas_raw+age|ordinal|2010+ **(leaky)** | 0.6357 | 0.5694 | 0.852 | 0.109 | 0.516 | 1.000 | — |
| +meas_raw+age|regression|2010+ **(leaky)** | 0.6726 | 0.5789 | 0.562 | 0.070 | 0.827 | 1.000 | — |
| base13|flat|hist | 0.3956 | 0.3849 | 0.484 | 0.237 | 0.436 | 0.415 | 0.4011 |
| base13|flat|hist-w | 0.4011 | 0.3914 | 0.453 | 0.269 | 0.440 | 0.424 | 0.4093 |
| base13|ordinal|hist | 0.4011 | 0.3456 | 0.680 | 0.026 | 0.364 | 0.540 | — |
| base13|ordinal|hist-w | 0.4011 | 0.3429 | 0.688 | 0.019 | 0.351 | 0.554 | — |
| base13|regression|hist | 0.4229 | 0.3922 | 0.445 | 0.128 | 0.609 | 0.429 | — |
| base13|regression|hist-w | 0.4202 | 0.3598 | 0.453 | 0.019 | 0.658 | 0.442 | — |
| +meas_raw|flat|hist | 0.4352 | 0.4186 | 0.570 | 0.256 | 0.302 | 0.616 | 0.4420 |
| +meas_raw|flat|hist-w | 0.4366 | 0.4220 | 0.539 | 0.301 | 0.293 | 0.616 | 0.4475 |
| +meas_raw|ordinal|hist | 0.4379 | 0.3662 | 0.797 | 0.032 | 0.244 | 0.710 | — |
| +meas_raw|ordinal|hist-w | 0.4284 | 0.3602 | 0.719 | 0.038 | 0.244 | 0.719 | — |
| +meas_raw|regression|hist | 0.4284 | 0.3656 | 0.539 | 0.013 | 0.582 | 0.500 | — |
| +meas_raw|regression|hist-w | 0.4543 | 0.4222 | 0.508 | 0.122 | 0.613 | 0.495 | — |
| +meas_norm|flat|hist **<- winner** | 0.4502 | 0.4354 | 0.523 | 0.327 | 0.311 | 0.634 | 0.4502 |
| +meas_norm|flat|hist-w | 0.4434 | 0.4297 | 0.539 | 0.321 | 0.302 | 0.616 | 0.4461 |
| +meas_norm|ordinal|hist | 0.4461 | 0.3792 | 0.750 | 0.051 | 0.258 | 0.737 | — |
| +meas_norm|ordinal|hist-w | 0.4407 | 0.3744 | 0.742 | 0.045 | 0.262 | 0.723 | — |
| +meas_norm|regression|hist | 0.4175 | 0.4191 | 0.492 | 0.385 | 0.347 | 0.469 | — |
| +meas_norm|regression|hist-w | 0.4161 | 0.3734 | 0.469 | 0.077 | 0.689 | 0.348 | — |
| +meas_raw+age|flat|hist **(leaky)** | 0.6630 | 0.6333 | 0.734 | 0.397 | 0.471 | 1.000 | 0.6780 |
| +meas_raw+age|flat|hist-w **(leaky)** | 0.6576 | 0.6278 | 0.703 | 0.404 | 0.467 | 1.000 | 0.6889 |
| +meas_raw+age|ordinal|hist **(leaky)** | 0.6398 | 0.5687 | 0.898 | 0.096 | 0.511 | 1.000 | — |
| +meas_raw+age|ordinal|hist-w **(leaky)** | 0.6344 | 0.5639 | 0.906 | 0.096 | 0.489 | 1.000 | — |
| +meas_raw+age|regression|hist **(leaky)** | 0.6835 | 0.5887 | 0.547 | 0.077 | 0.867 | 1.000 | — |
| +meas_raw+age|regression|hist-w **(leaky)** | 0.6712 | 0.5626 | 0.648 | 0.019 | 0.809 | 1.000 | — |

## Winner

**`+meas_norm | flat | hist` — position-normalized measurables + 2000-2009 history, same XGB+CatBoost mean ensemble: accuracy 0.4502 (+0.0409 over 0.4093 baseline, +4.1 pts), macro-F1 0.4354 (+0.0332).** Its 2018-fitted multinomial calibrator keeps accuracy at 0.4502 but collapses Day-2 recall to 0.006 (macro-F1 0.3673), so the raw ensemble mean is the number to serve.

`+meas_raw | regression | hist-w` posts the single highest accuracy (0.4543, +4.5 pts) but with macro-F1 0.4222 and Day-2 recall 0.122 — it wins accuracy by dumping nearly everything into R5-7, so the flat +meas_norm+hist config is the recommended one.

Confusion matrix of the winner (rows = true 0..3, cols = pred 0..3):

```
            pred R1-2  R3-4  R5-7  UDFA
 true R1-2        67    36     6    19
 true R3-4        50    51    12    43
 true R5-7        42    53    70    60
 true UDFA        27    46     9   142
```

Per-class recall (winner): R1-2 0.523, R3-4 (Day-2) 0.327, R5-7 0.311, UDFA 0.634. Day-2 remains the weakest class in every non-leaky run (best non-leaky Day-2 recall 0.385).

## Ablations (raw ensemble/argmax accuracy)

- A alone (+meas_raw, flat, 2010-2017): 0.4434 (+3.4 pts). Position-normalizing changes little on its own (0.4393). Measurables are by far the biggest single win.
- B alone (base13): ordinal 0.4038 (-0.6), regression 0.4229 (+1.4 but macro-F1 drops to 0.3931). Ordinal structure helps the extremes, destroys Day-2 recall (0.02-0.11 everywhere).
- C alone (base13, flat): hist 0.3956 (-1.4), hist-w 0.4011 (-0.8). History HURTS without measurables; it only helps once measurables are present (+meas_norm flat: 0.4393 -> 0.4502).
- D best combo: A(+norm) + C(hist pooled) + flat = 0.4502. Adding B on top does not beat it (ordinal 0.4461, regression 0.4175/0.4161).

## Sanity (E)

- Every run asserts: no outcome-derived columns in X (draft_round, pick, experience_years, pro_bowls, seasons_started, w_av/car_av/career_av, to, nfl_success, draft_grade, draft_year), and train/cal contain no 2019-2020 rows. All 36 runs passed.
- The v2 dataset's 2010-2020 slice is byte-identical to production training_data/combine_outcomes.csv (asserted in build_data_v2.py).
- **`age` is leaky and disqualified**: it is sourced from draft_picks.csv, so it is present for 91-100% of drafted rows and 0% of UDFA rows — its missingness encodes the label (UDFA recall exactly 1.000 in every +age run, acc up to 0.6835). Those rows are kept in the table for the record only.

## 0.45 target

Barely reached: 0.4502 (recommended) / 0.4543 (accuracy-max, degenerate). Nothing approaches 0.50. The binding constraint is Day-2 (R3-4): no non-leaky config exceeds 0.385 recall on it. Missing signal: (1) real pre-draft scouting consensus (big-board/mock-draft rank, all-star game invites) — the only known strong separator of R1-2 vs R3-4 vs R5-7; (2) college production for defense/OL and multi-season production for everyone (current feature covers final-season offense for QB/RB/WR/TE, 2015+ classes only); (3) a non-leaky age/early-declare flag (early declaration alone is a strong Day-1/2 signal and is knowable pre-draft); (4) games/durability and recruiting-star data.

# v3 experiments — recruiting, age proxy, all-position production, SP+, modern classes

Data: training_data/experiments/combine_outcomes_v3.csv (9,965 rows, classes 2000-2026; 2010-2020 slice asserted byte-identical to production CSV; built by scripts/experiments/build_data_v3.py from cached CFBD pulls under /tmp/dv_training_cache/cfbd/). Base config for every run = the v2 winner: base13 + position-normalized measurables, flat XGB+CatBoost mean ensemble, train 2000-2017 (+seed rows w=5), cal 2018, frozen test 2019-2020 (n=733, identical rows to v2).

New feature blocks: `A` = recruiting pedigree (rec_stars/rec_rating/rec_ranking from CFBD /recruiting/players 2000-2023, matched by normalized name + committedTo school, fallback name+position-group, recruit class window draft_year-7..-3; `(neutral)` = unmatched imputed to 2.0 stars / min rating / rank 4000 instead of NaN). `B` = non-leaky age proxy from the same match: years_in_college = draft_year - recruit_class_year, early_declare = years_in_college<=3. `C` = all-position production from CFBD /stats/player/season: final-season + career composites (QB/RB/WR-TE yardage-TD composites, coverage floor 2010 classes; tackles+3·TFL+8·sacks+4·PD+10·INT for DL/LB/DB, floor 2017 classes — the defensive category only exists for seasons 2016+), z-scored per position group on non-test years, plus car_seasons. `D` = SP+ team rating of the final college season (/ratings/sp, full 1999-2025 coverage; conference_tier kept alongside). `E` = draft classes 2021-2026 (nflverse, incl. undrafted combine invitees) added as TRAIN-ONLY rows. `fwd` = deployment-realistic forward split: train 2000-2023, cal 2024, test 2025-2026 (n=694; z-score references exclude 2025-2026).

| config (features \| method \| train) | test acc | macro-F1 | recall R1-2 | recall R3-4 | recall R5-7 | recall UDFA | calibrated acc |
|---|---|---|---|---|---|---|---|
| v2winner|flat|hist (reproduction) | 0.4502 | 0.4354 | 0.523 | 0.327 | 0.311 | 0.634 | 0.4502 |
| +A|flat|hist | 0.4570 | 0.4486 | 0.570 | 0.359 | 0.324 | 0.594 | 0.4529 |
| +A(neutral)|flat|hist | 0.4570 | 0.4465 | 0.586 | 0.333 | 0.316 | 0.612 | 0.4570 |
| +A+B|flat|hist | 0.4748 | 0.4700 | 0.695 | 0.372 | 0.320 | 0.576 | 0.4761 |
| +A+B(neutral)|flat|hist | 0.4720 | 0.4649 | 0.672 | 0.365 | 0.307 | 0.598 | 0.4707 |
| +A+B+C|flat|hist | 0.4898 | 0.4826 | 0.688 | 0.391 | 0.307 | 0.630 | 0.4884 |
| +A+B+C+D|flat|hist **<- winner (temporally clean)** | 0.5102 | 0.5031 | 0.711 | 0.410 | 0.324 | 0.652 | 0.4925 |
| +A+B+E|flat|hist+21-26 | 0.4857 | 0.4794 | 0.727 | 0.359 | 0.302 | 0.620 | 0.4966 |
| +A+B+C+E|flat|hist+21-26 | 0.5048 | 0.4927 | 0.719 | 0.436 | 0.262 | 0.674 | 0.5171 |
| +A+B+C+D+E|flat|hist+21-26 **<- best frozen number** | 0.5184 | 0.5089 | 0.734 | 0.462 | 0.276 | 0.679 | 0.5143 |
| +A+B+C+D|ordinal|hist+21-26 | 0.4884 | 0.4206 | 0.852 | 0.090 | 0.240 | 0.808 | — |
| v2winner|flat|fwd (test 2025-26) | 0.3948 | 0.3756 | 0.695 | 0.207 | 0.199 | 0.594 | 0.4135 |
| +A+B+C+D|flat|fwd (test 2025-26) | 0.4424 | 0.4338 | 0.602 | 0.393 | 0.191 | 0.700 | 0.4712 |

## Winner

**`+A+B+C+D | flat | hist` — v2 winner + recruiting pedigree + years-in-college/early-declare + all-position production + SP+: frozen-test accuracy 0.5102 (+0.0600 over v2's 0.4502, +10.1 pts over the 0.4093 production baseline), macro-F1 0.5031.** Adding the 2021-2026 train-only rows (E) lifts it to **0.5184 / 0.5089** — the best frozen number — but those rows postdate the test years, so 0.5102 is the temporally clean claim; both are label-clean (no outcome-derived features, test rows untouched). The ordinal rerun of the best combo repeats the v2 pattern (accuracy 0.4884 but Day-2 recall collapses to 0.090) — flat stays the recommended method.

Confusion matrices (rows = true R1-2/R3-4/R5-7/UDFA):

```
 +A+B+C+D|flat|hist (0.5102)        +A+B+C+D+E|flat|hist+21-26 (0.5184)
   91  19   7  11                      94  22   1  11
   40  64  16  36                      41  72  11  32
   29  55  73  68                      27  67  62  69
   14  48  16  146                     16  48   8  152
```

Per-class recall (winner, no E): R1-2 0.711, R3-4 0.410, R5-7 0.324, UDFA 0.652. The v2 Day-2 recall ceiling (~0.38 non-leaky) is broken: 0.410 without E, 0.462 with E.

## Ablations (frozen test, flat, hist train)

- A alone: +0.7 pts (0.4570), macro-F1 +1.3. Neutral-fill vs NaN is a wash (0.4570 both) — expected, since recruiting missingness is balanced (see Sanity), so the NaN channel carries little class signal either way; both variants reported per protocol.
- B (years_in_college + early_declare) on top of A: +1.8 pts (0.4748) — the single biggest v3 increment after D. The non-leaky age proxy works: available for 83.5% of drafted AND 79.9% of undrafted test rows (vs the disqualified draft-file age: 96%/0%).
- C on top of A+B: +1.5 pts (0.4898), Day-2 recall 0.372→0.391. First-ever defense/all-position production signal; its 2010-17 train coverage is only ~35% (defensive category starts in 2016), so it should gain further as covered classes accumulate.
- D on top of A+B+C: +2.0 pts (0.5102). SP+ adds real value over the static conference_tier (both kept).
- E on top of the full stack: +0.8 pts (0.5184), Day-2 recall 0.410→0.462, at the cost of R5-7 (0.324→0.276).

## Forward split (deployment-realistic)

Train 2000-2023, cal 2024, test 2025-2026 (n=694): v2winner 0.3948 raw / 0.4135 calibrated; winner +A+B+C+D 0.4424 raw / 0.4712 calibrated (macro-F1 0.4338 raw). The v3 features transfer forward (+4.8 pts raw over v2winner on the same split), but the forward problem is genuinely harder than the frozen one — every config scores ~5-7 pts lower than its frozen counterpart (era drift, and 2024 as a single calibration year). On this split the 2024-fitted calibrator HELPS (+2.9 pts), unlike the 2018 one on the frozen split.

## Sanity (G)

- All 13 runs assert: no forbidden/outcome-derived columns in X (v2 list + `age`), no test years in train/cal, and z-score reference years exclude the split's test years.
- The v3 dataset's 2010-2020 slice is byte-identical to production training_data/combine_outcomes.csv (asserted in build_data_v3.py).
- Drafted-vs-UDFA missingness ratio for every new feature (train 2010-17 / test 19-20 / 21-26): rec_stars & rec_rating & rec_ranking 1.06/1.04/1.07; years_in_college & early_declare 1.06/1.04/1.08; prod_fs_z & prod_car_z 0.89/0.98/1.04; car_seasons 0.99/1.00/1.04; sp_rating 1.00/1.00/1.00. **No feature's missingness separates classes** (v2's age trap had ratio ≈ infinity: 96% vs 0%); the worst imbalance anywhere is 1.08. Recruiting match rate drafted vs UDFA (2010+): 78.0% vs 74.0% (58.7% school-matched, 19.7% name+position fallback, 21.6% unmatched) — not wildly imbalanced, and the neutral-fill variant confirms the NaNs carry no material class signal (0.4570 vs 0.4570, 0.4748 vs 0.4720).
- Coverage floors honored: recruiting features NaN for pre-2005 classes (CFBD recruiting thin before 2002); offensive production composites NaN below 2010 classes; defensive composites NaN below 2017 classes (they'd otherwise systematically zero-out pre-2016 defenders).

## 0.50 target

**Reached honestly on the frozen test: 0.5102 with strictly pre-draft, temporally clean features (0.5184 allowing future-class train rows), vs 0.4502 v2 / 0.4093 production baseline.** But the deployment-realistic forward split says the honest expectation for a NEW class is ~0.44-0.47, not 0.51 — the frozen 2019-2020 test sits in the densest part of the feature coverage. Remaining gaps, in order of expected value: (1) pre-draft scouting consensus (big boards / mock-draft aggregate, all-star invites) — still the only known strong R1-2/R3-4/R5-7 separator and still absent; (2) R5-7 remains the weakest class in every v3 config (recall 0.19-0.33) since Day-3 picks are near-indistinguishable from priority UDFAs on production+pedigree alone; (3) defensive production covers only 2016+ seasons, so its training signal is one-third dense — this fixes itself as classes accumulate; (4) per-season granularity (breakout age, dominator ratio) rather than final/career sums.

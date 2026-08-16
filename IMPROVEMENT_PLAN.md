# DraftVision Improvement Plan

Ordered so each phase makes the next one worth doing. Total ~3-5 focused weeks for a solo dev; Phase 1 alone is ~2-3 days and fixes the results users see today.

---

## Phase 1 — Fix now (wrong results or active risk)

### 1.1 Delete the NFL-mascot tier-1 shortcut and rebuild the cache (~2-4 hrs + cache rebuild time)
**What:** In `classify_college_tier` (XGBOost.py:119-127), remove the `NFL_FRANCHISE_KEYWORDS` substring loop (or restrict it to exact full-franchise-name matches like "minnesota vikings" used only when the record is a known NFL player). Also fix the `_T1` substring collisions ("michigan" matching Michigan State, "georgia" matching Georgia Tech/Southern) by switching from `kw in t` to exact/prefix matching on normalized school names. Then re-run `build_prospect_cache.py` and redeploy.
**Why:** 2,484 of 9,033 cached prospects (27.5%) are misclassified as tier 1 because their college mascot matches an NFL mascot (Baylor Bears, Louisville Cardinals, Rhode Island Rams, Boise State Broncos…). Tier is 30% of the draft-grade heuristic (XGBOost.py:1667) and a top model feature — the very first cache entry is a South Alabama CB graded A+/"Top 50 Pick" off this bug. This is the single highest grade-error-per-hour fix in the repo.
**Verified corroboration:** the audit also found the "college" cache literally contains "Minnesota Vikings" as a tier-1 team (fetch_player_data overwrites college team with current NFL team, XGBOost.py:1260-1284) — filter NFL-franchise teams out of the college cache at build time in the same pass.

### 1.2 Stop presenting hash-fabricated stats as real grades (~4-8 hrs)
**What:** For players where `fetch_player_data` falls through to `generate_estimated_profile` (XGBOost.py:1314, 1341, 1357):
- Preserve `data_source` in the cache — `call_predict` in build_prospect_cache.py:129-142 currently discards it, so the leaderboard can't tell real from fabricated.
- In `/api/prospects` responses and the UI, tag these rows "Estimated" (or exclude them from ranked positions and list them unranked below graded players).
- Do NOT display the invented stat lines (a QB gets 8-38 pass TD purely from `sha256(name|position|team|jersey)`, XGBOost.py:846-958) as if they were his season.
**Why:** Verified finding: "John Smith" vs "Jon Smith" (same position/team) get materially different grades from a one-letter name change. Grades that differ because of a hash, not football, on a public leaderboard is the site's biggest credibility risk.

### 1.3 Fix the two guaranteed-wrong code paths (~1-2 hrs)
- **`top_factors` always `[]`:** `top_feature_importances` (XGBOost.py:2065-2081) calls `.get_booster()` on the `CalibratedClassifierCV` loaded in prod; the AttributeError is swallowed. Reach into `success_model.calibrated_classifiers_[0].estimator` (or load `success_xgboost_model.json` just for importances). This also unblocks the "factors" chips the new UI expects.
- **Latent NameError:** `ath_info` at XGBOost.py:1324 is only defined in the `if real_stats:` branch (1259). Add `ath_info = {}` default (or call the cached `_espn_resolve_athlete_info`) so a future refactor doesn't turn `/predict` into a 500.

### 1.4 Neutralize the leakage/bias traps before anyone runs the collector (~2-3 hrs)
**What:** These are dormant only because `training_data/combine_outcomes.csv` doesn't exist yet — but Phase 2 requires running the collector, so defuse them first:
- Delete `prod = 80.0 - dg * 12.0 + gauss(0,8)` (XGBOost.py:1705-1707). Never impute a feature from the label; leave production_score null and handle missingness (see 2.3).
- In `collect_training_data.py:151-162`, remove `draft_round <= 2` from `nfl_success_label` — the code claims draft round is "an output, not an input" (XGBOost.py:1391) but the label uses it.
- In `collect_training_data.py:82-83`, emit missing 40 times as blank, not 50.0, and actually increment the dead `errors` counter in `get()`.
**Why:** The moment you run the documented workflow, the draft-grade model's strongest feature would literally encode its target (Top50→~80, UDFA→~44), silently inflating every metric you'd use to judge Phase 2.

### 1.5 Close the open redirect (~30 min)
**What:** In `enforce_canonical_origin` (XGBOost.py:363-387), only honor `X-Forwarded-Host` when `CANONICAL_HOST` is set (redirect strictly to the canonical host); skip the redirect otherwise. Set `CANONICAL_HOST` in railway.toml/fly.toml.
**Why:** With the current defaults, `X-Forwarded-Host: evil.com` over http yields a 308 to `https://evil.com/<path>` — a phishing-usable open redirect. Trivial fix.

### 1.6 Delete or regenerate `nfl_players.csv` (~30 min)
**What:** The committed file is a header plus 1,000 rows of pure commas — `PLAYER_LOOKUP` is always empty and the `csv_fallback` branch (XGBOost.py:1341-1354) is dead code. Either regenerate the export or delete the file and the fallback path, and log non-empty row count at startup so a failed export can't ship silently again.

---

## Phase 2 — High-leverage accuracy work (model/data)

This is the core problem: **the models currently train on 0 real outcome rows** — 92% synthetic labels sampled from `_success_prob_from_college_profile`/`_draft_grade_from_profile` (the same in-file heuristics, XGBOost.py:1640-1674, admitted in the comment at 1742-43) plus 63 hand-typed seed players duplicated 5x. The "ML ensemble" is a noisy re-encoding of a hand-tuned rule. Everything below is in service of replacing that.

### 2.1 Build a real training set (2-4 days)
**What:** After 1.4's fixes, run `collect_training_data.py` — but restructure it:
- **Kill survivorship bias:** it currently harvests only ESPN draft picks 2015-2022 (fetch_draft_class, lines 108-125), so the "Undrafted" class would be learned entirely from synthetic rows. Add undrafted negatives: combine invitees and all-conference players who went undrafted. Practical shortcut: nflverse's `draft_picks` + `combine` datasets (free CSVs) give you drafted+combine-invited-undrafted in one join, with college, position, and 40 times — far less ESPN scraping.
- **Define success without draft round:** career AV / seasons as primary starter / second contract, from nflverse seasonal data or PFR.
- **Handle censoring:** restrict to classes with ≥5 completed seasons (2015-2020) or add draft-year as a group variable; 2021-22 classes are right-censored.
- **Fetch real college production** per player (collect_training_data.py already has each ESPN id) so production_score is computed from actual stats, not fabricated (1.4).
Commit the CSV so the Dockerfile ships it and training is reproducible.
**Why:** Every model-side improvement is meaningless until labels are real outcomes. If after honest effort the real dataset is too small (<~800 rows), the intellectually honest fallback is to ship the rule-based score labeled as such — which is fine for a student project and more defensible in a portfolio than laundering a heuristic through XGBoost.

### 2.2 Add evaluation before touching the model (1 day)
**What:** There are currently **zero** evaluation calls in the repo (grep-verified: no accuracy/AUC/Brier/log-loss anywhere), and the 5x seed duplication at XGBOost.py:1727 happens *before* the 80/20 split, so identical rows contaminate the Platt calibration set.
- Replace `SEED_TRAINING_PLAYERS * 5` with `sample_weight=5` on unique rows.
- Split by draft-year group (GroupKFold on draft year) — never random rows.
- At train time, print and persist AUC, Brier score, and a reliability curve for both models, **and for the raw heuristic as baseline**. If the model doesn't beat `_success_prob_from_college_profile` on held-out real data, you'll finally know.
- Write `models/metadata.json` (git SHA, data hash, row counts, metrics, seed, timestamp) alongside artifacts, and make missing artifacts fail startup loudly instead of silently retraining inside the gunicorn worker (load_or_train_* at 1918-1933/1974-1989 — with `-w 2`, both workers would train independently).
**Why (order matters):** eval infra must exist before 2.3/2.4 or you can't tell whether they help.

### 2.3 Fix feature saturation and missingness (1-2 days)
**What:**
- **production_score:** 1,388 cached prospects sit at exactly 100.0 because "elite" benchmarks are ordinary-starter per-game rates (RB 100 yds/g, QB 300 yds/g; XGBOost.py:260-297) with a hard clamp. Replace with percentile rank within position (you have 9,033 cached players — compute empirical percentiles per position once and store them), or at minimum double the benchmarks. This removes the "100.0 wall" and the 88-92% success-probability cluster the A+ tier (p≥88 & d==0, XGBOost.py:2500) is keyed off.
- **Remove the 0.92 synthetic-label ceiling** (XGBOost.py:1654) — moot once labels are real, but delete it anyway.
- **Missingness:** pass NaN to XGBoost (it handles it natively) instead of speed=50/tier=5/games=13 defaults, and add `speed_missing`/`tier_missing` indicator features. Replace every falsy `player_stats.get(x) or default` with `is None` checks — a legitimate 0 is currently swallowed (XGBOost.py:1410-1417).
- **OL:** production is hard-capped at `min(games/13,1)*60` (289-291) while grade thresholds are shared — verified result: 0 of 1,556 OL/OT/C players get an A-range grade vs 54% of CB/S, and guards ("G") escape the cap entirely because the position set omits them. Short-term: grade on within-position percentiles (an OL "B" = same percentile as a WR "B"). Long-term: games/starts + honors as the OL production proxy.

### 2.4 Fix the ensemble and the accolade features (1 day)
- **Calibration:** CatBoost is trained with `class_weights={0:1.0,1:spw}` (1829) and never calibrated, then averaged 50/50 with the Platt-calibrated XGBoost (2115, 1967) — the averaging voids the calibration. Calibrate the *averaged* output on the held-out set from 2.2, or calibrate each member.
- **Accolades:** `is_award_winner`/`is_all_american` come from ~37 hardcoded names with bidirectional substring matching (`aw in n or n in aw`, XGBOost.py:224-230) — verified live false positive: Notre Dame's DeVonta Smith flagged as the Alabama Heisman winner, and an empty name matches. Real training rows pin both flags to 0 (1715-1716), so the learned boost applies only to a celebrity list. Either pull awards programmatically (CollegeFootballData.com has free awards endpoints and you're a student — free API key) for both training and inference, or drop both features. Exact normalized-name matching at minimum.

### 2.5 Recenter grade thresholds (~2 hrs, after 2.3)
**What:** `compute_prospect_grade` (2496-2509) gives B- to anyone ≥44% — the cache has 0 D grades and 61% of all FBS players at B- or better. After 2.3 spreads the probability distribution, set thresholds from empirical percentiles (median ≈ C+/B- boundary). Rebuild the cache.

---

## Phase 3 — Product features the new UI expects (unlocked by the above)

### 3.1 Wire the expandable leaderboard rows — zero backend work (frontend: ~1 day)
Verified: every `/api/prospects` record already carries production_score, combine_speed_score, success_probability, grade, draft_grade_class, tier, and accolade flags with filters/sort/pagination (XGBOost.py:2199-2258). Build the mini-bars client-side from these fields; do **not** add N calls to `/predict`. Only games_played is missing if the design wants it — add it to `call_predict`'s dict in build_prospect_cache.py on the next cache rebuild (you're rebuilding anyway in 1.1).

### 3.2 Real "top factors" panel (~half day, unlocked by 1.3)
Once `top_feature_importances` works, the per-player factors display renders. Consider going one step further with per-prediction SHAP values (xgboost has `pred_contribs=True` built in — no new dependency) so factors are player-specific rather than global.

### 3.3 Honest data-provenance UI (~half day, unlocked by 1.2)
Surface `data_source` badges ("ESPN stats" vs "Estimated") and the cache's `generated_at` date ("Board updated Apr 21") in the board header. This converts the fabricated-data liability into a transparency feature.

### 3.4 Cache refresh without redeploy (~half day)
`_PROSPECT_CACHE`/`_HS_PROSPECT_CACHE` are import-time snapshots baked into the Docker image — updating the board requires a full redeploy. Add file-mtime checks on read (works across both gunicorn workers, unlike an admin endpoint that only hits one worker), then you can rebuild the cache on a schedule and just upload the JSON.

### 3.5 Balanced historical comps (~2 hrs)
All 31 entries in `NAMED_HISTORICAL_COMPS` (2522-2562) have `nfl_success=1`, so the comps panel can only ever flatter. Add 10-15 bust comps per position group (you already hand-entered 22 bust archetypes in SEED_TRAINING_PLAYERS — reuse those names/profiles). Also fix the README: comps are weighted Euclidean over 5 fields, not "cosine similarity across the 16-feature vector."

---

## Phase 4 — Later / polish

- **README + UI copy accuracy (~1 hr):** 15 features not 16 (6 base + 9 one-hots, XGBOost.py:70-88); "2-model ensemble" in prod (no TabPFN artifact shipped; the rule-based scorer is a fallback at 2872, never averaged); fix mockup stat cards (v2.dc.html:622-627). For a portfolio project, claims that survive scrutiny matter more than big numbers.
- **Endpoint hygiene (~1 hr):** remove or 410 `/teams`, `/positions`, `/players` (2761/2776/2791) — zero frontend callers, and `/players` is exactly the endpoint behind the commit-6620d97 College Stars regression. Keep `/health` (Railway healthcheck).
- **Stat-mapping cleanup (with 2.1):** stop stuffing WR/TE receiving yards into rushing fields (1066-1072); add receiving_* features; pull games from ESPN gamelog instead of attempts/28 or tackles/6 (1081-1087).
- **Robustness odds and ends (~2 hrs):** try/finally around DB connections in /init//search (2634-2679); `datetime.now(timezone.utc)` instead of the `__import__` utcnow at 2390; pin `anthropic` to an exact version (it's the one billable SDK); defensive text-block selection in the upload parser (2351); seed the `np.random.choice` at 1845.

---

## Suggested sequencing for a solo dev

Week 1: all of Phase 1 (the cache rebuild in 1.1 also picks up 1.2's data_source field and 3.1's games_played — batch them). Week 2: 2.1 + 2.2 (data + eval — the make-or-break work). Week 3: 2.3-2.5 and rebuild/redeploy. Then Phase 3 UI wiring, Phase 4 whenever. The honest decision gate is at the end of 2.2: if the model can't beat the heuristic on real held-out data, ship the heuristic with pride and keep collecting data — that's a better story in an interview than an ensemble that memorized its own rule.
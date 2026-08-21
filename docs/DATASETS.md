# Public Datasets That Could Strengthen DraftVision

*Researched and link-verified 2026-08-19. Every dataset below was fetched live (page, repo, or API response) before being listed; access claims reflect what actually worked today, not what a blog post said in 2021.*

**Context.** DraftVision grades college prospects from production, combine measurables, recruiting stars, SP+ team strength, and years-in-college, trained on nflverse draft/combine/outcome data (2000–2026) + CFBD. Our experiments (see `training_data/experiments/combine_outcomes_v2/v3.csv` and `IMPROVEMENT_PLAN.md` Phase 2) identified two missing signals:

1. **Pre-draft scouting consensus** — the single biggest separator we lack. Combine + production explain far less of draft position than "where does the scouting community rank this guy."
2. **Better per-player athletic/usage data** — full anthropometrics (arm/hand), and usage/efficiency beyond box-score production.

Each entry below maps to those gaps and gets: what it adds, access method, license/ToS risk (blunt), integration effort (S/M/L), and expected model value.

---

## 1. Grinding the Mocks (GTM) — Expected Draft Position

- **What it is:** Benjamin Robinson's project aggregating hundreds of mock drafts per cycle (2018–2026) into an Expected Draft Position (EDP) per prospect, with time-series history. Methodology published as a Bayesian hierarchical model at CMSAC 2020 ([paper PDF](https://www.stat.cmu.edu/cmsac/conference/2020/assets/pdf/Robinson.pdf)).
- **What it adds:** Exactly our #1 gap — a quantified, dated, pre-draft consensus rank, *including history* (rise/fall trajectory), which is stronger than a single final board.
- **Verified today:** The [interactive dashboard](https://grindingthemocks.shinyapps.io/Dashboard/) is live (Shiny app; leaderboards 2018–2026 per its own description) and the [Substack](https://grindingthemocks.substack.com/) is active ([2025 final mock, Apr 2025](https://grindingthemocks.substack.com/p/2025-final-grinding-the-mocks-mock)). **However: I could not verify any public bulk download.** No GitHub repo exists (GitHub search returns nothing for the project), the Substack posts I fetched contain no CSV/Sheet links, and the dashboard is a JS-rendered Shiny app with no visible export.
- **Access method:** Realistically, **email/DM the author** ([@GrindingMocks](https://x.com/GrindingMocks)) for a historical EDP extract — he has done periodic "data releases" per his X feed and collaborated with academics. Scraping a shinyapps.io app is impractical and rude.
- **License/ToS risk:** Unknown until he answers; treat as personal-permission data. Do not redistribute in the repo without explicit OK.
- **Integration effort:** **M** (S once a CSV is in hand — it's a name/year join; the outreach is the long pole).
- **Expected model value:** **Highest of anything on this list** for training years 2018–2021+ *if obtained* — it is literally the consensus signal our experiments flagged, with within-cycle movement as a bonus feature. But it's gated on a human saying yes, which is why the concrete #1 recommendation below pairs free alternatives that exist *today*.

## 1b. Wide Left (Arif Hasan) Consensus Big Board — the free, verified consensus source

- **What it is:** Arif Hasan's long-running Consensus Big Board: top-300 prospects aggregated from ~100–134 analyst boards per cycle. Verified live and **free including the data**: [2026 (134 analysts)](https://wideleft.football/p/2026-nfl-draft-consensus-big-board), [2025 (112)](https://www.wideleft.football/p/2025-consensus-big-board-the-top), [2024 (101)](https://wideleft.football/p/2024-consensus-big-board-the-top). The 2026 post explicitly says the data is downloadable ("Get the data" link + printable Google Sheet) and "this piece is free to everyone, as is the data."
- **What it adds:** Final pre-draft consensus rank for 2024–2026 classes (current-class inference + newest training rows). Earlier editions (2017–2023) ran at The Athletic and are paywalled.
- **Access method:** Manual download of the published sheet per year (3 small files). No API needed.
- **License/ToS risk:** Low for internal feature use; **no explicit reuse license**, so don't republish the board itself — store only the derived `consensus_rank` column, credit the source in the README.
- **Integration effort:** **S.**
- **Expected model value:** High for the live board product (a current-class consensus feature immediately fixes the "our A+ is everyone else's Day 3" failure mode); moderate for training (only 3 draft classes).

## 1c. JackLich10/nfl-draft-data — historical ESPN prospect ranks/grades (the training backfill)

- **What it is:** [github.com/JackLich10/nfl-draft-data](https://github.com/JackLich10/nfl-draft-data) — CSVs of ESPN draft-prospect data back to 1967: `nfl_draft_prospects.csv` (3.4 MB; ranks, grades, measurements, draft slot), `nfl_draft_profiles.csv` (7.2 MB; pre-draft scouting-report text), plus college stats and ID maps. Verified today: files present; last pushed 2021-05-05; **no license file**.
- **What it adds:** A per-player *pre-draft* ESPN rank/grade covering essentially our whole 2000–2021 training window — a single-outlet consensus proxy where GTM/Wide Left don't reach. The scouting text is also a future NLP feature source.
- **Access method:** Raw CSV over HTTPS (static repo).
- **License/ToS risk:** Medium. Unlicensed scrape of ESPN data; fine to *train on* privately, do not vendored-commit or redistribute it. Also frozen at 2021 — it's backfill, not a pipeline.
- **Integration effort:** **S–M** (name/year/position fuzzy join, same machinery as our `rec_match_type` recruiting join).
- **Expected model value:** High — this is what makes a consensus *training feature* (not just an inference-time display) possible across 20+ draft classes.

## 2. MockDraftable — full anthropometrics with a working JSON API

- **What it is:** [mockdraftable.com](https://www.mockdraftable.com/) combine/pro-day measurables with position percentiles. **API verified live today:** `GET https://www.mockdraftable.com/api/player?id=laquon-treadwell` returns JSON with school, draft year, positions, and a `measurements` array that includes arm length (33.375") and hand size (9.5") — fields nflverse's combine data lacks. Search/typeahead endpoints exist too (`/api/search?opts=...&pos=`, `/api/typeahead?search=`), discovered from the site's own MIT-licensed frontend ([marcusdarmstrong/mockdraftable-web](https://github.com/marcusdarmstrong/mockdraftable-web), `src/api/client.js`).
- **What it adds:** Gap #2 — arm length, hand size, wingspan and pro-day-supplemented measurables; plus MockDraftable's cross-era position percentiles to sanity-check our own `production_percentiles.json` approach for athleticism.
- **Access method:** Unofficial JSON API (no key). One request per player; a few thousand calls total for our training set.
- **License/ToS risk:** Medium-low but honest caveat: the *code* is MIT, the *data* has no stated license and there are no published API terms. It's a hobbyist site (frontend repo last pushed 2019) — throttle hard (≥1s/request), cache locally, never hammer it, and don't redistribute the dump.
- **Integration effort:** **M** (slug resolution via typeahead + join; caching layer).
- **Expected model value:** Moderate. Arm/hand matter at OL/DL/CB margins; expect a small AUC bump, not a step change. Worth doing after consensus.

## 3. nflverse extras we don't use

All free CSV/parquet from nflverse releases; Python-native via [nflreadpy](https://github.com/nflverse/nflreadpy) (official Python port — no R needed). Function inventory verified on the [nflreadr reference index](https://nflreadr.nflverse.com/reference/index.html).

- **Contracts (`load_contracts()`, OverTheCap data)** — [dictionary](https://nflreadr.nflverse.com/articles/dictionary_contracts.html) verified: `apy`, `guaranteed`, `value`, `years`, `year_signed`, plus cap-inflation-adjusted variants. **What it adds:** the *second-contract* success label upgrade — "signed a veteran deal ≥ X inflated APY percentile at position" is a cleaner outcome than career AV and directly fixes the label problems in `IMPROVEMENT_PLAN.md` 2.1 (success currently entangled with draft round). Risk: none (openly published, MIT-tooled). Effort: **S–M**. Value: **high** — label quality moves every metric.
- **Snap counts (`load_snap_counts()`, PFR-sourced, 2012+)** — [dictionary](https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html). Adds "seasons as primary starter" (snap-share threshold) as a second outcome definition, crucial for OL where box-score outcomes are useless (our OL grading is a known weak spot). Effort: **S**. Value: moderate-high (labels, esp. OL).
- **PFR advanced seasonal stats (`load_pfr_advstats()`)** — pressure/coverage/missed-tackle style advanced NFL outcomes. Useful for finer-grained success labels only. Effort: **S**. Value: low-moderate.
- **Draft value charts** — verified in [nflverse/nfldata](https://github.com/nflverse/nfldata): [`data/draft_values.csv`](https://raw.githubusercontent.com/nflverse/nfldata/master/data/draft_values.csv) with columns `pick,stuart,johnson,hill,otc,pff` (fetched today). **What it adds:** converts draft slot (and consensus rank) onto a value scale — the right target transform for the draft-grade model (predicting *pick value* instead of round buckets), and the honest way to score "steal/reach" on the board UI. Effort: **S** (7 KB lookup table). Value: moderate, cheap.

## 4. CFBD endpoints we don't use (same API key we already have)

Endpoint existence verified today from the live OpenAPI spec at `https://api.collegefootballdata.com/api-docs.json` (74 endpoints; [docs](https://apinext.collegefootballdata.com/), free key with [Patreon tiers for higher limits](https://collegefootballdata.com/key); commercial use permitted per tier).

- **`/ppa/players/season` + `/player/usage`** — per-player Predicted Points Added and usage shares (pct of team plays/passing downs). **What it adds:** gap #2's usage/efficiency half: an opponent-and-context-adjusted production measure to replace our saturating `production_score` (the "100.0 wall", plan item 2.3), plus usage share = role importance. Effort: **M** (backfill 2014+ into the cache builder; PPA only exists in the pbp era). Value: **high** — best available upgrade to the production feature.
- **`/player/portal`** — transfer portal entries/destinations by year. Adds a `transferred` / destination-tier-delta feature (transfers up to power conferences carry signal; also fixes multi-school production attribution). Effort: **S–M**. Value: moderate.
- **`/player/returning`** — returning production by team; team-context feature (was he the offense?) and useful for the HS/college-stars pages. Effort: **S**. Value: low-moderate.
- **`/lines`** — betting lines per game. Team-strength signal we mostly already have via SP+; skip for prospect grading. Effort: S. Value: low.
- Also relevant to existing plan items: `/draft/picks` (cross-check nflverse), `/stats/player/success`, `/wepa/players/*` (EPA-style player metrics).

## 5. All-star game invites (Senior Bowl, East-West Shrine) — cheap consensus proxy

- **What it is:** [Senior Bowl accepted-invites list](https://www.seniorbowl.com/the-game/accepted-invites/) (official, current year) and per-year Wikipedia rosters with position/school ([2019](https://en.wikipedia.org/wiki/2019_Senior_Bowl), [2020](https://en.wikipedia.org/wiki/2020_Senior_Bowl), [2021](https://en.wikipedia.org/wiki/2021_Senior_Bowl), through [2026](https://en.wikipedia.org/wiki/2026_Senior_Bowl)); same pattern for the [Shrine Bowl](https://en.wikipedia.org/wiki/East%E2%80%93West_Shrine_Bowl).
- **What it adds:** A binary/ordinal scouting-consensus proxy (`senior_bowl_invite`, `shrine_invite`) that exists for *every* training year and requires no paid source. NFL scouts pick these rosters — it's crowd wisdom from the actual crowd that drafts.
- **Access method:** Wikipedia roster tables (CC BY-SA 4.0, attribution required — clean to use) for history; official site for the current cycle.
- **License/ToS risk:** Low (Wikipedia licensing is explicit; official invite lists are factual rosters).
- **Integration effort:** **S–M** (one-time table extraction for ~15 years + a yearly refresh script).
- **Expected model value:** Moderate — coarse (binary) but present across the whole training window; strongest for the Day 2–3 vs UDFA boundary where our model is weakest.

## 6. cfbfastR-data / sportsdataverse; College Football Reference status

- **[sportsdataverse/cfbfastR-data](https://github.com/sportsdataverse/cfbfastR-data):** verified — hosts CFB play-by-play, rosters, schedules, player stats, betting data 2002–present as csv.gz/parquet/RDS. **What it adds:** bulk pbp without burning CFBD API quota (it's largely CFBD-derived data re-hosted). Useful if we ever compute our own usage/EPA features locally instead of calling `/ppa/*`. Effort: **M–L** (pbp aggregation pipeline). Value: redundant with #4 unless we outgrow API limits.
- **College Football Reference (sports-reference.com/cfb):** **Do not build on scraping this.** Verified from their own [bot-traffic policy](https://www.sports-reference.com/bot-traffic.html) (updated May 2024): no API ("it's not our business model"), no bulk downloads ("we can not provide the data available as a download" due to upstream licensing), hard rate limits (20 req/min sitewide, 10 for FBref/Stathead), violators blocked up to a day — and their [data-use page](https://www.sports-reference.com/data_use.html) even returns 403 to non-browser fetchers. Occasional manual lookups are fine; a pipeline dependency is against their explicit posture. CFBD covers the same ground legitimately.

## 7. NFL Big Data Bowl / tracking data — not usable for us

- **What it is:** NGS tracking data released via Kaggle for the [Big Data Bowl](https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl); the [2026 edition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) covers 2023–24 **NFL** seasons.
- **Verdict: skip.** It contains only NFL players in NFL games — no college tracking exists publicly — and Kaggle competition data comes with competition-use license terms. There is no path from this to a *pre-draft* feature: by the time a player appears in tracking data, he's already drafted. The only theoretical use (retroactive athleticism validation of drafted players) doesn't feed the prospect model. Stating this explicitly so we don't revisit it.

## 8. HS recruiting beyond 247/CFBD

- **On3 Industry Ranking / RPM:** [RPM exists](https://www.on3.com/news/on3-releases-recruiting-prediction-machine-rpm/) and lives on On3 player profile pages, but it predicts *commitment destination*, not player quality, and there is **no public API or licensed data access** — it's a commercial subscription site. Scraping would violate their ToS. **Skip.**
- **Rivals / ESPN recruiting:** same story — no public bulk access.
- **Practical answer:** CFBD's `/recruiting/players` already serves the 247Composite (an industry aggregate) — the marginal signal from a second recruiting aggregator is tiny compared to the consensus-board gap. Not worth ToS risk. The one free upgrade here: we already have `hs_prospect_cache.json`; deepen recruiting *features* (positional rank, in-class percentile) from CFBD rather than adding a new source.

---

# Ranked Top-3 Recommendation

**#1 — Pre-draft consensus rank, assembled from three free/verified sources.**
Wide Left consensus boards (2024–2026, explicitly free data) + JackLich10 ESPN ranks/grades (2000–2021 backfill) + Senior Bowl/Shrine invite flags (all years), with a parallel outreach email to Grinding the Mocks for EDP history (2018–2023 would fill the gap years). This is the identified missing separator, and every piece is obtainable today at S–M effort.

**#2 — nflverse contracts + snap counts → rebuild the success label.**
Second-contract APY (inflation-adjusted, position-relative) and snap-share starter seasons replace the current AV/draft-round-entangled label. Improves everything trained on it, fixes OL, zero licensing risk, S–M effort. Do this alongside `IMPROVEMENT_PLAN.md` 2.1.

**#3 — CFBD `/ppa/players/season` + `/player/usage` → replace `production_score`.**
Opponent-adjusted per-player efficiency + usage share, from the API key we already have, directly attacks the production-saturation problem (plan item 2.3). M effort.

(MockDraftable arm/hand is #4 — real but smaller signal; do it once #1–3 land.)

## Integration sketch for #1 (consensus rank)

1. **New collector** `scripts/collect_consensus.py` (run with `.venv/bin/python`), writing `training_data/consensus_ranks.csv` with `name, draft_year, position, school, consensus_rank, consensus_source, board_size`:
   - 2000–2021: parse `nfl_draft_prospects.csv` from the JackLich10 repo at build time (download, don't commit — unlicensed ESPN data); take ESPN pre-draft `rank`/`grade`.
   - 2024–2026: parse the Wide Left "Get the data" sheets (manual download into `training_data/raw/`, one file per year).
   - 2018–2026 (upgrade path): swap in GTM EDP if/when Ben Robinson shares an extract; keep `consensus_source` so we can measure source quality.
   - All years: `senior_bowl_invite` / `shrine_invite` booleans from Wikipedia roster tables.
2. **Join** on normalized `(name, draft_year)` with position/school tie-breaking — reuse the fuzzy-match machinery and `*_match_type` audit column pattern already used for recruiting joins in `combine_outcomes_v3.csv`.
3. **Features** in `dv_features.py`: `consensus_rank_log = log2(1 + rank)` (rank 1 vs 10 matters more than 200 vs 250), `consensus_missing` indicator (NaN passed to XGBoost per plan 2.3 — *missing is itself informative*: unranked ≈ fringe prospect), invite booleans. Optionally `consensus_value` via nfldata `draft_values.csv` (`stuart`/`otc` column) to put rank on a pick-value scale.
4. **Leakage rule:** consensus rank is a *pre-draft* signal, so it's legal for both the draft-grade and success models — but freeze each year's board at its pre-draft date (the sources above are final pre-draft editions, so this holds by construction).
5. **Evaluate** exactly per plan 2.2: GroupKFold by draft year, report AUC/Brier vs (a) current features and (b) consensus-rank-alone baseline. The honest success criterion: the model must beat consensus alone, otherwise the product story is "we re-serve Arif Hasan's board."
6. **Inference/UI:** current-class board joins the newest Wide Left sheet at cache-build time (`build_prospect_cache.py`), surfacing `consensus_rank` next to our grade and a "vs consensus" delta — which is also the "edge" the `dv_edge.py` ledger wants to measure.

## Source index (all fetched 2026-08-19)

- Grinding the Mocks: https://grindingthemocks.shinyapps.io/Dashboard/ · https://grindingthemocks.substack.com/ · https://www.stat.cmu.edu/cmsac/conference/2020/assets/pdf/Robinson.pdf · https://x.com/GrindingMocks
- Wide Left consensus boards: https://wideleft.football/p/2026-nfl-draft-consensus-big-board · https://www.wideleft.football/p/2025-consensus-big-board-the-top · https://wideleft.football/p/2024-consensus-big-board-the-top
- ESPN historical draft data: https://github.com/JackLich10/nfl-draft-data
- MockDraftable: https://www.mockdraftable.com/ · API sample: https://www.mockdraftable.com/api/player?id=laquon-treadwell · https://github.com/marcusdarmstrong/mockdraftable-web
- nflverse: https://nflreadr.nflverse.com/reference/index.html · https://nflreadr.nflverse.com/articles/dictionary_contracts.html · https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html · https://nflreadr.nflverse.com/articles/dictionary_draft_picks.html · https://github.com/nflverse/nflreadpy · https://github.com/nflverse/nfldata · https://raw.githubusercontent.com/nflverse/nfldata/master/data/draft_values.csv
- CFBD: https://apinext.collegefootballdata.com/ · https://api.collegefootballdata.com/api-docs.json · https://collegefootballdata.com/key
- All-star games: https://www.seniorbowl.com/the-game/accepted-invites/ · https://en.wikipedia.org/wiki/Senior_Bowl · https://en.wikipedia.org/wiki/East%E2%80%93West_Shrine_Bowl
- sportsdataverse: https://github.com/sportsdataverse/cfbfastR-data
- Sports-Reference policy: https://www.sports-reference.com/bot-traffic.html · https://www.sports-reference.com/data_use.html
- Big Data Bowl: https://operations.nfl.com/programs-initiatives/innovation/big-data-bowl · https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction
- On3 RPM: https://www.on3.com/news/on3-releases-recruiting-prediction-machine-rpm/

---

## Relationship to docs/RESEARCH_ROADMAP.md

The owner's master research roadmap (RESEARCH_ROADMAP.md) is the long-term
vision; this document is the verified near-term supply. Mapping: roadmap §2
Recruiting + §4 Snap/Usage + §8 Opponent Quality are servable TODAY from the
sources above (CFBD recruiting/usage/PPA, nflverse snap counts, SP+/SOS);
§3 Transfer Portal is one verified CFBD endpoint away; §1 consensus/tracking
and §9 pressure metrics map to the Top-3 recommendation; §5 Injuries, §6
Coaching, §10 CV, §11 Environment, and §14 Character have no verified free
bulk source yet and are original-data-collection projects, not integrations.

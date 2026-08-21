# training_data/staging/ — acquired datasets for the v4 training wave

Staged 2026-08-21 by the fetchers in `scripts/data/` (all idempotent; raw
downloads cached under `/tmp/dv_training_cache/`; run with
`.venv/bin/python`). Closes the Top-3 gaps in `docs/DATASETS.md`
(pre-draft consensus, success-label upgrade, CFBD usage/PPA) plus the
Transfer Portal item from `docs/RESEARCH_ROADMAP.md` §3.

Match rates below come from `scripts/data/join_check.py`, which reuses the
exact normalizers (`norm_name`, `clean_school`, `_normalize_team`,
`_team_compatible`) from `scripts/build_training_data.py`. References:
(a) `training_data/combine_outcomes.csv`, (b) `training_data/prospect_cache.json`.

**None of these files should be committed** until licensing notes below are
revisited (several sources are free-to-use but not free-to-redistribute).

---

## consensus_wideleft.csv — 973 rows
- **Fetcher:** `scripts/data/fetch_wideleft.py`
- **What:** Arif Hasan's Wide Left consensus big boards, draft classes
  2024 (320 players, 101 analyst boards), 2025 (325, 112), 2026 (328, 134).
- **Columns:** draft_year, player, school, position, consensus_rank (board
  overall rank), pos_rank, n_boards, source_url.
- **Source:** Google Sheets linked ("Get the data") from
  https://wideleft.football/p/2024-consensus-big-board-the-top ·
  https://www.wideleft.football/p/2025-consensus-big-board-the-top ·
  https://wideleft.football/p/2026-nfl-draft-consensus-big-board
- **License:** data explicitly free per the posts, but no reuse license —
  derive features only, do not republish the board; credit Arif Hasan.
- **Match rate:** **95.7%** vs combine_outcomes (917 name+year+school,
  14 name-only, 42 none — misses are mostly late-round sleepers who were
  neither drafted nor combine-invited).

## consensus_espn.csv — 5,995 rows
- **Fetcher:** `scripts/data/fetch_jacklich_espn.py`
- **What:** ESPN pre-draft overall rank (`consensus_rank` = ovr_rk), position
  rank and 0-100 scouting grade per prospect. Requested window 2000-2021;
  **actual coverage starts 2004** (the source has no ranks/grades 2000-2003).
  ~250-400 ranked prospects per class.
- **Columns:** draft_year, player, school, position, consensus_rank, pos_rank,
  espn_grade, overall_pick, n_boards(=1 — single-outlet proxy).
- **Source:** https://github.com/JackLich10/nfl-draft-data
  (`nfl_draft_prospects.csv`, frozen 2021-05-05). Beware: in this source the
  `school` column is the school and `school_name` is the *mascot*.
- **License:** none (unlicensed ESPN scrape). Train privately; never commit
  or redistribute.
- **Match rate:** **90.2%** vs combine_outcomes (5,202 name+year+school).

## allstar_invites.csv — 2,673 rows
- **Fetcher:** `scripts/data/fetch_allstar_rosters.py`
- **What:** All-star game rosters (scouting-consensus proxy). One row per
  (year, player, school, position, game, source_url). Game year == draft-class
  year. Coverage kept only where a full roster (>= 60 players) parses:
  - senior_bowl: 2009-2010, 2012-2026 (17 years, 102-143 players/yr)
  - shrine: 2010, 2013-2017 (6 years, 104-115 players/yr)
  Skipped (no per-year article or partial "notable players" lists only):
  Senior Bowl 2000-2008 (articles exist only from 2007 and lack rosters
  until 2009) and 2011; Shrine 2000-2012 (except 2010), 2018-2026 (articles
  list only ~30-40 stat-line players or just coaches).
- **Source:** per-year Wikipedia game articles via the MediaWiki API; exact
  article URL recorded per row in `source_url`.
- **License:** CC BY-SA 4.0 (attribution: Wikipedia contributors) — clean.
- **Match rate:** **81.9%** vs combine_outcomes (misses are largely small-
  school/DII/CFL invitees who never got a combine invite or draft slot —
  which is itself signal).

## contracts.csv — 31,893 rows (9,732 players)
- **Fetcher:** `scripts/data/fetch_nflverse_labels.py`
- **What:** every OverTheCap contract via nflverse. One row per contract:
  player, otc_id, position, team, college, draft_year, draft_round,
  draft_overall, year_signed, years, value, apy, guaranteed, apy_cap_pct,
  inflated_apy, inflated_guaranteed, contract_index, is_second_contract.
- **is_second_contract logic:** contracts ordered per otc_id by year_signed
  (ties: larger value first); contract_index is 1-based; the flag marks
  index == 2 (rookie/UDFA deal is index 1; fifth-year options are not
  separate OTC rows). 5,391 second contracts. The v4 label should AND this
  with an inflated_apy position-percentile threshold, since minimum-salary
  re-ups also count as "second contracts" here.
- **Source:** https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.csv.gz
  (dictionary: https://nflreadr.nflverse.com/articles/dictionary_contracts.html)
- **License:** openly published nflverse data (MIT tooling) — no risk.
- **Match rate:** **98.1%** for distinct DRAFTED players in classes 2000-2026
  (n=3,861). All-players rate is 52.8% (n=9,408) because combine_outcomes
  structurally lacks undrafted non-combine street free agents — that is
  reference coverage, not join failure.

## snap_counts.csv — 27,110 player-seasons (7,090 players)
- **Fetcher:** `scripts/data/fetch_nflverse_labels.py`
- **What:** per player-season REG-game snap summary: player, pfr_player_id,
  season, position, teams, games, offense_snaps, offense_pct_mean,
  defense_snaps, defense_pct_mean, st_snaps, st_pct_mean. `*_pct_mean` is the
  mean share of team snaps across games the player appeared in (not the full
  team schedule). Basis for "seasons as primary starter" labels (esp. OL).
- **Coverage:** 2013-2025. The 2012 release asset is header-only upstream.
- **Source:** https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv
  (PFR-sourced; dictionary: https://nflreadr.nflverse.com/articles/dictionary_snap_counts.html)
- **License:** nflverse — no risk.
- **Match rate:** **71.7%** of distinct players vs combine_outcomes
  (name + draft-year<=first-snap-season window; misses are pre-2000 draftees
  still active in 2013+ and undrafted non-combine players).

## cfbd_transfer_portal.csv — 18,878 rows
- **Fetcher:** `scripts/data/fetch_cfbd_extras.py`
- **What:** transfer portal entries: season, player, position, origin,
  destination, transfer_date, stars, rating, eligibility.
- **Coverage:** 2021-2026 (all years CFBD publishes; 2018-2020 probed empty).
- **Source:** CFBD `/player/portal` (key from `.env` at runtime; responses
  cached in /tmp/dv_training_cache/cfbd/).
- **License:** CFBD free-tier terms (attribution).
- **Match rate:** 3.2% of portal rows reach a 2021-2026 draft class in
  combine_outcomes and 26.6% of portal names appear in the live prospect
  cache. **Low is structural, not a join bug**: most portal entrants never
  become draft prospects, and 2024-2026 entrants' classes are mostly 2027+.
  Use as a per-prospect lookup (transferred flag, origin->destination tier
  delta), not a spine.

## cfbd_usage.csv / cfbd_ppa.csv — 41,252 rows each
- **Fetcher:** `scripts/data/fetch_cfbd_extras.py`
- **What:** per (player, school, season): usage shares (overall/pass/rush/
  by-down) and opponent-adjusted PPA (avg by facet + total). Replaces the
  saturating `production_score` for skill positions.
- **Coverage:** seasons 2015-2025. **Offense skill positions only**
  (QB/RB/WR/TE/FB, FBS only) — defenders, OL and FCS players are absent from
  these CFBD endpoints by design.
- **Source:** CFBD `/player/usage` and `/ppa/players/season`.
- **License:** CFBD free-tier terms (attribution).
- **Match rate:** **92.7%** coverage of combine_outcomes QB/RB/WR/TE/FB rows,
  classes 2016-2026 (n=1,370; misses are FCS/DII prospects, e.g. Carson
  Wentz/NDSU). Informational: 14.5% of the 12,223 live prospect-cache names
  have a 2025 row — expected, since the cache spans all positions and deep
  FBS backups.

---

## Re-running everything

```
.venv/bin/python scripts/data/fetch_wideleft.py
.venv/bin/python scripts/data/fetch_jacklich_espn.py
.venv/bin/python scripts/data/fetch_allstar_rosters.py
.venv/bin/python scripts/data/fetch_nflverse_labels.py
.venv/bin/python scripts/data/fetch_cfbd_extras.py
.venv/bin/python scripts/data/join_check.py
```

All fetchers hit the network only on cache miss; `join_check.py` is offline.

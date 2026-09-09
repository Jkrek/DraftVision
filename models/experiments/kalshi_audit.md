# Kalshi integration audit — 2026-09-09

Read-only audit. No orders placed, no authenticated endpoints touched, no API
key used. Every Kalshi call below was an unauthenticated GET against the
public market-data API (`https://api.elections.kalshi.com/trade-api/v2`).

## Verdict: DEGRADED (silently broken, not dormant)

The page renders and never 500s, but it is showing the wrong markets with no
prices, and it is missing the NFL-draft markets that are open on Kalshi today.
Three independent defects, each sufficient on its own to prevent an edge from
ever being computed:

1. **Price parser reads fields Kalshi no longer serves.** `_yes_price_cents`
   reads integer-cent `last_price` / `yes_bid` / `yes_ask`. Kalshi's public
   payload now carries `last_price_dollars` / `yes_bid_dollars` /
   `yes_ask_dollars` (strings like `"0.7700"`) and the old keys are absent
   (`None`). Result: **0 of 145 markets** observed today have a price
   (`yes_price_cents: null` on every row in production). `_model_fields`
   returns `(None, None)` when price is None, so no edge, no ledger row, ever.
2. **Discovery misses the live 2027 draft books.** `_KNOWN_SERIES` is
   `(KXNFLDRAFTWR, KXNFLSDRAFTTOP, KXNCAAFTEAMRECTD)`. `KXNFLSDRAFTTOP` is the
   *Supplemental* draft series (title: "Will player be drafted Top X in
   Supplemental Draft") — 0 events, ever useful only in July. `KXNFLDRAFTWR`
   has five 2026 events with 0 markets attached. The real main-draft series
   (`KXNFLDRAFTTOP`, `KXNFLDRAFTPICK`, `KXNFLDRAFT1ST`, `KXNFLDRAFTOU`,
   `KXNFLDRAFT{QB,RB,WR,TE,OL,EDGE,DT,LB,DB}`, `KXNFLDRAFT1`, `KXHEISMAN`,
   `KXNFLCOMBINE40`) are not probed directly. The stage-2 fallback walk of
   `/events?status=open` is capped at 6 pages x 200; the draft events sit on
   **page 12** (2,400 events deep — pages 5-9 are 100% Elections). So the
   walk never reaches them and production serves only the stage-1 NCAAF
   team-TD rows.
3. **What the board shows instead is noise.** Production `/api/edge` returned
   36 `KXNCAAFTEAMRECTD` rows ("Arizona St.: 1+ receiving touchdowns", …) —
   team-level game props from week-3 CFB games, not player markets. 0 matched
   players, 0 prices, 0 edges. The Edge page renders a 36-row table of dashes
   rather than its own "seasonal" empty state, because `markets` is non-empty.

Secondary issues:

- **Cold-cache latency 30.5 s.** First `/api/edge` hit after deploy/TTL
  expiry: 3 series probes + 6 event pages x (HTTP + 0.6 s pacing) run inline
  on the request thread. Cached hits are 0.2 s. Fine at 10-min TTL, but a
  user landing on a cold worker waits half a minute; a request > 30 s would
  hit typical proxy timeouts.
- **Name ambiguity refuses the two most-traded players.** `_build_prospect_index`
  drops any normalized name shared by two different-team players. "Jeremiah
  Smith" (Ohio State WR, projected pick 1) collides with a Louisiana Tech LB;
  "Dylan Stewart" (South Carolina DE) with a Delaware OL. Both are unmatched
  in every 2027 event. The refusal is the right instinct; it just needs a
  tiebreak (prefer `draft_grade_class==0` / lowest `projected_pick`, or the
  market's position/team hint) before giving up.
- **Cache gaps.** Ryan Williams, Gunner Stockton, Bryce Underwood, Demond
  Williams Jr. (all in Kalshi's Top-5/Heisman books) are not in
  `prospect_cache.json` at all.
- **Honest-mapping policy is out of date.** The docstring says only
  `success_probability` and the grade bucket are persisted, so only
  "Top 32-50" markets can be priced. The cache now carries `projected_pick`
  (15,184 / 15,197 rows) and `model_pick` from the v5 pick head, and
  `models/experiments/pick_intervals_results.json` holds conformal pick
  intervals. That is exactly the quantity Top-5 / #1-overall / draft-position
  O/U markets ask about. Today the only 2027 draft-position event is Top-5
  (`KXNFLDRAFTTOP-27-5`), which the current 32<=X<=50 rule rejects — so even
  with (1) and (2) fixed, **0 edges would be computed today**.
- `_extract_top_n` regex `\btop[\s\-]*(\d{1,3})\b` will read "top 5" from
  "Will Arch Manning be a top 5 draft pick in 2027?" correctly (verified), but
  a first-round event like last cycle's `KXNFLDRAFTTOP-26-R1` ("Drafted in the
  1st Round") relies on the `first|1st round` fallback -> 32. OK.
- Ledger file: `training_data/edge_ledger.json` = `{"entries": []}`,
  untouched since 2026-08-18. Production ledger endpoint agrees (0 entries).
  Consistent with the above: nothing has ever qualified.
- `DEPLOY.md` says the ledger "resets on redeploy" — still true, still a gap
  once entries exist.

## 1. Code map

| Location | Role |
|---|---|
| `dv_edge.py` (494 lines) | All logic: discovery, cache, matching, edge math, paper ledger. |
| `XGBOost.py:3438-3452` | `import dv_edge`; registers `GET /api/edge` (calls `_maybe_reload_prospect_cache()` then `dv_edge.edge_payload(_PROSPECT_CACHE)`) and `GET /api/edge/ledger` (`dv_edge.ledger_payload()`). |
| `src/components/pages/Edge.js` (194 lines) | React page at `/edge`: fetches both endpoints once on mount, renders a market table (Market / Price / Model / Edge / Player / View link) and a paper-ledger table. Empty state ("No draft markets on the board … thickest January-April") shows only when `markets.length === 0` or fetch fails. |
| `src/App.js:44`, `src/components/Footer.js:12` | Route + footer link "Market Edge". |
| `training_data/edge_ledger.json` | Paper ledger, `{"entries": []}`. |
| `DEPLOY.md:225-245` | Operator notes. |
| `scripts/` | No Kalshi references. No tests. |

Data flow (`edge_payload`):
1. `_get_markets_cached()` — module-level dict under a lock, TTL 600 s
   (`time.monotonic`), per gunicorn worker. Failures cached too.
2. `_fetch_relevant_markets()` — stage 1: `GET /markets?series_ticker=S&status=open&limit=200`
   for each of 3 `_KNOWN_SERIES`; stage 2: up to 6 pages of
   `GET /events?status=open&limit=200&with_nested_markets=true`, filtered by
   `_is_relevant` (prefix `KXNFLDRAFT|KXNFLSDRAFT|KXNCAAFTEAMREC` or a title
   regex). 0.6 s sleep between requests, 8 s timeout, fail-soft per stage.
3. `_market_row` -> `_yes_price_cents` (broken, see above), `_candidate_names`
   (yes_sub_title/subtitle, then "Will X be drafted" title patterns, then
   Capitalized-word runs).
4. `_build_prospect_index` (accent-fold/lowercase/strip suffix; ambiguous
   names refused) -> lookup -> `_model_fields` (edge only if 32<=top_n<=50
   and `draft_grade_class==0`; `edge = success_probability - price_cents`).
5. |edge| >= 10 -> `_record_ledger_entries` (atomic temp+replace, one row per
   ticker/day, mtime hot-reload).

No auth anywhere in the codebase; no `KALSHI_*` env var, no order endpoints,
no SDK. The read-only design constraint is respected.

## 2. Live check (2026-09-09 ~17:10 UTC)

### Production (`https://draft.jkrek.com`)

| Endpoint | Status | Time | Payload |
|---|---|---|---|
| `GET /api/edge` (cold) | 200 | 30.5 s | `{generated_at, markets:[36], note:"Read-only analysis…"}` — all 36 = `KXNCAAFTEAMRECTD-26SEP1*` team receiving-TD props; `yes_price_cents: null` x36, `matched_player: null` x36, `edge: null` x36 |
| `GET /api/edge` (warm) | 200 | 0.23 s | same (cached) |
| `GET /api/edge/ledger` | 200 | 0.08 s | `{entries: [], note: "Paper ledger only…"}` |
| `GET /edge` | 200 | 0.15 s | SPA shell (React renders the page) |

Row shape: `{ticker, title, yes_price_cents, matched_player, matched_team, model_prob, edge, url}`.

### Kalshi public API (no auth)

Known-series probes as coded:

| Query | Result |
|---|---|
| `/markets?series_ticker=KXNFLDRAFTWR&status=open|settled|closed|unopened` | 200, 0 markets each |
| `/markets?series_ticker=KXNFLSDRAFTTOP&status=*` | 200, 0 markets each (series = *Supplemental* draft) |
| `/markets?series_ticker=KXNCAAFTEAMRECTD&status=open` | 200, 36 markets, `status:"active"`, all price fields `*_dollars` only |
| `/events?series_ticker=KXNFLDRAFTWR` | 5 events (`-26P1..P5`, "Nth WR drafted in 2026"), 0 nested markets, `available_on_brokers:false` |
| `/events?series_ticker=KXNFLSDRAFTTOP` | 0 events |

Draft markets that ARE open today (found via `/series` catalog, 13,912 series,
77 with "draft" in ticker/title, ~24 NFL):

| Event | Markets | Opened | Closes | Example prices (last $) | Volume |
|---|---|---|---|---|---|
| `KXNFLDRAFTTOP-27-5` "Players Drafted Top 5" | 19 active | 2026-06-23 | 2027-05-22 | Manning 0.77, Moore 0.84, J.Smith 0.90, Sayin 0.28 | low (26-2,650 contracts) |
| `KXNFLDRAFTPICK-27-1` "#1 Overall Pick in 2027" | 20 active | 2026-04-24 | 2027-05-22 | Manning 0.28, Mensah 0.17, J.Smith 0.15, Moore 0.14 | 2.9k-70k per player |
| `KXNFLDRAFT1ST-27` "Team to make 1st pick" | 32 active | 2026-05-01 | 2027-05-15 | ARI 0.21 | team markets — not player |
| `KXHEISMAN-27` | 38 active | 2026-01-05 | 2027-01-01 | Manning 0.09, Smith 0.07, Sayin 0.07 | 500k-1.4M per player |
| `KXNFLDRAFTOU` (draft-position O/U) | 0 open (26 event exists, no 27 yet) | — | — | — | — |
| `KXNFLDRAFT{QB,EDGE,WR}` "Nth <pos> drafted" | 2026 events only, 0 markets | — | — | — | — |

Direct probe confirms the cheap fix works: `/markets?series_ticker=KXNFLDRAFTTOP&status=open`
-> 19; `KXNFLDRAFTPICK` -> 20; `KXNFLDRAFT1ST` -> 32; `KXHEISMAN` -> 38.

Seasonality answer: **draft markets do NOT wait for winter.** Kalshi opened
the 2027 #1-overall book on 2026-04-24 (the day after the 2026 draft), team
#1-pick on 05-01, Top-5 on 06-23. The finer books (first-round Y/N per
player, Nth-QB/WR/EDGE, draft-position O/U) appeared for the 2026 cycle but
their markets are no longer served by the public API (settled/closed queries
return 0; `status=finalized` is rejected as invalid), so their open dates
could not be recovered — the 2026 event tickers (`KXNFLDRAFTTOP-26-R1`,
`KXNFLDRAFTOU-26`, `KXNFLDRAFTQB-26P2..4`) show they will exist; expect them
roughly Jan-Apr 2027 (post-declaration deadline, combine week).

### Offline pipeline test (saved 2027 payloads x local prospect cache, 15,197 rows)

- `_yes_price_cents` on 109 raw 2027 markets: **0 non-null**. A
  `*_dollars`-aware parser: 104 non-null.
- `_is_relevant`: True for all four 2027 events (prefix `KXNFLDRAFT` + Heisman regex) — the filter is fine; only reach is the problem.
- Name matching: Top-5 16/19 matched, #1-pick 17/20, Heisman 32/38. Misses =
  ambiguity refusals (Jeremiah Smith, Dylan Stewart) + cache absences (Ryan
  Williams, Gunner Stockton, Bryce Underwood, Demond Williams Jr.).
- `_model_fields`: 0 edges — Top-5 fails the 32<=X<=50 gate; #1-pick and
  Heisman have no top_n. Correct per the current policy.

## 3. Winter-ready checklist (ordered by leverage)

1. **Fix price parsing** (`_yes_price_cents`): read `last_price_dollars`,
   `yes_bid_dollars`, `yes_ask_dollars` (strings, dollars) and convert to
   cents; keep the old int-cent keys as fallback. Treat `"0.0000"` bid with
   no last trade as "no price", not 0. One-function change; unit-test it
   against a saved raw market dict.
2. **Fix discovery**: replace `_KNOWN_SERIES` with the real main-draft
   families — at minimum `KXNFLDRAFTTOP`, `KXNFLDRAFTPICK`, `KXNFLDRAFTOU`,
   `KXNFLDRAFTQB/RB/WR/TE/OL/EDGE/DT/LB/DB`, `KXNFLDRAFT1`, `KXHEISMAN`,
   `KXNFLCOMBINE40`; drop `KXNFLSDRAFTTOP` (supplemental). Stage 1 then does
   the real work; stage 2's 6-page walk can stay as a cheap "new series"
   sniffer but should not be relied on (draft events are ~2,400 deep).
   Alternative: sniff via `/series?category=Sports` (5.7 MB, one call) for
   tickers matching `KXNFLDRAFT` and probe those — no pagination problem.
3. **Drop or demote `KXNCAAFTEAMRECTD`** (team game props). It is not a
   player market, has no draft relevance, and is what currently fills the
   board. If "context" rows are wanted, gate them behind a matched player.
4. **Move discovery off the request thread**: warm the cache at startup /
   from the weekly refresh job (or a background thread) so a cold worker does
   not block 30 s. Persist the last good discovery to disk so a redeploy is
   not cold.
5. **Update the honest-mapping policy for the pick head**: `projected_pick`
   + conformal intervals (`pick_intervals_results.json`) can price
   "Top X" for any X and draft-position O/U honestly (P(pick <= X) from the
   interval / a persisted pick distribution). Persist per-player P(top5),
   P(top10), P(R1) in the cache at board-rebuild time rather than deriving
   from `success_probability`, which is a different quantity. Until then the
   current null-out is the right behavior — but then the copy on the page
   should say so instead of implying edges are seasonal.
6. **Name matching**: tiebreak ambiguous names by `draft_grade_class` /
   `projected_pick` (or require the collision to be within the same class
   before refusing); backfill missing 2027 names (Ryan Williams, Gunner
   Stockton, Bryce Underwood) in the prospect cache.
7. **Ledger durability**: export/commit `edge_ledger.json` in the weekly
   refresh commit (same as `board_history`), otherwise the track record dies
   on each `fly deploy`.
8. **Add a tiny regression test** with the saved raw payloads under
   `models/experiments/` or `tests/` so the next Kalshi schema drift fails
   loudly instead of degrading to "seasonal".
9. Frontend: when every row has `edge == null`, show the seasonal/empty state
   (or a "listed, not priced" banner) rather than a table of dashes.

Nothing in this checklist touches order endpoints or authentication; the
integration should remain read-only.

## Files / evidence

- Raw payloads saved in the session scratchpad (`k_*.json`, `out__api_edge.txt`), not committed.
- Repo: `dv_edge.py`, `XGBOost.py:3438-3452`, `src/components/pages/Edge.js`, `training_data/edge_ledger.json`, `DEPLOY.md:225-245`.

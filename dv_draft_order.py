#!/usr/bin/env python3
"""Projected NFL draft slots from a season simulation.

For every team: where its OWN first-round pick is likely to land next April,
as a distribution (10th / 50th / 90th percentile slot, P(top 5), P(top 10),
P(#1)), plus the first-round picks the team actually HOLDS once trades are
applied (`picks`). The slot_* fields stay the slot the team's own record
earns, whoever ends up holding it.

Trades: the "Round 1" list on Wikipedia's "<year> NFL draft" page, read
through the MediaWiki API and kept in training_data/pick_trades.json. Only
who-sent-a-first-to-whom is stored. Two conditions are understood: "whichever
is earlier" and "whichever is later" of the sender's picks. Protections and
other conditions are flagged, not modelled.

Method
  1. Team ratings (points, vs. an average team) are fit from the point
     spreads posted this season: spread = hfa + r_home - r_away, least
     squares with a ridge penalty and a recency weight. Spreads beat final
     margins as a rating source because they carry no single-game noise.
  2. Every unplayed regular-season game is simulated: home margin ~
     Normal(mean, GAME_SD), with the posted spread as the mean when one
     exists and the rating difference otherwise. Each simulation also draws
     a per-team rating shift ~ Normal(0, RATING_SD), so a team's future is
     allowed to differ from its market rating today.
  3. Playoff field (4 division winners + 3 wild cards per conference) and
     the bracket are simulated, then the NFL's ordering rule is applied:
     non-playoff teams by record (ties: weaker schedule picks earlier), then
     playoff teams by round of exit.

Simplifications, stated so nobody has to find them: seeding tie-breakers stop
at division record / conference record / coin flip (no head-to-head, common
games or strength of victory), and tied games are not simulated.

Data: nflverse `games.csv` (github.com/nflverse/nfldata, CC-BY 4.0).

Usage
  .venv/bin/python dv_draft_order.py                # write training_data/draft_order.json
  .venv/bin/python dv_draft_order.py --backtest     # score 2020-2025 at weeks 6/10/14
"""
import argparse
import datetime
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT_PATH = os.path.join(REPO_ROOT, "training_data", "draft_order.json")
BACKTEST_PATH = os.path.join(REPO_ROOT, "models", "experiments", "draft_order_backtest.json")
TRADES_PATH = os.path.join(REPO_ROOT, "training_data", "pick_trades.json")
WIKI_API = "https://en.wikipedia.org/w/api.php"

GAME_SD = 13.2        # sd of an NFL final margin around the closing spread
RATING_SD = 2.0       # per-simulation team rating shift (chosen in --backtest)
RIDGE = 0.5           # pulls thin-sample ratings toward average
HALF_LIFE_WEEKS = 6   # recency weight on the spreads used for ratings
N_SIMS = 20000
SEED = 20261111

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAMS = sorted(t for d in DIVISIONS.values() for t in d)
IDX = {t: i for i, t in enumerate(TEAMS)}
DIV_OF = np.zeros(32, dtype=int)
CONF_OF = np.zeros(32, dtype=int)
for _d, (_name, _members) in enumerate(DIVISIONS.items()):
    for _t in _members:
        DIV_OF[IDX[_t]] = _d
        CONF_OF[IDX[_t]] = 0 if _name.startswith("AFC") else 1


def load_games(path=None):
    if path:
        return pd.read_csv(path, low_memory=False)
    import requests
    r = requests.get(GAMES_URL, timeout=60, headers={"User-Agent": "DraftVision/1.0"})
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), low_memory=False)


def fit_ratings(games, upto_week):
    """Ridge least squares on posted spreads through `upto_week` (inclusive).

    nflverse `spread_line` is the expected home margin (positive = home
    favoured). Returns (ratings[32], hfa).
    """
    g = games[(games.game_type == "REG") & (games.week <= upto_week) & games.spread_line.notna()]
    n = len(g)
    X = np.zeros((n + 32, 33))
    y = np.zeros(n + 32)
    w = np.ones(n + 32)
    for k, (h, a, s, wk, loc) in enumerate(zip(g.home_team, g.away_team, g.spread_line, g.week, g.location)):
        X[k, IDX[h]] = 1.0
        X[k, IDX[a]] = -1.0
        X[k, 32] = 0.0 if loc == "Neutral" else 1.0
        y[k] = s
        w[k] = 0.5 ** ((upto_week - wk) / HALF_LIFE_WEEKS)
    for i in range(32):                      # ridge rows
        X[n + i, i] = 1.0
        w[n + i] = RIDGE
    sw = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * sw, y * sw[:, 0], rcond=None)
    r = beta[:32] - beta[:32].mean()
    return r, float(beta[32])


def _order_from(wins, gp, opp, exit_round, rng):
    """Draft slot (1..32) per team for each simulation.

    wins/gp: (N, 32) full regular season; opp: (32, 32) games vs each
    opponent; exit_round: (N, 32) 0 = missed playoffs, 1 = lost wild card,
    2 = divisional, 3 = conference, 4 = lost Super Bowl, 5 = won it.
    """
    pct = wins / np.maximum(gp, 1)
    sos = (wins @ opp.T) / np.maximum(gp @ opp.T, 1)
    key = exit_round * 10.0 + pct + sos * 0.01 + rng.random(wins.shape) * 1e-5
    return np.argsort(np.argsort(key, axis=1), axis=1) + 1


def _seed_conference(wins, div_w, conf_w, conf, rng):
    """(N, 7) team indices, seeds 1..7, for one conference."""
    n = wins.shape[0]
    members = np.where(CONF_OF == conf)[0]
    noise = rng.random((n, 32)) * 1e-6
    div_key = wins + div_w * 1e-2 + conf_w * 1e-4 + noise
    seed_key = wins + conf_w * 1e-2 + noise
    winners = np.zeros((n, 4), dtype=int)
    for j, d in enumerate(sorted(set(DIV_OF[members]))):
        dm = np.where(DIV_OF == d)[0]
        winners[:, j] = dm[np.argmax(div_key[:, dm], axis=1)]
    rows = np.arange(n)[:, None]
    worder = np.argsort(-seed_key[rows, winners], axis=1)
    top4 = winners[rows, worder]
    rest = seed_key[:, members].copy()
    is_winner = (members[None, :, None] == winners[:, None, :]).any(axis=2)
    rest[is_winner] = -1.0
    wc = members[np.argsort(-rest, axis=1)[:, :3]]
    return np.concatenate([top4, wc], axis=1)


def _play(a, b, home_edge, r, rng):
    """Winner/loser arrays for a vs b (a is home when home_edge != 0)."""
    rows = np.arange(len(a))
    margin = r[rows, a] - r[rows, b] + home_edge + rng.normal(0, GAME_SD, len(a))
    a_wins = margin > 0
    return np.where(a_wins, a, b), np.where(a_wins, b, a)


def simulate(games, season, upto_week, n_sims=N_SIMS, rating_sd=RATING_SD, seed=SEED):
    """Simulate the rest of `season` as of the end of `upto_week`.

    Games with week <= upto_week count as played (their real results); later
    ones are simulated. Returns dict of arrays plus the ratings used.
    """
    rng = np.random.default_rng(seed)
    g = games[(games.season == season) & (games.game_type == "REG")].copy()
    ratings, hfa = fit_ratings(g, upto_week + 1)     # next week's lines are already posted
    played = g[(g.week <= upto_week) & g.result.notna()]
    todo = g[~g.index.isin(played.index)]

    opp = np.zeros((32, 32))
    for h, a in zip(g.home_team, g.away_team):
        opp[IDX[h], IDX[a]] += 1
        opp[IDX[a], IDX[h]] += 1

    base = np.zeros((4, 32))                 # wins, games, division wins, conference wins
    for h, a, res in zip(played.home_team, played.away_team, played.result):
        hi, ai = IDX[h], IDX[a]
        hw = 1.0 if res > 0 else 0.5 if res == 0 else 0.0
        base[1, hi] += 1
        base[1, ai] += 1
        base[0, hi] += hw
        base[0, ai] += 1 - hw
        if DIV_OF[hi] == DIV_OF[ai]:
            base[2, hi] += hw
            base[2, ai] += 1 - hw
        if CONF_OF[hi] == CONF_OF[ai]:
            base[3, hi] += hw
            base[3, ai] += 1 - hw

    r = ratings[None, :] + rng.normal(0, rating_sd, (n_sims, 32))
    wins = np.tile(base[0], (n_sims, 1))
    gp = np.tile(base[1], (n_sims, 1))
    div_w = np.tile(base[2], (n_sims, 1))
    conf_w = np.tile(base[3], (n_sims, 1))
    for h, a, s, wk, loc in zip(todo.home_team, todo.away_team, todo.spread_line, todo.week, todo.location):
        hi, ai = IDX[h], IDX[a]
        edge = 0.0 if loc == "Neutral" else hfa
        if pd.notna(s) and wk <= upto_week + 1:
            # a posted line already prices tonight's injuries; keep the shift
            mean = s + (r[:, hi] - ratings[hi]) - (r[:, ai] - ratings[ai])
        else:
            mean = r[:, hi] - r[:, ai] + edge
        hw = (mean + rng.normal(0, GAME_SD, n_sims) > 0).astype(float)
        wins[:, hi] += hw
        wins[:, ai] += 1 - hw
        gp[:, hi] += 1
        gp[:, ai] += 1
        if DIV_OF[hi] == DIV_OF[ai]:
            div_w[:, hi] += hw
            div_w[:, ai] += 1 - hw
        if CONF_OF[hi] == CONF_OF[ai]:
            conf_w[:, hi] += hw
            conf_w[:, ai] += 1 - hw

    exit_round = np.zeros((n_sims, 32))
    rows = np.arange(n_sims)
    champs = []
    for conf in (0, 1):
        seeds = _seed_conference(wins, div_w, conf_w, conf, rng)
        seed_no = np.zeros((n_sims, 32), dtype=int)
        for k in range(7):
            seed_no[rows, seeds[:, k]] = k + 1
        alive = [seeds[:, 0]]
        for hi_seed, lo_seed in ((1, 6), (2, 5), (3, 4)):
            w_, l_ = _play(seeds[:, hi_seed], seeds[:, lo_seed], hfa, r, rng)
            exit_round[rows, l_] = 1
            alive.append(w_)
        alive = np.stack(alive, axis=1)
        order = np.argsort(seed_no[rows[:, None], alive], axis=1)
        alive = alive[rows[:, None], order]              # best seed first
        w1, l1 = _play(alive[:, 0], alive[:, 3], hfa, r, rng)
        w2, l2 = _play(alive[:, 1], alive[:, 2], hfa, r, rng)
        exit_round[rows, l1] = 2
        exit_round[rows, l2] = 2
        first_home = seed_no[rows, w1] < seed_no[rows, w2]
        home, away = np.where(first_home, w1, w2), np.where(first_home, w2, w1)
        wc_, lc_ = _play(home, away, hfa, r, rng)
        exit_round[rows, lc_] = 3
        champs.append(wc_)
    wsb, lsb = _play(champs[0], champs[1], 0.0, r, rng)
    exit_round[rows, lsb] = 4
    exit_round[rows, wsb] = 5

    slots = _order_from(wins, gp, opp, exit_round, rng)
    return {"slots": slots, "wins": wins, "exit_round": exit_round, "ratings": ratings,
            "hfa": hfa, "base": base, "opp": opp}


# ── First-round pick trades ──────────────────────────────────────────────────
WIKI_TEAM = {
    "arizona": "ARI", "atlanta": "ATL", "baltimore": "BAL", "buffalo": "BUF", "carolina": "CAR",
    "chicago": "CHI", "cincinnati": "CIN", "cleveland": "CLE", "dallas": "DAL", "denver": "DEN",
    "detroit": "DET", "green bay": "GB", "houston": "HOU", "indianapolis": "IND",
    "jacksonville": "JAX", "kansas city": "KC", "las vegas": "LV", "la chargers": "LAC",
    "los angeles chargers": "LAC", "la rams": "LA", "los angeles rams": "LA", "miami": "MIA",
    "minnesota": "MIN", "new england": "NE", "new orleans": "NO", "ny giants": "NYG",
    "new york giants": "NYG", "ny jets": "NYJ", "new york jets": "NYJ", "philadelphia": "PHI",
    "pittsburgh": "PIT", "san francisco": "SF", "seattle": "SEA", "tampa bay": "TB",
    "tennessee": "TEN", "washington": "WAS",
}
_BULLET = re.compile(r"'''\s*(.+?)\s*(?:→|->|&rarr;)\s*(.+?)\s*\((P?D)\)\.?\s*'''")
_MONTHS = {m: i + 1 for i, m in enumerate(
    "january february march april may june july august september october november december".split())}


def _wiki_date(text):
    m = re.search(r"\|\s*date\s*=\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", text)
    if not m or m.group(1).lower() not in _MONTHS:
        return None
    return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"


def parse_trades(wikitext):
    """Round-1 bullets of a Wikipedia '<year> NFL draft' page -> trade list.

    Raises ValueError on a bullet whose teams cannot be read, so a changed
    page format fails loudly instead of silently dropping a trade.
    """
    m = re.search(r"===\s*Round 1\s*===(.*?)(?:\n==|\Z)", wikitext, re.S)
    if not m:
        return []
    out = []
    for line in m.group(1).splitlines():
        if not line.startswith("*"):
            continue
        line = re.sub(r"\{\{\s*Anchor\s*\|[^}]*\}\}", "", line, flags=re.I)
        b = _BULLET.search(line)
        if not b:
            raise ValueError(f"unreadable trade bullet: {line[:80]!r}")
        frm, to = (WIKI_TEAM.get(x.strip().lower()) for x in b.group(1, 2))
        if not frm or not to:
            raise ValueError(f"unknown team in trade bullet: {b.group(0)!r}")
        body = re.sub(r"<ref.*?(</ref>|/>)", "", line[b.end():], flags=re.S)
        body = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", body).strip()
        low = body.lower()
        mode = ("earlier" if "whichever is earlier" in low or "the earlier of" in low else
                "later" if "whichever is later" in low or "the later of" in low else "own")
        # "traded a first-round selection it acquired from X": X's pick moves on
        via = None
        v = re.match(r"[^.]*?traded (?:a|its|their) first-round selection,? (?:\(?(?:which )?(?:it|they) |\(?)?"
                     r"(?:originally |previously )?(?:acquired |received )?from ([A-Z][A-Za-z ]+?)[,.) ]", body)
        if mode == "own" and v:
            via = WIKI_TEAM.get(v.group(1).strip().lower())
        out.append({
            "from": frm, "to": to, "mode": mode, "via": via,
            "during_draft": b.group(3) == "D",
            "date": _wiki_date(line),
            "unmodelled_condition": mode == "own" and bool(re.search(r"conditional|protected", low)),
            "summary": body[:400],
        })
    # the page lists them in order; a stable sort on date only fixes a misplaced one
    return sorted(out, key=lambda t: t["date"] or "9999")


def fetch_trades(draft_year):
    import requests
    r = requests.get(WIKI_API, timeout=60, headers={"User-Agent": "DraftVision/1.0 (draft.jkrek.com)"},
                     params={"action": "parse", "page": f"{draft_year}_NFL_draft", "prop": "wikitext",
                             "format": "json", "formatversion": 2})
    r.raise_for_status()
    d = r.json()
    if "error" in d:
        raise ValueError(d["error"].get("info", "wikipedia error"))
    return parse_trades(d["parse"]["wikitext"])


def load_trades(draft_year, refresh=True, path=TRADES_PATH):
    """Committed trade list for `draft_year`, refreshed from Wikipedia when
    possible. A refresh that fails, or that lists fewer trades than the file
    already holds (a trade cannot be undone), keeps the file."""
    old = None
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
        if old.get("draft_year") != draft_year:
            old = None
    if refresh:
        try:
            trades = fetch_trades(draft_year)
            if old and len(trades) < len(old["trades"]):
                raise ValueError(f"page lists {len(trades)} trades, file has {len(old['trades'])}")
            if not old or old["trades"] != trades:
                old = {
                    "draft_year": draft_year,
                    "as_of": datetime.date.today().isoformat(),
                    "source": f"https://en.wikipedia.org/wiki/{draft_year}_NFL_draft",
                    "license": "facts from Wikipedia (text CC BY-SA 4.0)",
                    "trades": trades,
                }
                with open(path, "w") as fh:
                    json.dump(old, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")
        except Exception as exc:                      # noqa: BLE001 - best effort
            print(f"pick trades not refreshed ({exc}); using the committed file", file=sys.stderr)
    return old


def apply_trades(slots, trades):
    """owner[sim, pick] = index of the team holding the pick that began as
    team `pick`'s own.

    Trades are applied in order. 'own' moves the sender's original pick (or
    the pick named by `via`); 'earlier' / 'later' move whichever of the
    sender's picks lands earlier / later in that simulation.
    """
    n = slots.shape[0]
    rows = np.arange(n)
    owner = np.tile(np.arange(32), (n, 1))
    for t in trades or []:
        a, b = IDX[t["from"]], IDX[t["to"]]
        held = owner == a
        if t.get("mode") in ("earlier", "later"):
            sign = 1 if t["mode"] == "earlier" else -1
            j = np.argmin(np.where(held, sign * slots, np.inf), axis=1)
        else:
            j = np.full(n, IDX[t["via"]] if t.get("via") else a)
        ok = held[rows, j]                # the sender must still hold it
        owner[rows[ok], j[ok]] = b
    return owner


def _slot_stats(s):
    return {
        "slot_mean": round(float(s.mean()), 1),
        "slot_p10": int(np.percentile(s, 10)),
        "slot_p50": int(np.percentile(s, 50)),
        "slot_p90": int(np.percentile(s, 90)),
        "p_first": round(float((s == 1).mean()), 3),
        "p_top5": round(float((s <= 5).mean()), 3),
        "p_top10": round(float((s <= 10).mean()), 3),
    }


def held_picks(slots, owner, team_idx):
    """The first-round picks a team holds, earliest first, each with its slot
    distribution and the team whose record most often produces it."""
    held = owner == team_idx
    k = int(held[0].sum())               # trades move one pick per sim: constant
    if k == 0:
        return []
    s = np.where(held, slots, 99)
    order = np.argsort(s, axis=1)[:, :k]
    picks = []
    for r in range(k):
        src = order[:, r]
        origin = int(np.bincount(src, minlength=32).argmax())
        picks.append(_slot_stats(np.take_along_axis(s, order[:, r:r + 1], axis=1)[:, 0]) | {
            "from": TEAMS[origin],
            "own": origin == team_idx,
            "from_share": round(float((src == origin).mean()), 3),
        })
    return picks


def actual_slots(games, season):
    """The order the real results produced, by the same ordering rule."""
    g = games[games.season == season]
    reg = g[g.game_type == "REG"]
    if reg.result.isna().any():
        return None
    wins, gp, opp = np.zeros((1, 32)), np.zeros((1, 32)), np.zeros((32, 32))
    for h, a, res in zip(reg.home_team, reg.away_team, reg.result):
        hi, ai = IDX[h], IDX[a]
        hw = 1.0 if res > 0 else 0.5 if res == 0 else 0.0
        wins[0, hi] += hw
        wins[0, ai] += 1 - hw
        gp[0, hi] += 1
        gp[0, ai] += 1
        opp[hi, ai] += 1
        opp[ai, hi] += 1
    exit_round = np.zeros((1, 32))
    post = g[g.game_type != "REG"]
    if post.result.isna().any() or len(post) < 13:
        return None
    for gt, h, a, res in zip(post.game_type, post.home_team, post.away_team, post.result):
        loser, winner = (a, h) if res > 0 else (h, a)
        exit_round[0, IDX[loser]] = {"WC": 1, "DIV": 2, "CON": 3, "SB": 4}[gt]
        if gt == "SB":
            exit_round[0, IDX[winner]] = 5
    # coin-flip ties get the midpoint of many flips
    rng = np.random.default_rng(0)
    reps = np.stack([_order_from(wins, gp, opp, exit_round, rng)[0] for _ in range(200)])
    return reps.mean(axis=0)


def summarise(sim, trades=None):
    slots, wins, ex, base = sim["slots"], sim["wins"], sim["exit_round"], sim["base"]
    owner = apply_trades(slots, trades)
    out = []
    for t in TEAMS:
        i = IDX[t]
        w, gpd = base[0, i], base[1, i]
        losses = gpd - np.floor(w) - (1 if w % 1 else 0)
        holder = int(np.bincount(owner[:, i], minlength=32).argmax())
        out.append({
            "team": t,
            "record": f"{int(w)}-{int(losses)}" + ("-1" if w % 1 else ""),
            "rating": round(float(sim["ratings"][i]), 1),
            "projected_wins": round(float(wins[:, i].mean()), 1),
            "playoff_prob": round(float((ex[:, i] > 0).mean()), 3),
            **_slot_stats(slots[:, i]),
            # who holds the pick this team's record earns, and what it holds
            "own_pick_holder": TEAMS[holder],
            "own_pick_holder_share": round(float((owner[:, i] == holder).mean()), 3),
            "picks": held_picks(slots, owner, i),
            "acquired_from": [x["from"] for x in trades or [] if x["to"] == t],
            "sent_to": [{"team": x["to"], "mode": x["mode"]} for x in trades or [] if x["from"] == t],
        })
    out.sort(key=lambda d: d["slot_mean"])
    return out


def backtest(games, seasons=range(2020, 2026), weeks=(6, 10, 14), n_sims=4000):
    """Score the projection against the order the real results produced."""
    report = {}
    for rating_sd in (0.0, 1.0, 2.0, 3.0):
        for wk in weeks:
            err, base_err, cover, n = [], [], [], 0
            b5, b5_base = [], []
            for season in seasons:
                truth = actual_slots(games, season)
                if truth is None:
                    continue
                sim = simulate(games, season, wk, n_sims=n_sims, rating_sd=rating_sd, seed=SEED + season)
                s = sim["slots"]
                err.extend(np.abs(s.mean(axis=0) - truth))
                lo, hi = np.percentile(s, 10, axis=0), np.percentile(s, 90, axis=0)
                cover.extend((truth >= lo - 0.5) & (truth <= hi + 0.5))
                # baseline: today's standings, frozen (ties by point differential)
                g = games[(games.season == season) & (games.game_type == "REG") & (games.week <= wk)]
                pd_ = np.zeros(32)
                for h, a, res in zip(g.home_team, g.away_team, g.result):
                    pd_[IDX[h]] += res
                    pd_[IDX[a]] -= res
                pct = sim["base"][0] / np.maximum(sim["base"][1], 1)
                naive = np.argsort(np.argsort(pct + pd_ * 1e-5)) + 1
                base_err.extend(np.abs(naive - truth))
                y5 = (truth <= 5.5).astype(float)
                b5.extend(((s <= 5).mean(axis=0) - y5) ** 2)
                b5_base.extend(((naive <= 5).astype(float) - y5) ** 2)
                n += 32
            report[f"rating_sd={rating_sd}|week={wk}"] = {
                "team_seasons": n,
                "mae_slots": round(float(np.mean(err)), 2),
                "mae_frozen_standings": round(float(np.mean(base_err)), 2),
                "coverage_p10_p90": round(float(np.mean(cover)), 3),
                "top5_brier": round(float(np.mean(b5)), 4),
                "top5_brier_frozen_standings": round(float(np.mean(b5_base)), 4),
                "top5_brier_base_rate": round(float((5 / 32) * (27 / 32)), 4),
            }
            print(f"rating_sd={rating_sd} week {wk:>2}: {report[f'rating_sd={rating_sd}|week={wk}']}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", help="local games.csv (default: download from nflverse)")
    ap.add_argument("--season", type=int)
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--no-trade-refresh", action="store_true",
                    help="use the committed pick_trades.json without asking Wikipedia")
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    games = load_games(args.games)
    games = games[games.home_team.isin(TEAMS) & games.away_team.isin(TEAMS)]

    if args.backtest:
        rep = backtest(games)
        os.makedirs(os.path.dirname(BACKTEST_PATH), exist_ok=True)
        json.dump({"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "seasons": "2020-2025", "report": rep}, open(BACKTEST_PATH, "w"), indent=2)
        print(f"wrote {BACKTEST_PATH}")
        return 0

    season = args.season or int(games[games.result.notna()].season.max())
    reg = games[(games.season == season) & (games.game_type == "REG")]
    done = reg[reg.result.notna()]
    if done.empty:
        print(f"no completed {season} games yet", file=sys.stderr)
        return 2
    # the last week in which every scheduled game has a result
    by_week = reg.groupby("week").result.apply(lambda s: s.notna().all())
    complete = [w for w, ok in by_week.items() if ok]
    upto = max(complete) if complete else int(done.week.max())
    sim = simulate(games, season, upto)
    traded = load_trades(season + 1, refresh=not args.no_trade_refresh)
    trades = (traded or {}).get("trades", [])
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "season": season,
        "draft_year": season + 1,
        "through_week": int(upto),
        "games_played": int(len(done)),
        "games_total": int(len(reg)),
        "n_sims": N_SIMS,
        "method": "spread-implied ratings, simulated remainder + playoffs, NFL ordering rule",
        "caveats": ["first-round trades applied; pick protections and conditions other than "
                    "'earlier/later of two picks' are not",
                    "seeding tie-breakers simplified to division/conference record, then coin flip"],
        "source": "nflverse games.csv (CC-BY 4.0)",
        "trades_as_of": (traded or {}).get("as_of"),
        "trades_source": (traded or {}).get("source"),
        "trades": [{k: t[k] for k in ("from", "to", "mode", "via", "date", "unmodelled_condition")}
                   for t in trades],
        "teams": summarise(sim, trades),
    }
    json.dump(payload, open(args.out, "w"), indent=2)
    print(f"{season} through week {upto}: wrote {args.out}")
    for t in payload["teams"]:
        print(f"{t['team']:<4}{t['record']:>7}  rating {t['rating']:>5}  slot {t['slot_p10']:>2}-{t['slot_p50']:>2}-{t['slot_p90']:>2}"
              f"  #1 {t['p_first']:.0%}  top5 {t['p_top5']:.0%}  top10 {t['p_top10']:.0%}  playoffs {t['playoff_prob']:.0%}"
              f"  holds {', '.join(str(p['slot_p50']) + ('' if p['own'] else '<' + p['from']) for p in t['picks']) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

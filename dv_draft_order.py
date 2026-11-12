#!/usr/bin/env python3
"""Projected NFL draft slots from a season simulation.

For every team: where its OWN first-round pick is likely to land next April,
as a distribution (10th / 50th / 90th percentile slot, P(top 5), P(top 10),
P(#1)). Traded picks are not modelled here: the output is the slot a team's
record earns, whoever ends up holding it.

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
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUT_PATH = os.path.join(REPO_ROOT, "training_data", "draft_order.json")
BACKTEST_PATH = os.path.join(REPO_ROOT, "models", "experiments", "draft_order_backtest.json")

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


def summarise(sim):
    slots, wins, ex, base = sim["slots"], sim["wins"], sim["exit_round"], sim["base"]
    out = []
    for t in TEAMS:
        i = IDX[t]
        s = slots[:, i]
        w, gpd = base[0, i], base[1, i]
        losses = gpd - np.floor(w) - (1 if w % 1 else 0)
        out.append({
            "team": t,
            "record": f"{int(w)}-{int(losses)}" + ("-1" if w % 1 else ""),
            "rating": round(float(sim["ratings"][i]), 1),
            "projected_wins": round(float(wins[:, i].mean()), 1),
            "playoff_prob": round(float((ex[:, i] > 0).mean()), 3),
            "slot_mean": round(float(s.mean()), 1),
            "slot_p10": int(np.percentile(s, 10)),
            "slot_p50": int(np.percentile(s, 50)),
            "slot_p90": int(np.percentile(s, 90)),
            "p_first": round(float((s == 1).mean()), 3),
            "p_top5": round(float((s <= 5).mean()), 3),
            "p_top10": round(float((s <= 10).mean()), 3),
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
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "season": season,
        "draft_year": season + 1,
        "through_week": int(upto),
        "games_played": int(len(done)),
        "games_total": int(len(reg)),
        "n_sims": N_SIMS,
        "method": "spread-implied ratings, simulated remainder + playoffs, NFL ordering rule",
        "caveats": ["own-pick slot only; traded picks are not applied",
                    "seeding tie-breakers simplified to division/conference record, then coin flip"],
        "source": "nflverse games.csv (CC-BY 4.0)",
        "teams": summarise(sim),
    }
    json.dump(payload, open(args.out, "w"), indent=2)
    print(f"{season} through week {upto}: wrote {args.out}")
    for t in payload["teams"]:
        print(f"{t['team']:<4}{t['record']:>7}  rating {t['rating']:>5}  slot {t['slot_p10']:>2}-{t['slot_p50']:>2}-{t['slot_p90']:>2}"
              f"  #1 {t['p_first']:.0%}  top5 {t['p_top5']:.0%}  top10 {t['p_top10']:.0%}  playoffs {t['playoff_prob']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

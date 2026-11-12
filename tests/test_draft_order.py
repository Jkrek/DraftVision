"""Draft-slot simulation: the ordering rule and the bracket bookkeeping."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dv_draft_order as dvo


def _season(strong="BUF", weak="TEN"):
    """A made-up nine-week season: week 1 played, the rest unplayed."""
    rows = []
    teams = dvo.TEAMS
    for wk in range(1, 10):
        shift = teams[wk:] + teams[:wk]
        seen = set()
        for h, a in zip(teams, shift):
            if h in seen or a in seen:
                continue
            seen.update((h, a))
            spread = 14.0 * ((h == strong) - (a == strong)) - 14.0 * ((h == weak) - (a == weak))
            rows.append({"season": 2099, "game_type": "REG", "week": wk, "home_team": h, "away_team": a,
                         "location": "Home", "spread_line": spread,
                         "result": (3.0 if wk == 1 else np.nan)})
    return pd.DataFrame(rows)


def test_every_simulation_is_a_full_order_with_one_champion():
    sim = dvo.simulate(_season(), 2099, 1, n_sims=300, seed=1)
    slots, ex = sim["slots"], sim["exit_round"]
    assert (np.sort(slots, axis=1) == np.arange(1, 33)).all()
    counts = [(ex == k).sum(axis=1) for k in range(6)]
    assert [int(c[0]) for c in counts] == [18, 6, 4, 2, 1, 1]
    assert all((c == c[0]).all() for c in counts)
    rows = np.arange(len(slots))
    champ = np.argmax(ex, axis=1)
    assert (slots[rows, champ] == 32).all()
    # every playoff team picks after every non-playoff team
    assert slots[ex == 0].max() == 18 and slots[ex > 0].min() == 19


def test_market_rating_moves_the_slot():
    sim = dvo.simulate(_season(), 2099, 1, n_sims=2000, seed=2)
    mean = sim["slots"].mean(axis=0)
    assert mean[dvo.IDX["TEN"]] < mean[dvo.IDX["BUF"]] - 5


def test_order_breaks_record_ties_by_weaker_schedule():
    wins = np.zeros((1, 32))
    gp = np.full((1, 32), 2.0)
    opp = np.zeros((32, 32))
    # teams 0 and 1 are both 0-2; team 0 played the two best teams
    wins[0, 2:] = 1.0
    wins[0, [4, 5]] = 2.0
    for a, b in ((0, 4), (0, 5), (1, 2), (1, 3)):
        opp[a, b] = opp[b, a] = 1
    slots = dvo._order_from(wins, gp, opp, np.zeros((1, 32)), np.random.default_rng(0))
    assert slots[0, 1] == 1 and slots[0, 0] == 2


WIKI = """==Trades involving draft picks==
===Round 1===
*'''Green Bay → Dallas (PD).''' Green Bay traded a first-round selection and DT [[Kenny Clark (defensive tackle)|Kenny Clark]] to Dallas in exchange for DE [[Micah Parsons]].<ref group=Trade>{{Cite web|title=x|date=August 28, 2025}}</ref>
*{{Anchor|X}}'''Dallas → NY Jets (PD).''' Dallas traded a first-round selection to the NY Jets. New York will receive whichever is earlier of Dallas' original selection or the selection they acquired from Green Bay.<ref group=Trade>{{cite news|date=November 4, 2025}}</ref>

===Round 2===
*'''Houston → Arizona (PD).''' Houston traded a second-round selection.
"""


def test_trade_bullets_are_read_from_round_one_only():
    trades = dvo.parse_trades(WIKI)
    assert [(t["from"], t["to"], t["mode"], t["date"]) for t in trades] == [
        ("GB", "DAL", "own", "2025-08-28"), ("DAL", "NYJ", "earlier", "2025-11-04")]
    assert "Kenny Clark" in trades[0]["summary"] and "[[" not in trades[0]["summary"]


def test_unreadable_trade_bullet_fails_loudly():
    try:
        dvo.parse_trades("===Round 1===\n*'''Gotham → Dallas (PD).''' x\n")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_earlier_of_two_picks_goes_to_the_buyer_in_every_simulation():
    sim = dvo.simulate(_season(), 2099, 1, n_sims=400, seed=3)
    slots = sim["slots"]
    owner = dvo.apply_trades(slots, dvo.parse_trades(WIKI))
    gb, dal, nyj = (dvo.IDX[t] for t in ("GB", "DAL", "NYJ"))
    assert (owner == gb).sum() == 0
    assert ((owner == dal).sum(axis=1) == 1).all() and ((owner == nyj).sum(axis=1) == 2).all()
    earlier = np.minimum(slots[:, gb], slots[:, dal])
    got = np.where(owner[:, gb] == nyj, slots[:, gb], slots[:, dal])
    assert (got == earlier).all()
    # every pick still has exactly one holder, and summaries agree
    teams = {t["team"]: t for t in dvo.summarise(sim, dvo.parse_trades(WIKI))}
    assert sum(len(t["picks"]) for t in teams.values()) == 32
    assert teams["GB"]["picks"] == [] and len(teams["NYJ"]["picks"]) == 3 - 1
    assert teams["DAL"]["acquired_from"] == ["GB"] and teams["DAL"]["sent_to"] == [{"team": "NYJ", "mode": "earlier"}]


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)

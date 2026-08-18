#!/usr/bin/env python3
"""Generate a YouTube-ready weekly recap script from the board refresh outputs.

Reads (all read-only):
  training_data/board_movers.json           risers/fallers vs prior snapshot
  training_data/board_history/board_*.json  most recent snapshot(s)
  training_data/big_boards.json             owner big boards (Name|Team lists)
  training_data/prospect_cache.json         (optional) for verified-stat %

Writes:
  content/recaps/recap_<YYYY-MM-DD>.md

Always exits 0; if movers are missing/empty it writes a minimal recap.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING = os.path.join(ROOT, "training_data")
OUT_DIR = os.path.join(ROOT, "content", "recaps")

TAKE = "<!-- add take -->"


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def latest_snapshots(n=2):
    paths = sorted(glob.glob(os.path.join(TRAINING, "board_history", "board_*.json")))
    snaps = []
    for p in paths[-n:]:
        data = load_json(p)
        if data and isinstance(data.get("players"), list):
            snaps.append((os.path.basename(p), data))
    return snaps  # oldest -> newest


def recap_date(movers):
    stamp = (movers or {}).get("generated_at", "")
    try:
        return datetime.fromisoformat(stamp).date().isoformat()
    except ValueError:
        return date.today().isoformat()


def fmt_delta(v, suffix=""):
    return f"{'+' if v >= 0 else ''}{v:g}{suffix}"


def mover_lines(movers_list, limit=5):
    lines = []
    for i, p in enumerate(movers_list[:limit], 1):
        lines.append(
            f"{i}. **{p['name']}** — {p.get('position', '?')}, {p.get('team', '?')} — "
            f"now **{p.get('success_probability', '?')}%** ({p.get('grade', '?')}), "
            f"{fmt_delta(p.get('delta_prob', 0), ' pts')} / "
            f"{fmt_delta(p.get('delta_rank', 0), ' board spots')}"
        )
        lines.append(f"   - Why it matters: {TAKE}")
    return lines


def new_top_grades(snaps):
    """A+/A players in the newest snapshot who weren't A+/A (or present) last week."""
    if not snaps:
        return None, "No board snapshots found."
    newest = snaps[-1][1]["players"]
    top_now = [p for p in newest if p.get("grade") in ("A+", "A")]
    if len(snaps) < 2:
        return None, (
            f"First tracked snapshot — baseline week. {len(top_now)} players sit at "
            "A+/A; new-entry tracking starts next refresh."
        )
    prev = snaps[-2][1]["players"]
    prev_top = {f"{p['name']}|{p.get('team', '')}" for p in prev if p.get("grade") in ("A+", "A")}
    fresh = [p for p in top_now if f"{p['name']}|{p.get('team', '')}" not in prev_top]
    fresh.sort(key=lambda p: p.get("rank", 10**9))
    return fresh, None


def biggest_disagreement(big_boards, snaps):
    """Largest gap between owner big-board rank and model rank in the latest snapshot."""
    if not big_boards or not snaps:
        return None
    model_rank = {
        f"{p['name']}|{p.get('team', '')}": p.get("rank")
        for p in snaps[-1][1]["players"]
    }
    best = None
    for year, entries in sorted(big_boards.items()):
        if not isinstance(entries, list):
            continue
        for i, key in enumerate(entries, 1):
            mr = model_rank.get(key)
            if mr is None:
                continue
            gap = mr - i
            if best is None or abs(gap) > abs(best["gap"]):
                name, _, team = key.partition("|")
                best = {"name": name, "team": team, "year": year,
                        "owner_rank": i, "model_rank": mr, "gap": gap}
    return best


def verified_stat_pct():
    cache = load_json(os.path.join(TRAINING, "prospect_cache.json"))
    prospects = (cache or {}).get("prospects")
    if not prospects:
        return None
    verified = sum(1 for p in prospects if p.get("data_source") == "espn_live")
    return round(100.0 * verified / len(prospects), 1)


def build_recap(day, movers, snaps, big_boards):
    lines = [f"# DraftVision Weekly Recap — {day}", ""]

    risers = (movers or {}).get("risers") or []
    fallers = (movers or {}).get("fallers") or []

    if not risers and not fallers:
        lines += [
            "_No board movement recorded this week (movers file missing or empty)._",
            "",
            "Use this week for an evergreen: board methodology, a position deep-dive,",
            "or a viewer-mailbag episode.",
        ]
        return "\n".join(lines) + "\n"

    r1 = risers[0]
    board_n = len(snaps[-1][1]["players"]) if snaps else (movers or {}).get("count", "?")

    lines += [
        "## Title options",
        f"1. {r1['name']} Just Broke Our Model — Weekly Board Recap {day}",
        f"2. {len(risers)} Risers, {len(fallers)} Fallers: The Machine Re-Ranked "
        f"{board_n} Prospects",
        f"3. Who the Model Loves (and Who It Just Gave Up On) — Week of {day}",
        "",
        "## Cold open",
        f"> \"{r1['name']} jumped {fmt_delta(r1.get('delta_rank', 0))} spots this week — "
        f"the model now has him at {r1.get('success_probability', '?')}%. "
        "Here's what moved, and why.\"",
        "",
        "## Top 5 risers",
    ]
    lines += mover_lines(risers)
    lines += ["", "## Top 5 fallers"]
    lines += mover_lines(fallers)

    lines += ["", "## New A+/A entries this week"]
    fresh, note = new_top_grades(snaps)
    if note:
        lines.append(f"_{note}_")
    elif not fresh:
        lines.append("_None — the top of the board held steady._")
    else:
        for p in fresh[:10]:
            lines.append(
                f"- **{p['name']}** ({p.get('position', '?')}, {p.get('team', '?')}) — "
                f"{p.get('grade')} at {p.get('success_probability', '?')}%, "
                f"model rank #{p.get('rank', '?')}"
            )

    lines += ["", "## Big board vs. model — biggest disagreement"]
    dis = biggest_disagreement(big_boards, snaps)
    if dis is None:
        lines.append("_No overlap between owner big boards and the current model board._")
    else:
        side = "the board is way higher than the model" if dis["gap"] > 0 \
            else "the model is way higher than the board"
        lines += [
            f"- **{dis['name']}** ({dis['team']}, class of {dis['year']}): "
            f"owner board **#{dis['owner_rank']}** vs model **#{dis['model_rank']}** — "
            f"a {abs(dis['gap'])}-spot gap ({side}).",
            f"  - Take: {TAKE}",
        ]

    lines += ["", "## Stats stinger"]
    stinger = [f"**{board_n}** prospects on the board"]
    pct = verified_stat_pct()
    if pct is not None:
        stinger.append(f"**{pct}%** graded on verified live stats")
    lines.append("- " + " · ".join(stinger) + ". Roll credits.")

    return "\n".join(lines) + "\n"


def main():
    movers = load_json(os.path.join(TRAINING, "board_movers.json"))
    snaps = latest_snapshots()
    big_boards = load_json(os.path.join(TRAINING, "big_boards.json"))

    day = recap_date(movers)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"recap_{day}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_recap(day, movers, snaps, big_boards))
    print(f"Wrote {os.path.relpath(out_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

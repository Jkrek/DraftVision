#!/usr/bin/env python3
"""Join staged consensus-board ranks + all-star invites into
training_data/combine_outcomes.csv (v4 feature wave).

Adds:
  consensus_rank     — best available scout consensus rank (WideLeft multi-board
                       preferred, ESPN otherwise); blank when unranked
  consensus_covered  — 1 if the row's class year has consensus coverage
                       (unranked-in-covered-year is SIGNAL; uncovered is NaN)
  allstar_invite     — 1/0 within all-star coverage years; blank outside

The overall_pick / espn_grade columns in the ESPN file are OUTCOMES and are
deliberately not joined.
"""

import csv
import re
import sys

SRC      = "training_data/combine_outcomes.csv"
WIDELEFT = "training_data/staging/consensus_wideleft.csv"
ESPN     = "training_data/staging/consensus_espn.csv"
ALLSTAR  = "training_data/staging/allstar_invites.csv"


def norm(name: str) -> str:
    n = (name or "").lower().strip()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?$", "", n).strip()
    return re.sub(r"[^a-z ]", "", n)


def load_ranks(path, year_col="draft_year"):
    out, years = {}, set()
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                yr = int(float(r[year_col]))
                rk = int(float(r["consensus_rank"]))
            except (KeyError, TypeError, ValueError):
                continue
            years.add(yr)
            key = (norm(r.get("player")), yr)
            if key[0] and (key not in out or rk < out[key]):
                out[key] = rk
    return out, years


def main() -> int:
    wl, wl_years = load_ranks(WIDELEFT)
    es, es_years = load_ranks(ESPN)
    covered = wl_years | es_years

    invites, as_years = set(), set()
    with open(ALLSTAR) as f:
        for r in csv.DictReader(f):
            try:
                yr = int(float(r["year"]))
            except (TypeError, ValueError):
                continue
            as_years.add(yr)
            invites.add((norm(r.get("player")), yr))

    with open(SRC) as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    for c in ("consensus_rank", "consensus_covered", "allstar_invite"):
        if c not in fields:
            fields.append(c)

    matched = cov_rows = as_matched = 0
    for r in rows:
        yr = int(float(r["draft_year"]))
        key = (norm(r.get("name")), yr)
        if yr in covered:
            cov_rows += 1
            r["consensus_covered"] = 1
            rk = wl.get(key) or es.get(key)
            r["consensus_rank"] = rk or ""
            matched += bool(rk)
        else:
            r["consensus_covered"] = 0
            r["consensus_rank"] = ""
        if yr in as_years:
            r["allstar_invite"] = 1 if key in invites else 0
            as_matched += key in invites
        else:
            r["allstar_invite"] = ""

    with open(SRC, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"coverage years: {min(covered)}-{max(covered)} | rows in covered years: {cov_rows}")
    print(f"consensus-ranked rows: {matched} | allstar invites matched: {as_matched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

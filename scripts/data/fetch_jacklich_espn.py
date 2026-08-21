#!/usr/bin/env python
"""Fetch historical ESPN pre-draft prospect ranks/grades from
github.com/JackLich10/nfl-draft-data (frozen 2021-05-05, no license file).

Source file: nfl_draft_prospects.csv (raw.githubusercontent.com) — per-player
ESPN draft data back to 1967 with pre-draft overall rank (`ovr_rk`), position
rank (`pos_rk`) and scouting grade (`grade`), plus actual draft slot.

Output: training_data/staging/consensus_espn.csv
  draft_year, player, school, position, consensus_rank (= ESPN ovr_rk),
  pos_rank, espn_grade, overall_pick, n_boards (=1: single-outlet proxy)

Scope kept to draft classes 2000-2021 (our training window; the repo is
frozen at the 2021 class anyway).

License note: unlicensed scrape of ESPN data. Train on it privately; do NOT
commit/redistribute the raw dump. The staged CSV is a derived subset kept out
of version control.

Run: .venv/bin/python scripts/data/fetch_jacklich_espn.py
"""

import pandas as pd

from _common import cached_download, write_csv

RAW = "https://raw.githubusercontent.com/JackLich10/nfl-draft-data/main/nfl_draft_prospects.csv"
FIRST_CLASS, LAST_CLASS = 2000, 2021


def main() -> None:
    path = cached_download(RAW, "jacklich", "nfl_draft_prospects.csv")
    df = pd.read_csv(path, low_memory=False)
    df = df[(df.draft_year >= FIRST_CLASS) & (df.draft_year <= LAST_CLASS)].copy()

    out = pd.DataFrame({
        "draft_year": df["draft_year"].astype(int),
        "player": df["player_name"].astype(str).str.strip(),
        # `school` is the school name ("Ole Miss"); `school_name` is the MASCOT
        # ("Rebels") in this source — do not swap them.
        "school": df["school"].fillna(df["school_name"]).astype(str).str.strip(),
        "position": df["pos_abbr"].astype(str).str.strip(),
        "consensus_rank": pd.to_numeric(df["ovr_rk"], errors="coerce"),
        "pos_rank": pd.to_numeric(df["pos_rk"], errors="coerce"),
        "espn_grade": pd.to_numeric(df["grade"], errors="coerce"),
        "overall_pick": pd.to_numeric(df["overall"], errors="coerce"),
        "n_boards": 1,
    })
    # Keep rows that carry at least one pre-draft signal (rank or grade).
    out = out[out["consensus_rank"].notna() | out["espn_grade"].notna()]
    out = out[out["player"].str.len() > 1].reset_index(drop=True)

    cov = out.groupby("draft_year")["consensus_rank"].agg(["size", "count"])
    print("Per-year rows / with ovr_rk:")
    print(cov.to_string())
    write_csv(out, "consensus_espn.csv")


if __name__ == "__main__":
    main()

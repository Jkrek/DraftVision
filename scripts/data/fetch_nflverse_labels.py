#!/usr/bin/env python
"""Stage nflverse success-label upgrade materials: OTC contracts + snap counts.

Sources (nflverse-data GitHub releases, openly published, MIT-tooled):
  contracts:   https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.csv.gz
               (nflreadr load_contracts(); OverTheCap data — dictionary:
               https://nflreadr.nflverse.com/articles/dictionary_contracts.html)
  snap counts: https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv
               (nflreadr load_snap_counts(); PFR-sourced, 2012+ — the 2012
               file is header-only upstream, so effective coverage is 2013+)

Outputs
-------
training_data/staging/contracts.csv — one row per contract:
  player, otc_id, position, team, college, draft_year, draft_round,
  draft_overall, year_signed, years, value, apy, guaranteed, apy_cap_pct,
  inflated_apy, inflated_guaranteed, contract_index, is_second_contract

  is_second_contract logic (documented, deliberately simple):
    * contracts are ordered per player (otc_id) by year_signed (ties broken
      by larger total value first — extensions logged the same year as a
      restructure count as the "real" deal);
    * contract_index = 1-based position in that order;
    * is_second_contract = (contract_index == 2). For drafted players index 1
      is the rookie deal, so index 2 is the classic "second contract" the
      success-label upgrade wants. Fifth-year options are part of the rookie
      deal in OTC data (not separate rows). For UDFAs index 1 is the UDFA
      deal, so index 2 is likewise their first earned veteran contract.
    * The label build (next wave) should combine is_second_contract with an
      APY threshold (e.g. inflated_apy percentile by position) — the flag
      alone includes minimum-salary re-ups.

training_data/staging/snap_counts.csv — one row per player-season (REG games):
  player, pfr_player_id, season, position, teams, games,
  offense_snaps, offense_pct_mean, defense_snaps, defense_pct_mean,
  st_snaps, st_pct_mean
  (offense_pct/defense_pct are per-game shares of team snaps; the mean is
  across games the player appeared in, NOT across the full team schedule.)

Run: .venv/bin/python scripts/data/fetch_nflverse_labels.py
"""

import gzip
import shutil
import os

import pandas as pd

from _common import cache_dir, cached_download, write_csv

CONTRACTS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
                 "contracts/historical_contracts.csv.gz")
SNAPS_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
             "snap_counts/snap_counts_{season}.csv")
SNAP_SEASONS = range(2012, 2026)


def stage_contracts() -> None:
    gz = cached_download(CONTRACTS_URL, "nflverse", "historical_contracts.csv.gz")
    csv_path = os.path.join(cache_dir("nflverse"), "historical_contracts.csv")
    if not os.path.exists(csv_path) or os.path.getmtime(csv_path) < os.path.getmtime(gz):
        with gzip.open(gz, "rb") as fin, open(csv_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)

    df = pd.read_csv(csv_path, low_memory=False)
    for col in ("draft_year", "draft_round", "draft_overall", "year_signed", "years"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["otc_id", "year_signed", "value"],
                        ascending=[True, True, False])
    df["contract_index"] = df.groupby("otc_id").cumcount() + 1
    df["is_second_contract"] = df["contract_index"] == 2

    out = df[["player", "otc_id", "position", "team", "college", "draft_year",
              "draft_round", "draft_overall", "year_signed", "years", "value",
              "apy", "guaranteed", "apy_cap_pct", "inflated_apy",
              "inflated_guaranteed", "contract_index", "is_second_contract"]]
    n2 = int(out.is_second_contract.sum())
    print(f"  contracts: {len(out)} rows, {out.otc_id.nunique()} players, "
          f"{n2} second contracts, draft_year known for "
          f"{out.draft_year.notna().mean():.0%}")
    write_csv(out, "contracts.csv")


def stage_snap_counts() -> None:
    frames = []
    for season in SNAP_SEASONS:
        path = cached_download(SNAPS_URL.format(season=season), "nflverse",
                               f"snap_counts_{season}.csv", min_bytes=100)
        df = pd.read_csv(path)
        if df.empty:  # 2012 is header-only upstream
            print(f"  snap_counts {season}: empty upstream, skipped")
            continue
        frames.append(df[df.game_type == "REG"])
    snaps = pd.concat(frames, ignore_index=True)

    grp = snaps.groupby(["pfr_player_id", "season"])
    out = grp.agg(
        player=("player", "first"),
        position=("position", "first"),
        teams=("team", lambda s: "/".join(sorted(set(s)))),
        games=("game_id", "nunique"),
        offense_snaps=("offense_snaps", "sum"),
        offense_pct_mean=("offense_pct", "mean"),
        defense_snaps=("defense_snaps", "sum"),
        defense_pct_mean=("defense_pct", "mean"),
        st_snaps=("st_snaps", "sum"),
        st_pct_mean=("st_pct", "mean"),
    ).reset_index()
    for c in ("offense_pct_mean", "defense_pct_mean", "st_pct_mean"):
        out[c] = out[c].round(3)
    out = out[["player", "pfr_player_id", "season", "position", "teams", "games",
               "offense_snaps", "offense_pct_mean", "defense_snaps",
               "defense_pct_mean", "st_snaps", "st_pct_mean"]]
    print(f"  snap_counts: {len(out)} player-seasons "
          f"({out.season.min()}-{out.season.max()}), "
          f"{out.pfr_player_id.nunique()} distinct players")
    write_csv(out.sort_values(["season", "player"]), "snap_counts.csv")


if __name__ == "__main__":
    stage_contracts()
    stage_snap_counts()

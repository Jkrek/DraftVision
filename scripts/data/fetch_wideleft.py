#!/usr/bin/env python
"""Fetch Arif Hasan's Wide Left consensus big boards (2024-2026 draft classes).

Source posts (each links a free, publicly shared Google Sheet — "this piece is
free to everyone, as is the data"):
  2024: https://wideleft.football/p/2024-consensus-big-board-the-top
  2025: https://www.wideleft.football/p/2025-consensus-big-board-the-top
  2026: https://wideleft.football/p/2026-nfl-draft-consensus-big-board

Sheet IDs below were extracted from the "Get the data" links embedded in those
posts (verified 2026-08-21). Sheets are fetched via the public CSV export URL.

Output: training_data/staging/consensus_wideleft.csv
  draft_year, player, school, position, consensus_rank, pos_rank, n_boards,
  source_url

n_boards = number of analyst boards aggregated per cycle, as stated in each
post (101 for 2024, 112 for 2025, 134 for 2026).

License note: no explicit reuse license — internal feature derivation only,
do NOT republish the board. Credit Arif Hasan / Wide Left.

Run: .venv/bin/python scripts/data/fetch_wideleft.py
"""

import pandas as pd

from _common import cached_download, write_csv

BOARDS = {
    2024: {
        "sheet_id": "1u_7bYeFLyPGldL6OqvyEXICH4LCvNZb_iqc2Z3740mo",
        "n_boards": 101,
        "source_url": "https://wideleft.football/p/2024-consensus-big-board-the-top",
    },
    2025: {
        "sheet_id": "1IUxTL9PXAmkasscUiGVYovtdKo7tawIuzXdMfDzmqI4",
        "n_boards": 112,
        "source_url": "https://www.wideleft.football/p/2025-consensus-big-board-the-top",
    },
    2026: {
        "sheet_id": "1kMMdFfdPhcIlmSRFWnKVzx3IGm5VODx5oqJRraGms5E",
        "n_boards": 134,
        "source_url": "https://wideleft.football/p/2026-nfl-draft-consensus-big-board",
    },
}


def main() -> None:
    frames = []
    for year, meta in sorted(BOARDS.items()):
        url = f"https://docs.google.com/spreadsheets/d/{meta['sheet_id']}/export?format=csv"
        path = cached_download(url, "wideleft", f"wideleft_{year}.csv")
        # Row 0-2 are title/credit lines; row 3 is the header.
        df = pd.read_csv(path, skiprows=3)
        df.columns = [c.strip().lower() for c in df.columns]
        # 2024 sheet titles the player column "PLAYER", later years "Player".
        df = df.rename(columns={"ovr": "consensus_rank", "pos rk": "pos_rank"})
        out = pd.DataFrame({
            "draft_year": year,
            "player": df["player"].astype(str).str.strip(),
            "school": df["school"].astype(str).str.strip(),
            "position": df["position"].astype(str).str.strip(),
            "consensus_rank": pd.to_numeric(df["consensus_rank"], errors="coerce"),
            "pos_rank": pd.to_numeric(df["pos_rank"], errors="coerce"),
            "n_boards": meta["n_boards"],
            "source_url": meta["source_url"],
        })
        out = out.dropna(subset=["consensus_rank"])
        out = out[out["player"].str.len() > 1]
        print(f"  {year}: {len(out)} ranked players (boards aggregated: {meta['n_boards']})")
        frames.append(out)

    allb = pd.concat(frames, ignore_index=True)
    allb["consensus_rank"] = allb["consensus_rank"].astype(int)
    write_csv(allb, "consensus_wideleft.csv")


if __name__ == "__main__":
    main()

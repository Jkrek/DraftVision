#!/usr/bin/env python
"""Fetch Senior Bowl and East-West Shrine rosters from per-year Wikipedia
game articles (CC BY-SA 4.0 — attribution: Wikipedia contributors; the roster
lists themselves are factual).

Method: MediaWiki API (action=parse&prop=wikitext) per year article; roster
wikitables are detected by header (must contain a Player/Name column and a
College/School column) and parsed cell-wise. Player names inside wikilinks use
the link target ("[[Danario Alexander|Alexander, Danario]]" -> "Danario
Alexander"); plain "Last, First" cells are flipped.

Quality gate: a year is kept only if >= MIN_PLAYERS players parse (full
rosters run ~100); years whose articles are missing, are redirects to the
generic franchise page, or list only partial/notable players are SKIPPED and
reported — better no data than a biased partial invite list.

Output: training_data/staging/allstar_invites.csv
  year (game year == draft-class year), player, school, position, game
  (senior_bowl | shrine), source_url

Run: .venv/bin/python scripts/data/fetch_allstar_rosters.py
"""

import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from _common import SSL_CTX, cache_dir, write_csv

API = "https://en.wikipedia.org/w/api.php"
UA = "DraftVisionResearch/1.0 (contact: repo owner; data staging for draft model)"
MIN_PLAYERS = 60

YEARS = range(2000, 2027)
TITLES = {
    "senior_bowl": ["{y} Senior Bowl"],
    # Renamed "Game" -> "Bowl" for the 2020 edition.
    "shrine": ["{y} East–West Shrine Bowl", "{y} East–West Shrine Game"],
}

POS_RE = re.compile(
    r"^(QB|RB|FB|HB|WR|TE|OT|OG|OC|OL|C|G|T|DE|DT|NT|DL|EDGE|LB|ILB|OLB|MLB|"
    r"CB|S|FS|SS|DB|K|P|LS|PK|KR|RS|ATH)(/[A-Z]{1,4})?$"
)


def _api_wikitext(title: str, year: int, game: str):
    """Cached fetch of an article's wikitext; returns (wikitext, resolved_title) or None."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title)
    fname = f"{game}_{year}_{slug}.json"
    path = f"{cache_dir('allstar')}/{fname}"
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        q = urllib.parse.urlencode({
            "action": "parse", "page": title, "prop": "wikitext",
            "format": "json", "formatversion": 2, "redirects": 1,
        })
        req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
        data = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as resp:
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise
        if data is None:
            raise RuntimeError(f"Wikipedia rate-limited after retries: {title}")
        with open(path, "w") as fh:
            json.dump(data, fh)
        time.sleep(1.5)
    if "parse" not in data:
        return None
    resolved = data["parse"].get("title", title)
    # Redirected to the generic franchise article (no per-year roster) -> skip.
    if not re.match(r"^\d{4} ", resolved):
        return None
    return data["parse"]["wikitext"], resolved


_SORTNAME_RE = re.compile(r"\{\{\s*sortname\s*\|([^|}]+)\|([^|}]+)(?:\|[^}]*)?\}\}", re.I)


def _expand_sortname(wikitext: str) -> str:
    """{{sortname|Trey|Adams}} / {{sortname|F|L|dab-or-nolink}} -> [[F L]]."""
    return _SORTNAME_RE.sub(lambda m: f"[[{m.group(1).strip()} {m.group(2).strip()}]]", wikitext)


def _strip_markup(cell: str) -> str:
    cell = re.sub(r"<ref[^>]*/>", "", cell)
    cell = re.sub(r"<ref[^>]*>.*?</ref>", "", cell, flags=re.S)
    cell = re.sub(r"\{\{[^{}]*\}\}", "", cell)
    cell = re.sub(r"'''?", "", cell)
    cell = re.sub(r"<[^>]+>", "", cell)
    return cell.strip()


def _link_parts(cell: str):
    """Return (target, display) of the first wikilink, else (None, cleaned text)."""
    m = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", cell)
    if not m:
        return None, _strip_markup(cell)
    target = m.group(1).strip()
    display = (m.group(2) or m.group(1)).strip()
    return target, display


def _player_name(cell: str) -> str:
    target, display = _link_parts(cell)
    if target:
        name = re.sub(r"\s*\([^)]*\)$", "", target)  # drop "(American football)"
    else:
        name = display
    name = _strip_markup(name)
    if "," in name:  # "Alexander, Danario" -> "Danario Alexander"
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    return re.sub(r"\s+", " ", name).strip()


def _school_name(cell: str) -> str:
    target, display = _link_parts(cell)
    school = display if display else (target or "")
    school = _strip_markup(school)
    school = re.sub(r"\s*\((?:D-?I+|DII|DIII|FCS|NAIA)[^)]*\)$", "", school, flags=re.I)
    return school.strip()


def _split_cells(line: str, header: bool) -> list:
    body = line[1:]
    return re.split(r"\|\||!!", body) if not header else re.split(r"\|\||!!", body)


def _parse_tables(wikitext: str):
    """Yield lists of row-dicts for each wikitable with Player+College headers."""
    for tbl in re.findall(r"\{\|.*?\n\|\}", wikitext, flags=re.S):
        lines = tbl.split("\n")
        header_cells, rows, current = [], [], []
        for ln in lines:
            ln = ln.strip()
            if ln.startswith("{|") or ln.startswith("|}"):
                continue
            if ln.startswith("|-"):
                if current:
                    rows.append(current)
                current = []
            elif ln.startswith("!"):
                header_cells.extend(_strip_markup(c).lower()
                                    for c in re.split(r"\|\||!!", ln[1:]))
            elif ln.startswith("|"):
                current.extend(re.split(r"\|\|", ln[1:]))
        if current:
            rows.append(current)

        def col_idx(*names):
            for i, h in enumerate(header_cells):
                h = h.split("|")[-1].strip()  # drop cell-attribute prefix
                if any(h.startswith(n) for n in names):
                    return i
            return None

        p_i = col_idx("player", "name")
        s_i = col_idx("college", "school")
        pos_i = col_idx("position", "pos")
        if p_i is None or s_i is None:
            continue
        parsed = []
        for cells in rows:
            if len(cells) <= max(p_i, s_i):
                continue
            player = _player_name(cells[p_i])
            school = _school_name(cells[s_i])
            pos = ""
            if pos_i is not None and len(cells) > pos_i:
                pos = _strip_markup(re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]",
                                           r"\1", cells[pos_i])).upper()
            if not player or not school or len(player) < 4 or " " not in player:
                continue
            if pos and not POS_RE.match(pos):
                pos = pos if len(pos) <= 4 else ""
            parsed.append({"player": player, "school": school, "position": pos})
        if len(parsed) >= 10:  # ignore scoring-summary etc. tables
            yield parsed


def main() -> None:
    kept, skipped = [], []
    for game, patterns in TITLES.items():
        for y in YEARS:
            got = None
            for pat in patterns:
                res = _api_wikitext(pat.format(y=y), y, game)
                if res:
                    got = res
                    break
            if not got:
                skipped.append((game, y, "no per-year article"))
                continue
            wikitext, resolved = got
            players = [r for tbl in _parse_tables(_expand_sortname(wikitext)) for r in tbl]
            # de-dup within year (players occasionally appear in two tables)
            seen, uniq = set(), []
            for r in players:
                key = (r["player"].lower(), r["school"].lower())
                if key not in seen:
                    seen.add(key)
                    uniq.append(r)
            if len(uniq) < MIN_PLAYERS:
                skipped.append((game, y, f"only {len(uniq)} players parsed (partial list)"))
                continue
            url = "https://en.wikipedia.org/wiki/" + resolved.replace(" ", "_")
            for r in uniq:
                r.update({"year": y, "game": game, "source_url": url})
            kept.extend(uniq)
            print(f"  {game} {y}: {len(uniq)} players  ({url})")

    df = pd.DataFrame(kept, columns=["year", "player", "school", "position", "game", "source_url"])
    df = df.sort_values(["game", "year", "player"]).reset_index(drop=True)
    write_csv(df, "allstar_invites.csv")
    if skipped:
        print("\nSkipped years (no reliable full roster):")
        for game, y, why in skipped:
            print(f"  {game} {y}: {why}")


if __name__ == "__main__":
    main()

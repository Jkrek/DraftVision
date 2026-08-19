"""College tier classification — shared by the Flask app and the cache
builder (which pre-filters sub-FBS teams locally to avoid thousands of
pointless roster fetches). Extracted verbatim from XGBOost.py."""
from typing import Dict


# NFL franchise full names — a record stored with one of these teams is an
# active pro, not a college program. Matched exactly, never by substring:
# mascot keywords ("Bears", "Cardinals") collide with Baylor, Louisville,
# Brown, Columbia and dozens of other college programs.
NFL_FRANCHISE_NAMES = {
    "arizona cardinals", "atlanta falcons", "baltimore ravens", "buffalo bills",
    "carolina panthers", "chicago bears", "cincinnati bengals", "cleveland browns",
    "dallas cowboys", "denver broncos", "detroit lions", "green bay packers",
    "houston texans", "indianapolis colts", "jacksonville jaguars",
    "kansas city chiefs", "las vegas raiders", "los angeles chargers",
    "los angeles rams", "miami dolphins", "minnesota vikings",
    "new england patriots", "new orleans saints", "new york giants",
    "new york jets", "philadelphia eagles", "pittsburgh steelers",
    "san francisco 49ers", "seattle seahawks", "tampa bay buccaneers",
    "tennessee titans", "washington commanders",
}


def _normalize_team(team: str) -> str:
    t = (team or "").lower().strip()
    for ch in ("’", "'", "."):
        t = t.replace(ch, "")
    return t.replace("é", "e")  # San José State


def is_nfl_franchise(team: str) -> bool:
    return _normalize_team(team) in NFL_FRANCHISE_NAMES


# School → tier, matched against ESPN display names ("Michigan State Spartans")
# by longest prefix on a word boundary, so "michigan state" wins over "michigan"
# and mascots never participate in matching.
_TIER_SCHOOLS = {
    1: {"alabama", "ohio state", "georgia", "clemson", "lsu", "michigan"},
    2: {"texas", "oklahoma", "florida", "penn state", "notre dame", "florida state",
        "tennessee", "texas a&m", "usc", "oregon", "miami", "auburn", "washington"},
    3: {"north carolina", "virginia tech", "pittsburgh", "wisconsin", "iowa",
        "michigan state", "nebraska", "oklahoma state", "baylor", "tcu", "arkansas",
        "ole miss", "mississippi state", "south carolina", "stanford", "utah",
        "arizona state", "colorado", "georgia tech"},
    4: {"west virginia", "kansas state", "iowa state", "texas tech", "kentucky",
        "vanderbilt", "missouri", "arizona", "cal", "california", "oregon state",
        "washington state", "indiana", "purdue", "illinois", "minnesota",
        "maryland", "rutgers", "louisville", "virginia", "nc state", "duke",
        "wake forest", "syracuse", "boston college", "cincinnati", "ucf"},
    5: {"ucla", "northwestern", "navy", "army", "air force", "liberty", "byu",
        "western kentucky", "louisiana tech"},
    6: {"memphis", "houston", "smu", "tulane", "east carolina", "south florida",
        "temple", "connecticut", "uconn", "tulsa", "rice", "utep", "uab"},
    7: {"boise state", "fresno state", "hawaii", "san diego state", "wyoming",
        "utah state", "nevada", "colorado state", "new mexico", "san jose state"},
    8: {"appalachian state", "app state", "coastal carolina", "marshall", "utsa",
        "troy", "louisiana", "james madison", "buffalo", "kent state", "ohio",
        "miami (oh)", "western michigan", "central michigan", "eastern michigan",
        "northern illinois", "ball state", "toledo",
        # FBS programs previously absent from the tables entirely — they fell
        # to tier 10 and were silently excluded from the board
        "north texas", "old dominion", "charlotte", "middle tennessee",
        "southern miss", "south alabama", "akron", "bowling green", "umass",
        "jacksonville state", "sam houston", "kennesaw state", "missouri state",
        "arkansas state", "georgia southern", "georgia state", "texas state",
        "florida atlantic", "fiu", "florida international", "unlv", "delaware"},
    9: {"north dakota state", "montana", "south dakota state", "furman",
        "villanova", "richmond", "sacramento state",
        "central arkansas", "cal poly"},
    # Prefix-collision guards: sub-FBS schools whose names START WITH a power
    # school's name inherit its tier without these ("north carolina a&t" ->
    # "north carolina" tier 3, "georgia southern" -> "georgia" tier 1).
    # Longest-key-first matching makes these explicit entries win; tier 10
    # keeps them off the FBS board.
    10: {"north carolina a&t", "north carolina central", "south carolina state",
         "texas southern", "texas a&m commerce", "texas a&m kingsville",
         "florida a&m", "alabama a&m", "alabama state", "houston christian",
         "tennessee state", "tennessee tech", "tennessee martin",
         "illinois state", "indiana state", "virginia state", "virginia union",
         "arkansas pine bluff", "utah tech", "minnesota state",
         "mississippi valley state", "delaware state"},
}

_SCHOOL_TIERS: Dict[str, int] = {
    school: tier for tier, schools in _TIER_SCHOOLS.items() for school in schools
}
_SCHOOL_KEYS = sorted(_SCHOOL_TIERS, key=len, reverse=True)


def classify_college_tier(team: str) -> int:
    """10-tier conference classification. 1=SEC/OSU elite, 10=FCS lower/unknown.

    Exact NFL franchise names → tier 1 (an NFL record implies a major-program
    background). School names match by longest word-boundary prefix.
    """
    t = _normalize_team(team)
    if not t:
        return 10
    if t in NFL_FRANCHISE_NAMES:
        return 1
    for key in _SCHOOL_KEYS:
        if t == key or t.startswith(key + " "):
            return _SCHOOL_TIERS[key]
    return 10

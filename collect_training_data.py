"""
Collect real NFL draft + combine + career outcome data from ESPN APIs.

Outputs (saved to training_data/):
  combine_outcomes.csv  — one row per historical player with:
    position, conference_tier, combine_speed_score, production_score,
    is_award_winner, draft_round, nfl_success

Run once:
  python collect_training_data.py

XGBOost.py reads training_data/combine_outcomes.csv on startup if present.
"""

import csv
import json
import os
import time
import requests

OUTPUT_DIR = "training_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combine_outcomes.csv")

DRAFT_YEARS = list(range(2015, 2023))  # 2015–2022: enough career data by now

# ── ESPN endpoints ────────────────────────────────────────────────────────────
ESPN_DRAFT_URL  = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/draft/{year}"
ESPN_NFL_ATHLETE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/athletes/{id}"
ESPN_CORE_ATHLETE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/athletes/{id}"

# ── Conference tier map (1=best, 10=FCS lower) ───────────────────────────────
TIER1  = {"alabama","ohio state","georgia","clemson","lsu","michigan"}
TIER2  = {"texas","oklahoma","florida","penn state","notre dame","florida state",
          "tennessee","texas a&m","usc","oregon","miami","auburn","washington"}
TIER3  = {"north carolina","virginia tech","pittsburgh","wisconsin","iowa",
          "michigan state","nebraska","oklahoma state","baylor","tcu","arkansas",
          "ole miss","mississippi state","south carolina","stanford","utah",
          "arizona state","colorado","georgia tech","florida int","fiu"}
TIER4  = {"west virginia","kansas state","iowa state","texas tech","kentucky",
          "vanderbilt","missouri","arizona","cal","oregon state","washington state",
          "indiana","purdue","illinois","minnesota","maryland","rutgers",
          "louisville","virginia","nc state","duke","wake forest","syracuse",
          "boston college","cincinnati","ucf"}
TIER5  = {"ucla","northwestern","navy","army","air force","liberty","byu",
          "western kentucky","louisiana tech"}
TIER6  = {"memphis","houston","smu","tulane","east carolina","south florida",
          "temple","connecticut","tulsa","rice","utep"}
TIER7  = {"boise state","fresno state","hawaii","san diego state","wyoming",
          "utah state","nevada","colorado state","new mexico","san jose state",
          "air force"}
TIER8  = {"appalachian state","coastal carolina","marshall","utsa","troy",
          "louisiana","james madison","buffalo","kent state","ohio",
          "western michigan","central michigan","eastern michigan",
          "northern illinois","ball state","miami ohio","toledo"}
TIER9  = {"north dakota state","montana","south dakota state","furman",
          "villanova","richmond","delaware","stony brook","sacramento state",
          "central arkansas","southern utah","weber state"}

def classify_tier(school: str) -> int:
    s = (school or "").lower().strip()
    for kw in TIER1:
        if kw in s: return 1
    for kw in TIER2:
        if kw in s: return 2
    for kw in TIER3:
        if kw in s: return 3
    for kw in TIER4:
        if kw in s: return 4
    for kw in TIER5:
        if kw in s: return 5
    for kw in TIER6:
        if kw in s: return 6
    for kw in TIER7:
        if kw in s: return 7
    for kw in TIER8:
        if kw in s: return 8
    for kw in TIER9:
        if kw in s: return 9
    return 10

def forty_to_speed(position: str, forty: float) -> float:
    if not forty or forty <= 0:
        return 50.0
    p = (position or "").upper()
    benchmarks = {
        "QB":(4.30,5.10),"RB":(4.20,4.80),"WR":(4.20,4.70),
        "TE":(4.40,5.00),"CB":(4.20,4.65),"S":(4.30,4.75),
        "DB":(4.25,4.70),"LB":(4.35,4.85),"DL":(4.50,5.30),
        "DE":(4.45,5.10),"DT":(4.55,5.35),"EDGE":(4.45,5.10),
        "OL":(4.70,5.55),"OT":(4.75,5.60),"OG":(4.80,5.55),"C":(4.85,5.60),
    }
    elite_t, poor_t = benchmarks.get(p, (4.35, 5.00))
    score = (poor_t - forty) / (poor_t - elite_t) * 100.0
    return float(max(0.0, min(100.0, score)))

def get(url, params=None, retries=2):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=8)
            if r.ok:
                return r.json()
        except Exception as e:
            if attempt == retries - 1:
                print(f"  GET failed: {url} — {e}")
        time.sleep(0.5)
    return {}

def fetch_draft_class(year: int) -> list:
    data = get(ESPN_DRAFT_URL.format(year=year))
    picks = []
    for rnd in data.get("rounds", []):
        rnum = rnd.get("number", 0)
        for pick in rnd.get("picks", []):
            ath = pick.get("athlete") or {}
            college_obj = ath.get("college") or {}
            picks.append({
                "year": year,
                "round": rnum,
                "overall": pick.get("overall", 0),
                "name": ath.get("displayName", ""),
                "espn_id": str(ath.get("id", "")),
                "position": (ath.get("position") or {}).get("abbreviation", ""),
                "college": college_obj.get("displayName", ""),
            })
    return picks

def fetch_combine_data(espn_id: str) -> dict:
    """Fetch 40-yard dash and vertical from ESPN core athlete endpoint."""
    if not espn_id:
        return {}
    data = get(ESPN_CORE_ATHLETE.format(id=espn_id))
    draft_info = data.get("draft") or {}
    return {
        "forty":    float(draft_info.get("combined40yd") or 0),
        "vertical": float(draft_info.get("combineVert") or 0),
        "bench":    int(draft_info.get("combineBench") or 0),
    }

def fetch_nfl_career(espn_id: str) -> dict:
    """Return years_experience and pro_bowls from ESPN NFL athlete endpoint."""
    if not espn_id:
        return {"experience": 0, "pro_bowls": 0, "active": False}
    data = get(ESPN_NFL_ATHLETE.format(id=espn_id))
    ath = data.get("athlete") or data
    exp = int((ath.get("experience") or {}).get("years", 0) if isinstance(ath.get("experience"), dict) else ath.get("experience") or 0)
    honors = ath.get("honors") or []
    pro_bowls = sum(1 for h in honors if "pro bowl" in str(h.get("displayName","")).lower())
    active = str(ath.get("status", {}).get("type", "")).lower() == "active" if isinstance(ath.get("status"), dict) else False
    return {"experience": exp, "pro_bowls": pro_bowls, "active": active}

def nfl_success_label(career: dict, draft_round: int) -> int:
    """
    1 = NFL success (starter/star), 0 = bust/journeyman.
    Success criteria: Pro Bowl OR 5+ years experience OR (4+ years AND round 1-2).
    """
    if career["pro_bowls"] >= 1:
        return 1
    if career["experience"] >= 5:
        return 1
    if career["experience"] >= 4 and draft_round <= 2:
        return 1
    return 0

def draft_grade_label(draft_round: int) -> int:
    """0=Top50(R1-2), 1=Day2(R3-4), 2=LateRound(R5-7), 3=Undrafted"""
    if draft_round <= 2: return 0
    if draft_round <= 4: return 1
    if draft_round <= 7: return 2
    return 3

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    total_picks = 0
    errors = 0

    for year in DRAFT_YEARS:
        print(f"\n── {year} draft class ──")
        picks = fetch_draft_class(year)
        print(f"  {len(picks)} picks found")
        total_picks += len(picks)

        for i, pick in enumerate(picks):
            if not pick["name"] or not pick["espn_id"]:
                continue

            espn_id  = pick["espn_id"]
            position = pick["position"]
            college  = pick["college"]
            rnd      = pick["round"]

            combine  = fetch_combine_data(espn_id)
            career   = fetch_nfl_career(espn_id)
            tier     = classify_tier(college)
            speed    = forty_to_speed(position, combine.get("forty", 0))

            rows.append({
                "name":              pick["name"],
                "year":              year,
                "position":          position,
                "college":           college,
                "conference_tier":   tier,
                "combine_speed_score": round(speed, 1),
                "combine_forty":     combine.get("forty", 0),
                "combine_vertical":  combine.get("vertical", 0),
                "draft_round":       rnd,
                "draft_grade":       draft_grade_label(rnd),
                "nfl_success":       nfl_success_label(career, rnd),
                "experience":        career["experience"],
                "pro_bowls":         career["pro_bowls"],
            })

            if (i + 1) % 20 == 0:
                print(f"  processed {i+1}/{len(picks)}")
            time.sleep(0.15)  # be polite to ESPN API

    print(f"\n── Writing {len(rows)} rows to {OUTPUT_FILE} ──")
    if not rows:
        print("No data collected. Check ESPN API access.")
        return

    fieldnames = ["name","year","position","college","conference_tier",
                  "combine_speed_score","combine_forty","combine_vertical",
                  "draft_round","draft_grade","nfl_success","experience","pro_bowls"]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(r["nfl_success"] for r in rows)
    print(f"Total: {len(rows)} players | Successes: {successes} ({100*successes//len(rows)}%) | Errors: {errors}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based (heuristic) prospect scoring shared by the app and the trainer.

Single source of truth for the pre-ML heuristics:
  - XGBOost.py imports these for its emergency rule-based fallback path.
  - scripts/train_models.py imports these to evaluate the RULE-BASED BASELINE
    against the trained models on the held-out test years.

Keeping one implementation prevents drift between "the baseline we report"
and "the fallback we serve". This module must stay import-side-effect free
(stdlib only, no Flask, no model loading).
"""

from __future__ import annotations


def success_prob_from_college_profile(
    production_score: float,
    conference_tier: float,
    combine_speed: float,
    is_award_winner: int = 0,
    is_all_american: int = 0,
) -> float:
    """Estimate P(NFL success) purely from college profile — NO draft round.

    Rule-based fallback / baseline, uncapped at the top so elite profiles are
    not squashed against a 0.92 wall. Accolade boosts are fine here (display /
    fallback-only path — accolades are not ML features)."""
    prod_factor  = production_score / 100.0
    tier_factor  = max(0.0, (11.0 - conference_tier) / 10.0)  # 1.0=T1 elite, 0.1=T10 FCS
    speed_factor = combine_speed / 100.0
    award_boost  = 0.08 if is_award_winner else 0.0
    aa_boost     = 0.05 if is_all_american else 0.0
    base = prod_factor * 0.45 + tier_factor * 0.30 + speed_factor * 0.25
    return float(max(0.03, base + award_boost + aa_boost))


def draft_grade_from_profile(
    production_score: float,
    conference_tier: float,
    combine_speed: float,
    is_award_winner: int = 0,
    is_all_american: int = 0,
) -> int:
    """Deterministic draft-grade class (0=Top50, 1=Day2, 2=LateRound, 3=UDFA)
    from the college profile. Rule-based fallback / baseline."""
    score = (
        production_score * 0.40
        + max(0.0, (11.0 - conference_tier) / 10.0) * 100 * 0.30
        + combine_speed * 0.20
        + (is_award_winner * 8 + is_all_american * 5)
    )
    if score >= 78: return 0  # Top 50 (R1-2)
    if score >= 60: return 1  # Day 2 (R3-4)
    if score >= 42: return 2  # Late Round (R5-7)
    return 3                   # UDFA

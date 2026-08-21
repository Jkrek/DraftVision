# Football Analytics Research Roadmap

*Last updated: August 2026*

This document is a master checklist for building a best-in-class CFB/NFL
analytics and prospect evaluation database.

------------------------------------------------------------------------

# Progress Tracker

  Area                           Status   Notes
  ------------------------------ -------- -------
  Core play-by-play              ☐        
  Recruiting                     ☐        
  Transfer portal                ☐        
  Combine                        ☐        
  Tracking data                  ☐        
  Injury history                 ☐        
  Coaching history               ☐        
  Character/off-field events     ☐        
  Advanced feature engineering   ☐        
  Computer vision                ☐        

------------------------------------------------------------------------

# 1. Core Datasets

## Must Have

-   [ ] cfbfastR
-   [ ] nflverse / nflfastR
-   [ ] NFL Big Data Bowl tracking
-   [ ] CollegeFootballData API
-   [ ] Pro Football Reference
-   [ ] Sports Reference

### Capture

-   [ ] Play-by-play
-   [ ] EPA
-   [ ] WPA
-   [ ] Drives
-   [ ] Personnel
-   [ ] Formation
-   [ ] Situational football
-   [ ] Betting lines
-   [ ] Team efficiency
-   [ ] Player participation

------------------------------------------------------------------------

# 2. Recruiting

Collect: - \[ \] Composite stars - \[ \] National rank - \[ \] Position
rank - \[ \] State rank - \[ \] Offer list - \[ \] Camp ratings - \[ \]
Elite 11 - \[ \] Multi-sport athlete - \[ \] Recruiting class ranking -
\[ \] Recruiting service agreement/disagreement

Engineered features: - \[ \] Recruiting pedigree - \[ \] Recruiting
overperformance - \[ \] Under-recruited index

------------------------------------------------------------------------

# 3. Transfer Portal

Collect: - \[ \] Transfer count - \[ \] Origin school - \[ \]
Destination - \[ \] Conference upgrade/downgrade - \[ \] Timing - \[ \]
Years remaining - \[ \] Immediate eligibility - \[ \] Coach followed -
\[ \] NIL context (public only)

Features: - \[ \] Transfer difficulty - \[ \] Transfer success - \[ \]
Portal trajectory

------------------------------------------------------------------------

# 4. Snap & Usage

-   [ ] Offensive snap %
-   [ ] Defensive snap %
-   [ ] Special teams snaps
-   [ ] Red-zone usage
-   [ ] Third-down usage
-   [ ] Target share
-   [ ] Route participation
-   [ ] Pass-block snaps
-   [ ] Run-block snaps

Features: - \[ \] Opportunity index - \[ \] Breakout age - \[ \]
Breakout season

------------------------------------------------------------------------

# 5. Injuries

Track verified public information only: - \[ \] Games missed - \[ \]
Injury type - \[ \] Surgery - \[ \] Repeat injuries - \[ \]
Concussions - \[ \] Recovery timeline

------------------------------------------------------------------------

# 6. Coaching & Scheme

-   [ ] Head coach
-   [ ] OC/DC
-   [ ] Position coach
-   [ ] NFL coaching experience
-   [ ] Scheme
-   [ ] Tempo
-   [ ] Motion rate

Features: - \[ \] Coaching tree - \[ \] Scheme continuity - \[ \]
Development environment

------------------------------------------------------------------------

# 7. Offensive Line Context

-   [ ] Returning starters
-   [ ] Career starts
-   [ ] Sack rate
-   [ ] Pressure rate
-   [ ] Experience
-   [ ] Transfer additions

------------------------------------------------------------------------

# 8. Opponent Quality

-   [ ] SOS
-   [ ] Position-specific SOS
-   [ ] Future NFL defenders faced
-   [ ] Top-25 games
-   [ ] Ranked opponents
-   [ ] Conference strength

------------------------------------------------------------------------

# 9. Pressure Metrics

-   [ ] Blitz %
-   [ ] Pressure %
-   [ ] Time to pressure
-   [ ] Clean pocket %
-   [ ] Double-team rate
-   [ ] Throw under pressure

------------------------------------------------------------------------

# 10. Tracking / Computer Vision

-   [ ] Separation
-   [ ] Route tree
-   [ ] Release package
-   [ ] Motion
-   [ ] Alignment
-   [ ] QB footwork
-   [ ] Receiver spacing
-   [ ] Coverage shell
-   [ ] Pre-snap leverage

------------------------------------------------------------------------

# 11. Environment

-   [ ] Weather
-   [ ] Wind
-   [ ] Temperature
-   [ ] Humidity
-   [ ] Rain/Snow
-   [ ] Altitude
-   [ ] Turf/Grass
-   [ ] Dome
-   [ ] Travel distance
-   [ ] Time-zone changes
-   [ ] Attendance
-   [ ] Rivalry
-   [ ] Primetime

------------------------------------------------------------------------

# 12. Feature Engineering Ideas

-   [ ] Career improvement slope
-   [ ] Age-adjusted production
-   [ ] Opponent-adjusted EPA
-   [ ] Production vs NFL talent
-   [ ] Athleticism over expectation
-   [ ] Consistency index
-   [ ] Development trajectory
-   [ ] Versatility score
-   [ ] Production under pressure
-   [ ] Clutch performance

------------------------------------------------------------------------

# 13. Prospect Development Timeline

Build a chronological event log.

Examples: - Recruitment - Position change - Coaching change - Injuries -
Suspensions - Transfers - Awards - Captaincy - Senior Bowl - Combine -
Draft

------------------------------------------------------------------------

# 14. Off-Field / Character Database

## Team Discipline

-   [ ] Suspension
-   [ ] Dismissal
-   [ ] Reinstatement
-   [ ] Team rules violation

## Legal Events (verified public records only)

-   [ ] Arrest
-   [ ] Citation
-   [ ] Charges filed
-   [ ] Conviction
-   [ ] Charges dismissed

## NCAA / Academic

-   [ ] Academic suspension
-   [ ] Eligibility issue
-   [ ] NCAA violation

## Transfer Context

-   [ ] Transferred after discipline
-   [ ] Transferred after coach left
-   [ ] Multiple transfers

## Suggested Features

-   [ ] Games suspended
-   [ ] Discipline count
-   [ ] Years since last incident
-   [ ] Reinstated
-   [ ] Character Risk Index (transparent methodology)

## Research Sources

-   [ ] Official university statements
-   [ ] Local newspapers
-   [ ] National sports reporting
-   [ ] Public court records (where applicable)
-   [ ] Recruiting coverage

**Do not include rumors, anonymous reports, or unverified social media
claims. Clearly distinguish allegations from confirmed disciplinary
actions.**

------------------------------------------------------------------------

# 15. Long-Term Vision

## Goal

Create a comprehensive longitudinal player-development database that
explains **how prospects develop**, not just how they perform in one
season.

## End State

-   Complete career timelines
-   Advanced contextual features
-   Tracking integration
-   Character timeline (verified)
-   Computer vision features
-   Explainable ML models
-   NFL Draft success prediction
-   Public-facing scouting platform

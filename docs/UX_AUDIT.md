# DraftVision — 10-Laws UX Audit

Audit + implementation pass over the React frontend against the owner's 10 UX laws
(source: `10uxlawsexplainedvisually.md`). Nocturne (`src/nocturne.css`) stays the
visual language — every change below tunes behavior/hierarchy inside it; nothing
was restyled, nothing is solid-filled, all new motion is gated behind
`prefers-reduced-motion: no-preference`.

Legend: **PASS** = already followed the law, left alone. **FIXED** = changed in
this pass. **LEFT** = deliberately not changed, with reasoning.

---

## 1. Jakob's Law — familiarity

| Page | Verdict | Notes |
|---|---|---|
| Navbar | **PASS** | Standard pattern: logo left → links → auth/CTA right; hamburger + slide-down on mobile; sticky bar that condenses on scroll. Active link highlighted via `NavLink`. |
| Search/autocomplete (Predict, Compare) | **PASS** | Debounced type-ahead with a dropdown listbox, Enter-to-submit — the expected shape. |
| Filters (Leaderboard, HS Prospects, Big Board) | **PASS** | Segmented pill groups + search input + sort select — familiar board-site furniture. |
| Tables/boards | **PASS** | Rank / player / pos / school / grade columns, expandable rows with a chevron, Load More pagination. |

No changes made for this law.

## 2. Hick's Law — fewer, clearer choices

- **Nav grouping — LEFT.** The nav has 7 links + auth cluster. I considered a
  "Boards" grouping (College Stars + HS Prospects + Big Board) but decided
  against it: a hover/click dropdown adds a step to every board visit, hides
  destinations behind an unfamiliar label (Jakob's), and 7 flat links is at the
  edge of fine, not past it. The order already front-loads and end-loads the
  important items (see law 7). Documented as a judgment call.
- **Predict empty state — FIXED.** The page opened with an empty text box (max
  choice paralysis: 9,000 possible answers, zero suggestions). Added a
  recommended-default row: `Try: Jeremiah Smith · Arch Manning · Julian Sayin`
  as one-click chips that run the model immediately
  (`PredictionComponent.js` — `SUGGESTED_PLAYERS`, `.predict-try-chip`).
- **Mock draft setup — PASS.** Choices already chunked into labeled rows (Class /
  Rounds / Your team) with the 32-team grid and draft-order editor collapsed
  behind `<details>` — advanced options revealed later, as the law asks.

## 3. Fitts's Law — target size & reach

Audited every tap target at 390 px width; anything under ~36 px got enlarged in
that page's existing mobile media query (desktop sizing untouched):

- **Leaderboard** (`Leaderboard.css` ≤560px): `.lb-seg-btn` → 40 px min-height
  (was ~33 px); Load More `.btn` full-width. Verified with Playwright: seg
  buttons now measure exactly 40 px.
- **Mock Draft** (`MockDraft.css` ≤720px): `.sim-btn` 40 px, `.sim-seg-btn`
  40 px, `.sim-chip` 38 px, `.sim-arrow` 26×24 → 36×34, **Start draft**
  full-width.
- **Big Board** (`BigBoard.css` ≤560px): `.bb-seg-btn` / `.bb-btn` 40 px,
  `.bb-ctl` (↑/↓/✕ row controls) 30×30 → 36×36, Load More full-width.
- **HS Prospects** (`HSProspects.css` ≤720px): `.hsp-seg-btn` 40 px, selects
  42 px, `.hsp-clear` given real padding, Load More full-width.
- **College Stars** (`services.css` ≤760px): Load More full-width.
- **Predict** (`PredictionComponent.css` ≤640px): **Run model** full-width
  (measured 44 px tall), search box stretches, report next-action buttons
  full-width.
- **Home / CTA band** (`HeroSection.css` ≤640px): `.dv-cta` CTAs full-width.
- **LEFT:** row-expand chevrons (Leaderboard) — the entire row is the tap
  target, chevron is just an indicator; already generous. Nav links on mobile
  already 12 px padding at 14 px font (~43 px). Desktop-only hover targets left
  at pointer-appropriate sizes.

## 4. Miller's Law — chunking

- **Player page — PASS.** Already four labeled sections: Grade card → Profile →
  Why this grade → Film; SHAP factors and comps are separate sub-blocks.
- **Predict results — PASS** (structure): hero identity card → three labeled
  metrics → "Top prediction factors" panel → "Closest historical comps" panel →
  meta footer. No walls of text found anywhere; the longest prose is a
  two-line lede. No changes made beyond the Peak-End additions (law 10).
- **Mock draft results — PASS**: summary head → haul rows → round-grouped board.

## 5. Proximity — space creates meaning

- **Leaderboard controls — FIXED.** The position and draft-class segmented
  groups were two visually identical pill clusters with nothing tying either to
  its meaning (aria-labels only). Each seg now carries a micro-label
  ("Position" / "Draft class") grouped directly above it
  (`.lb-seg-group`/`.lb-seg-label`); the controls row aligns to `flex-end` so
  everything shares a baseline. Filters were already separated from the board
  by the top-of-class strip and distribution band.
- **Sim setup panel — PASS.** Every control row already leads with its label
  (`.sim-label`: Class / Rounds / Your team); the panel is boxed apart from the
  board.
- **Forms — PASS.** Big Board editor rank inputs sit inside their rows; the
  admin-key gate is a single labeled inline form. HS Prospects sort selects
  self-label ("Sort: …").
- **LEFT:** HS Prospects seg groups un-labeled — their contents ("QB…OL",
  "All stars/5★/4★") are self-describing, unlike the Leaderboard's ambiguous
  pair ("ALL '27 '28" vs "ALL QB RB").

## 6. Von Restorff — one thing stands out

Nocturne outlines everything, so the primary treatment is a *stronger outline*:
accent border + ~10% accent tint + double glow ring — never solid-filled. One
primary per screen:

- **Home hero & CTA band — FIXED**: "Run a prediction" (`.dv-cta`) gets the
  treatment; the ghost CTA explicitly resets to quiet (`HeroSection.css`).
- **Predict — FIXED**: "Run model" (`.predict-run-btn`).
- **Mock draft — FIXED**: new `.sim-primary` class → "Start draft" (setup) and
  "Run it back" (results). "Sim to my pick" keeps the mid-tier
  `sim-btn-accent` so it reads above plain buttons but below the primary.
- **Big Board — FIXED**: "Save" (`.bb-save`, only when enabled — a disabled
  primary shouldn't glow). And the "recommended-row" analog: the №1 curated
  player (`.bb-row-no1`) gets extra tint, an inner glow, a thicker glowing rank
  bar and a larger rank numeral, above the existing curated treatment.
- **LEFT:** Leaderboard has no primary action (it's a browse surface); its
  standout is already the top-3/top-10 row treatment.

## 7. Serial Position — first and last

- **Nav — PASS, documented choice.** Order is already `Overview` first and
  `Predict` last — exactly the first/last pair worth remembering (the product's
  front door and its signature action). Middle slots hold the boards. No change.
- **Home sections — PASS.** Hero (strongest claim + primary CTA) → How it works
  → Top of the board → CTA band closing with "Run a prediction" again: strong
  start, strong end.

## 8. Tesler's Law — the product does the work

- **Predict auto-run via `?name=` — VERIFIED PASS.** Arriving from the
  Leaderboard/Big Board with `?name=` auto-runs the model and cleans the URL so
  back-navigation doesn't re-trigger. Left as-is.
- **Sim defaults — VERIFIED PASS.** 2027 class, 1 round, projected draft order
  preloaded; last finished mock restored from localStorage with a resume prompt.
- **Search auto-focus — FIXED.** Predict is the one page whose whole job is
  search; the input now focuses itself on arrival — skipped when `?name=` is
  auto-running and on <768 px viewports (keyboard pop-over would hurt more than
  help). **LEFT:** no autofocus on Leaderboard/HS/Stars — those are browse
  pages; stealing focus breaks spacebar-scroll.
- **PASS:** HS search pre-fills from `?q=`; Compare hydrates both slots from
  `/compare?a=&b=` and keeps the URL shareable.

## 9. Doherty Threshold — instant feedback, confirmed success

- **Predict run — PASS**: instant status line with pulsing dot + player name;
  error state with dismiss; per-request timeout.
- **Big Board Save — FIXED**: had a silent success (editor just closed). Now
  shows an inline "Board saved ✓" confirmation (`role="status"`, Nocturne
  accent text) in the controls row for 4 s after a successful save. Failure
  paths already had inline error text.
- **Copy buttons — FIXED**: mock-draft "Copied" → "Copied ✓" (both draft and
  haul), matching the law's confirm-success step.
- **Sim actions — PASS**: "Simulating · {team} on the clock" pulse, pause/resume,
  pool-loading status on setup.
- **Page loads — PASS / LEFT**: Player page already uses shimmer skeletons.
  Leaderboard/Stars use a text state but paint the first 300 rows fast and
  stream the rest with a visible "loading full board…" note — added skeletons
  would be redundant scaffolding for a sub-second first paint, so left.
- **My Board — LEFT**: edits persist automatically and the header already says
  "saved in your browser"; a toast per keystroke would be noise.

## 10. Peak-End Rule — design the peak and the end

- **Mock draft end (the END) — FIXED.** The haul summary now opens with a
  reveal: an overall **Draft grade** letter (average pick-value delta through
  the existing `deltaLetter` buckets) rendered as a large glowing tile with a
  scale-in pop (motion-gated). "Run it back" is now the standout primary
  (`.sim-primary`), with New draft / Copy draft / Copy my haul ✓ alongside.
  Control-every-pick mode has no haul, so it keeps the plain completion card.
- **Predict result (the PEAK) — VERIFIED + FIXED.** The reveal already lands:
  staggered rise-in on identity/metrics/panels, Ken-Burns hero, factor bars
  growing in sequence — all behind `prefers-reduced-motion`. What was missing
  was the ending: the report just stopped at a meta footer. Added a "Next"
  action row — **View full profile · Compare him · Back to the board** —
  (profile/compare links appear when the school is known, so the slug
  resolves; the board link always shows).

---

## Files touched

- `src/components/PredictionComponent.js` / `.css` — try-chips, autofocus,
  next-actions row, primary Run model, mobile full-width.
- `src/components/HeroSection.css` — primary `.dv-cta` treatment, ghost reset,
  mobile full-width CTAs (also covers the Cards.js CTA band).
- `src/components/pages/MockDraft.js` / `.css` — haul grade reveal,
  `.sim-primary`, Copied ✓, mobile tap targets.
- `src/components/pages/BigBoard.js` / `.css` — Saved ✓ confirmation,
  `.bb-save` primary, `.bb-row-no1` treatment, mobile tap targets.
- `src/components/pages/Leaderboard.js` / `.css` — labeled filter groups,
  mobile tap targets, full-width Load More.
- `src/components/pages/HSProspects.css`, `src/components/pages/services.css` —
  mobile tap targets / full-width Load More.

Verified: `CI=true npx react-scripts build` clean; Playwright spot-checks at
390 px and 1400 px (seg buttons measure 40 px, Run model 44 px full-width,
chips render, home hero primary reads above the ghost CTA).

---

## Mobile — systematic pass (390×844 iPhone-class, 360×780 small Android)

Method: Playwright audit of every route at both widths — programmatic
horizontal-overflow check (`scrollingElement.scrollWidth > innerWidth`),
tiny-text scan, hero-video mount check, plus visual review of 54 captures
per run (`/tmp/mobile_audit/{before,after}`). Re-verified after fixes:
**0 overflow failures / 54 captures at both widths, before and after** —
the layouts never leaked horizontally; the failures were *inside* rows.

### Per-page verdicts

- **/ (home) — PASS as found.** Hero stacks, CTAs full-width, stats grid
  2-up, steps + footer clean. Video verified NOT mounted ≤900px (JS gate in
  `HeroSection.js`). Only fix needed: ticker legibility (below).
- **Ticker (global) — FIXED.** The fixed "The board" label ate a third of a
  360px strip and label/grade-chip type sat at 10px. ≤560px: label padding
  slimmed, label text and grade chips lifted to 11px, item text to 12.5px.
- **Navbar — PASS + small fix.** Slide-down menu is roomy with 44px rows;
  the hamburger itself measured ~34px, now `min-width/height: 44px`.
- **/leaderboard — FIXED (the crux).** At 390/360 the fixed columns
  (rank 34 + pos 58 + grade 46 + success 86 + chev 18 + 10px gaps) starved
  the name column to ~10-70px — rows rendered as *logo, pos, grade, number*
  with **no player name at all**, and the POS header overlapped PLAYER.
  ≤560px rebalance: rank 26, pos 34, grade 40, success 64 (17px numeral),
  chev 10, gaps 8, row padding 12 → the name keeps ~140px. Essentials read
  left-to-right: name+logo, pos, grade, success+trend. Hidden at narrow
  widths: school (≤560) and projection (≤860) — both appear in the row's
  expanded panel ("Full scouting report") one tap away, which is why they
  hide rather than wrap. Expanded panel verified clean at 360.
- **/big-board — FIXED, same disease/same cure.** Names truncated to
  "Jere…" at 390. Same column rebalance (success 52 since no trend chip);
  edit-mode rows share the classes so the editor heals too. The fixed
  unsaved-changes save bar now pads `env(safe-area-inset-bottom)` so Save
  clears the home indicator. Hidden ≤560: school (on the player page).
- **/services (College Stars) — FIXED.** Names collapsed to two letters
  ("Sa…"). New ≤560 block mirrors the leaderboard rebalance (rank 26,
  pos 34, grade 40, success 64/17px, gaps 8). Hidden ≤760 (pre-existing):
  school, projection — both on the player's full report.
- **/hs-prospects — FIXED.** At 390 the name and HS/City columns split
  ~56px between them: names showed one letter and the thead overlapped
  ("PL/POS/CITY" jumble). ≤560: **HS/City column hidden** — it's the
  lowest-value column (searchable via the search box, and commit/year
  already hide at 860/720); rank 30, pos 34, stars pill tightened (72px),
  rating 58/15px. Names now read ~10+ chars; thead is clean.
- **/mock-draft — PASS + polish.** Setup (team grid 3-up, full-width Start),
  live board and on-the-clock list were already right (grade/prob columns
  already hide ≤720). Haul summary rows truncated names ("Leonard Moc…") —
  grid rebalanced (30px pick, 34px pos, 40px grade, 10px gaps) so full
  names fit alongside STEAL/REACH tags.
- **/predict — PASS as found.** Empty state (chips, full-width Run model)
  and the full report both verified at 360 — no changes needed.
- **/player/:slug — PASS + one fix.** Hero, metric grid and factor bars
  stack correctly. Historical-comps rows wrapped the comp's *name*
  mid-word while the outcome text squeezed beside it: `.pp-comp` now
  `flex-wrap: wrap`, letting the outcome line (already `flex-basis: 100%`)
  take its own row — improves desktop too.
- **/compare — PASS as found.** Empty and `?a=&b=` populated verified at
  both widths: slot cards, verdict, aligned factor pairs all stack.
- **/backtest, /edge — PASS as found.** Card-based lists; no tables to fix.
- **Footer — polish.** Link row gets padded ~40px tap targets and
  `env(safe-area-inset-bottom)` ≤560px.

### What hides at narrow widths, and why

| Page | Hidden ≤560-860px | Where it still lives |
| --- | --- | --- |
| Leaderboard | school (≤560), projection (≤860) | expanded row panel / full report |
| Big Board | school (≤560) | player page |
| College Stars | school, projection (≤760) | player page |
| HS Prospects | HS/City (≤560), year (≤720), commit (≤860) | search matches school/state/commit |
| Mock draft lists | school, grade/prob on dense lists (≤720) | pick detail / Predict link |

The kept quartet on every board row — **name+logo, pos, grade, success** —
is the minimum a scout needs to rank players; everything hidden is
secondary identification or duplicated one tap deeper.

Remaining sub-12px strings are all Nocturne eyebrow labels and badges
(10-10.5px uppercase kickers, EST/STEAL/REACH chips) — display metadata by
design, not body copy; body text everywhere measures ≥12.5px.

### Files touched (mobile pass)

- `src/components/pages/Leaderboard.css` — ≤560 column rebalance
- `src/components/pages/BigBoard.css` — ≤560 rebalance + save-bar safe area
- `src/components/pages/services.css` — new ≤560 block
- `src/components/pages/HSProspects.css` — new ≤560 block, school hidden
- `src/components/pages/MockDraft.css` — haul-row grid rebalance
- `src/components/pages/PlayerPage.css` — `.pp-comp` wrap fix
- `src/components/Ticker.css` — ≤560 legibility block
- `src/components/Navbar.css` — 44px hamburger
- `src/components/Footer.css` — ≤560 tap targets + safe area

Verified: `CI=true npx react-scripts build` clean; full Playwright re-run at
390 px and 360 px — 54/54 captures overflow-free on every route, video never
mounts at phone widths, desktop (1400 px) spot-checked for regressions.

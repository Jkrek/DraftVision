/* Draft engine — pure logic for the interactive Mock Draft Simulator.
   No React, no fetch: everything is deterministic given its inputs
   (jitter comes from a seed carried in the sim object), so the whole
   engine can be sanity-checked from node. */

export const ROUND_PICKS = 32;

/* Fallback first-round order, used only until /api/draft-order answers (or
   if it cannot): the page normally runs on the simulated order instead. */
export const DRAFT_ORDER_2027 = [
  'New Orleans Saints',
  'Cleveland Browns',
  'Tennessee Titans',
  'New York Jets',
  'Las Vegas Raiders',
  'New York Giants',
  'Carolina Panthers',
  'Arizona Cardinals',
  'Indianapolis Colts',
  'Miami Dolphins',
  'Atlanta Falcons',
  'Dallas Cowboys',
  'New England Patriots',
  'Pittsburgh Steelers',
  'Seattle Seahawks',
  'Chicago Bears',
  'Tampa Bay Buccaneers',
  'Houston Texans',
  'Los Angeles Chargers',
  'Denver Broncos',
  'Minnesota Vikings',
  'Jacksonville Jaguars',
  'Washington Commanders',
  'Cincinnati Bengals',
  'San Francisco 49ers',
  'Los Angeles Rams',
  'Green Bay Packers',
  'Detroit Lions',
  'Kansas City Chiefs',
  'Baltimore Ravens',
  'Philadelphia Eagles',
  'Buffalo Bills',
];

/* Sentinel for "control every pick" mode (no CPU-vs-user split). */
export const CONTROL_ALL = 'ALL';

/* ── Grades & position groups (mirrors Leaderboard's vocabulary) ─────── */

const GRADE_ORDER = {
  'A+': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4,
  'B-': 5, 'C+': 6, 'C': 7, 'C-': 8, 'D': 9,
};

export function gradeRank(g) {
  return GRADE_ORDER[g] !== undefined ? GRADE_ORDER[g] : 9;
}

const POS_GROUP_MAP = {
  DB: ['CB', 'S', 'DB', 'FS', 'SS'],
  LB: ['LB', 'ILB', 'OLB', 'MLB'],
  DL: ['DL', 'DE', 'DT', 'EDGE', 'NT'],
  OL: ['OL', 'OT', 'OG', 'C', 'LS'],
};

export const POS_TABS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB'];

export function posGroup(pos) {
  const p = (pos || '').toUpperCase();
  for (const group of Object.keys(POS_GROUP_MAP)) {
    if (POS_GROUP_MAP[group].indexOf(p) !== -1) return group;
  }
  return p || '?';
}

/* ── Player pool ─────────────────────────────────────────────────────── */

export function boardSort(a, b) {
  return (
    gradeRank(a.grade) - gradeRank(b.grade) ||
    (Number(b.success_probability) || 0) - (Number(a.success_probability) || 0)
  );
}

/* Attach a stable 1-based consensus rank to a board-ordered pool. The
   rank never changes as players come off the board, so value deltas
   (rank − pick number) stay meaningful all draft long. */
function withRanks(pool) {
  return pool.map((p, i) => Object.assign({}, p, { boardRank: i + 1 }));
}

/* /api/prospects rows → draftable pool for a draft class: only players
   projected eligible that year (fallback: everyone if the field is
   missing from an old cache), best board order first. */
export function buildPool(rows, draftClass = 2027) {
  const list = Array.isArray(rows) ? rows : [];
  const cls = list.filter((p) => Number(p.draft_class) === Number(draftClass));
  const pool = (cls.length >= ROUND_PICKS ? cls : list).slice();
  pool.sort(boardSort);
  return withRanks(pool);
}

/* /api/big-board rows → pool: the owner-curated board first (in owner
   order), then the model-ranked rest. Rank = overall board position. */
export function buildPoolFromBoard(board, rest) {
  const b = Array.isArray(board) ? board : [];
  const r = Array.isArray(rest) ? rest : [];
  return withRanks(b.concat(r).filter((p) => p && p.name));
}

/* ── Pick value: steal/reach tags + letter grades ────────────────────────
   delta = (board/consensus rank) − (pick number): positive → the player
   lasted past his rank (value), negative → taken early (cost). */

export const STEAL_DELTA = 12;
export const REACH_DELTA = -12;

export function valueTag(delta) {
  if (!Number.isFinite(delta)) return null;
  if (delta >= STEAL_DELTA) return 'STEAL';
  if (delta <= REACH_DELTA) return 'REACH';
  return null;
}

/* A–F from delta buckets (no E, classic report card). */
export function deltaLetter(delta) {
  if (!Number.isFinite(delta)) return null;
  if (delta >= STEAL_DELTA) return 'A';
  if (delta >= 4) return 'B';
  if (delta >= -3) return 'C';
  if (delta > REACH_DELTA) return 'D';
  return 'F';
}

/* ── Pick grade: value and fit ───────────────────────────────────────────
   A pick starts at 82. Value moves it by how far the player's board rank is
   from the pick, measured against a tolerance that grows through the draft
   (four spots matter at pick 5, not at pick 90). Fit adds up to 10 for how
   much the team needed the position when it picked. 90+ A, 80+ B, 70+ C,
   60+ D, else F. Returns { score, letter, value, fit } or null. */
export function pickGrade(pk) {
  if (!pk || !Number.isFinite(pk.delta)) return null;
  const tolerance = Math.max(4, pk.pick * 0.15);
  const value = Math.round(Math.max(-22, Math.min(12, (pk.delta / tolerance) * 8)));
  const fit = Math.round((Number(pk.need) || 0) * 10);
  const score = Math.max(40, Math.min(100, 82 + value + fit));
  const letter = score >= 90 ? 'A' : score >= 80 ? 'B' : score >= 70 ? 'C' : score >= 60 ? 'D' : 'F';
  return { score, letter, value, fit };
}

export function gradeLetter(score) {
  if (!Number.isFinite(score)) return null;
  return score >= 90 ? 'A' : score >= 80 ? 'B' : score >= 70 ? 'C' : score >= 60 ? 'D' : 'F';
}

/* ── Trade value: approximated Jimmy Johnson chart ───────────────────────
   value(pick) = 3000 · 0.9^(pick−1), floored at 1 point. */

export function chartValue(pick) {
  const p = Math.max(1, Math.floor(pick));
  return Math.max(1, Math.floor(3000 * Math.pow(0.9, p - 1)));
}

/* The CPU accepts when what it receives (userGive) is roughly fair
   against what it surrenders (userGet): within ±15% on the chart. */
export function cpuAccepts(userGive, userGet) {
  return userGive >= userGet * 0.85 && userGive <= userGet * 1.15;
}

/* ── Deterministic jitter ────────────────────────────────────────────── */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* One deterministic value in [0, 1) per (session seed, key). Keyed by
   pick number (and candidate slot) so runs differ across sessions but a
   replay with the same seed reproduces exactly. */
export function pickJitter(seed, key) {
  return mulberry32(((seed >>> 0) ^ Math.imul(key, 0x9e3779b9)) >>> 0)();
}

/* ── Team needs ──────────────────────────────────────────────────────────
   `needs` is the /api/draft-needs payload: per team, a need of 0..1 for each
   position group (the share of that group's starters whose contracts are
   running out), plus the table that maps a college position label onto
   those groups. A drafted player replaces one starter: he takes 1/starters
   off his group's need, so one quarterback settles the quarterback need and
   one linebacker settles a third of a three-linebacker need. */

export const NEED_HIGH = 0.75; // "top need" threshold (e.g. the only QB is out of contract)
export const NEED_SOME = 0.4;  // "need" threshold (e.g. one of two starters)

/* Remaining need per group for a team, after the picks it has already made. */
export function teamNeeds(sim, team) {
  const base = sim && sim.needs && sim.needs.teams && sim.needs.teams[team];
  if (!base) return {};
  const out = {};
  for (const g of Object.keys(base.groups)) out[g] = base.groups[g].need;
  for (const pk of sim.picks) {
    if (pk.team === team && pk.needGroup && out[pk.needGroup] !== undefined) {
      const starters = base.groups[pk.needGroup].starters || 1;
      out[pk.needGroup] = Math.max(0, Math.round((out[pk.needGroup] - 1 / starters) * 1000) / 1000);
    }
  }
  return out;
}

/* Which of a team's need groups a player fills, and how much it needs him:
   { group, need }. A generic label (OL, DL, DB) counts toward whichever of
   its groups the team needs more. need = 0 when it fills nothing. */
export function playerFit(sim, team, player, remaining) {
  const table = (sim && sim.needs && sim.needs.positions) || {};
  const groups = table[(player.position || '').toUpperCase()] || [];
  const left = remaining || teamNeeds(sim, team);
  let best = { group: groups[0] || null, need: 0 };
  for (const g of groups) {
    if ((left[g] || 0) > best.need) best = { group: g, need: left[g] };
  }
  return best;
}

/* ── CPU pick model ──────────────────────────────────────────────────── */

/* Mild positional value: QBs go early, RBs slide. */
const POS_VALUE = { QB: 5, DL: 2.5, OL: 2.5, WR: 1.5, DB: 1.5, RB: -2 };

const CPU_WINDOW = 14; // best-available candidates considered per pick

/* How far need can move a pick. The window's players sit about 0.6 to 3
   score points apart, so a top need (1.0) is worth roughly four to six board
   spots: enough to take the quarterback over a slightly better guard, not
   enough to reach a round early. */
export const NEED_WEIGHT = 9;
const JITTER = 8;

/* The two dials on the setup screen, each 0..10. `need` 0 is pure best
   available and 10 doubles the default pull of need; `chaos` 0 follows the
   board exactly and 10 is about three times the default randomness. */
export const DEFAULT_SETTINGS = { need: 5, chaos: 3 };

function tuning(settings) {
  const s = Object.assign({}, DEFAULT_SETTINGS, settings);
  return { needWeight: NEED_WEIGHT * (s.need / 5), jitter: JITTER * (s.chaos / 3) };
}

/* Returns an index into `pool` (board-sorted, best first), or -1. Scores
   the top of the board on (a) success probability + grade, (b) positional
   need — repeat picks of a position group are dampened via posCounts —
   and (c) small deterministic jitter. */
export function cpuSelect(pool, posCounts, pickNumber, seed, needOf, tune) {
  const needWeight = tune ? tune.needWeight : NEED_WEIGHT;
  const jitterAmp = tune ? tune.jitter : JITTER;
  if (!pool || pool.length === 0) return -1;
  // The field of realistic candidates widens as the draft goes on: boards
  // agree at the top and scatter later.
  const round = Math.max(1, Math.ceil(pickNumber / ROUND_PICKS));
  const n = Math.min(CPU_WINDOW + 6 * (round - 1), pool.length);
  let best = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < n; i++) {
    const p = pool[i];
    const group = posGroup(p.position);
    const taken = (posCounts && posCounts[group]) || 0;
    const jitter = seed === null ? 0 : (pickJitter(seed, pickNumber * 131 + i) - 0.5) * jitterAmp;
    const score =
      (Number(p.success_probability) || 0) +
      (9 - gradeRank(p.grade)) * 1.5 -
      i * 0.6 + // slight respect for board order
      (POS_VALUE[group] || 0) -
      taken * 9 + // do not stack one position group
      (needOf ? needOf(p) * needWeight : 0) +
      jitter;
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best;
}

/* ── Sim state (immutable transitions) ───────────────────────────────── */

/* Round-one picks that already belong to another team. `holders` maps the
   team whose record earned a pick to the team that holds it; the result is
   the slots/via pair createDraft takes. Later rounds are untouched. */
export function tradedSlots(order, holders) {
  const slots = {};
  const via = {};
  (order || []).forEach((team, i) => {
    const holder = holders && holders[team];
    if (holder && holder !== team) {
      slots[i + 1] = holder;
      via[i + 1] = team;
    }
  });
  return { slots, via };
}

export function createDraft({ order, rounds, userTeam, pool, seed, slots, via, needs, settings }) {
  const ord = order && order.length ? order : DRAFT_ORDER_2027;
  return {
    order: ord.slice(),
    rounds,
    userTeam, // team name, or CONTROL_ALL
    seed: seed >>> 0,
    pool: pool.slice(),
    picks: [],
    posCounts: {},
    slots: Object.assign({}, slots), // pickNo → team (trade overrides of the base order)
    via: Object.assign({}, via),     // pickNo → counterparty team ("via trade with X")
    tradeLog: [], // accepted trades, in order
    needs: needs || null, // /api/draft-needs payload (null → needs ignored)
    settings: Object.assign({}, DEFAULT_SETTINGS, settings),
    totalPicks: Math.min(rounds * ord.length, pool.length),
  };
}

export function isDone(sim) {
  return sim.picks.length >= sim.totalPicks;
}

export function currentPickNumber(sim) {
  return sim.picks.length + 1;
}

/* Which team owns a pick slot — base order unless traded. */
export function pickTeam(sim, pickNo) {
  return (
    (sim.slots && sim.slots[pickNo]) ||
    sim.order[(pickNo - 1) % sim.order.length]
  );
}

export function pickRound(sim, pickNo) {
  return Math.ceil(pickNo / sim.order.length);
}

export function currentTeam(sim) {
  return pickTeam(sim, currentPickNumber(sim));
}

export function isUserOnClock(sim) {
  return (
    !isDone(sim) &&
    (sim.userTeam === CONTROL_ALL || currentTeam(sim) === sim.userTeam)
  );
}

export function commitPick(sim, poolIndex) {
  if (isDone(sim) || poolIndex < 0 || poolIndex >= sim.pool.length) return sim;
  const player = sim.pool[poolIndex];
  const team = currentTeam(sim);
  const pickNo = currentPickNumber(sim);
  const group = posGroup(player.position);
  const teamCounts = Object.assign({}, sim.posCounts[team]);
  teamCounts[group] = (teamCounts[group] || 0) + 1;
  const delta = Number.isFinite(player.boardRank)
    ? player.boardRank - pickNo
    : null;
  const fit = playerFit(sim, team, player);
  return Object.assign({}, sim, {
    pool: sim.pool.filter((_, i) => i !== poolIndex),
    picks: sim.picks.concat({
      pick: pickNo,
      round: pickRound(sim, pickNo),
      team,
      player,
      delta,
      tag: valueTag(delta),
      via: (sim.via && sim.via[pickNo]) || null,
      needGroup: fit.group, // the need group this pick counts toward
      need: fit.need,       // how much the team needed it when it picked (0..1)
    }),
    posCounts: Object.assign({}, sim.posCounts, { [team]: teamCounts }),
  });
}

export function advanceCpu(sim) {
  if (isDone(sim)) return sim;
  const team = currentTeam(sim);
  const left = teamNeeds(sim, team);
  const idx = cpuSelect(
    sim.pool,
    sim.posCounts[team],
    currentPickNumber(sim),
    sim.seed,
    (p) => playerFit(sim, team, p, left).need,
    tuning(sim.settings)
  );
  return commitPick(sim, idx);
}

/* The pick the model would make for the team on the clock: the same score
   the CPU uses, without the randomness. Returns { index, player, fit } or
   null. Shown to the user as a suggestion, never made for them. */
export function suggestPick(sim) {
  if (isDone(sim) || !sim.pool.length) return null;
  const team = currentTeam(sim);
  const left = teamNeeds(sim, team);
  const index = cpuSelect(sim.pool, sim.posCounts[team], currentPickNumber(sim), null,
    (p) => playerFit(sim, team, p, left).need, tuning(sim.settings));
  const player = sim.pool[index];
  return { index, player, fit: playerFit(sim, team, player, left) };
}

/* The picks a team still has to make, in order. */
export function upcomingPicks(sim, team) {
  const out = [];
  for (let p = currentPickNumber(sim); p <= sim.totalPicks; p++) {
    if (pickTeam(sim, p) === team) out.push(p);
  }
  return out;
}

/* CPU-picks everything that remains (user team included) — "sim to end"
   core, and the node-side sanity harness. */
export function simulateToEnd(sim) {
  let s = sim;
  while (!isDone(s)) s = advanceCpu(s);
  return s;
}

/* ── Trades ──────────────────────────────────────────────────────────────
   Offers are plain descriptions: { dir, team, userGives, userGets, give,
   get, fair }. give/get are chart points from the user's side; `fair` is
   whether the CPU would accept (±15% on the chart). Offer lists always
   show the next 3 candidate picks; unfair ones render disabled. */

/* First not-yet-made pick at or after `from` owned by `team`, optionally
   restricted to rounds later than `afterRound`. 0 when none. */
function nextTeamPick(sim, team, from, afterRound) {
  for (let p = Math.max(from, currentPickNumber(sim)); p <= sim.totalPicks; p++) {
    if (pickTeam(sim, p) !== team) continue;
    if (afterRound && pickRound(sim, p) <= afterRound) continue;
    return p;
  }
  return 0;
}

/* User on the clock at P → the next 3 CPU picks as trade-down partners.
   Multi-round drafts package the partner's later pick with their next
   pick in a future round; single-round drafts are straight swaps. Either
   way the package shown is the one closest to chart-fair. */
export function tradeDownOffers(sim) {
  if (sim.userTeam === CONTROL_ALL || !isUserOnClock(sim)) return [];
  const P = currentPickNumber(sim);
  const give = chartValue(P);
  const offers = [];
  for (let q = P + 1; q <= sim.totalPicks && offers.length < 3; q++) {
    const team = pickTeam(sim, q);
    if (team === sim.userTeam) continue;
    const swapGet = chartValue(q);
    let userGets = [q];
    let get = swapGet;
    if (sim.rounds > 1) {
      const future = nextTeamPick(sim, team, q + 1, pickRound(sim, q));
      if (future) {
        const pkgGet = swapGet + chartValue(future);
        // prefer the two-pick package when it's fair, else whichever is closer
        if (
          cpuAccepts(give, pkgGet) ||
          (!cpuAccepts(give, swapGet) &&
            Math.abs(pkgGet - give) < Math.abs(swapGet - give))
        ) {
          userGets = [q, future];
          get = pkgGet;
        }
      }
    }
    offers.push({
      dir: 'down',
      team,
      userGives: [P],
      userGets,
      give,
      get,
      fair: cpuAccepts(give, get),
    });
  }
  return offers;
}

/* CPU on the clock, user pick still ahead → the up-to-3 CPU picks before
   the user's next slot as trade-up targets (mirror logic: the user sends
   their next pick, sweetened with a future-round pick when the straight
   swap falls short of chart-fair). */
export function tradeUpOffers(sim) {
  if (sim.userTeam === CONTROL_ALL || isDone(sim) || isUserOnClock(sim)) {
    return [];
  }
  const P = nextTeamPick(sim, sim.userTeam, currentPickNumber(sim), 0);
  if (!P) return [];
  const sweet =
    sim.rounds > 1 ? nextTeamPick(sim, sim.userTeam, P + 1, pickRound(sim, P)) : 0;
  const base = chartValue(P);
  // mirror of trade-down: the up-to-3 CPU picks closest ahead of P that
  // haven't happened yet
  const targets = [];
  for (let q = P - 1; q >= currentPickNumber(sim) && targets.length < 3; q--) {
    if (pickTeam(sim, q) !== sim.userTeam) targets.push(q);
  }
  targets.reverse();
  const offers = [];
  for (const q of targets) {
    const team = pickTeam(sim, q);
    const get = chartValue(q);
    let userGives = [P];
    let give = base;
    if (!cpuAccepts(give, get) && sweet) {
      userGives = [P, sweet];
      give = base + chartValue(sweet);
    }
    offers.push({
      dir: 'up',
      team,
      userGives,
      userGets: [q],
      give,
      get,
      fair: cpuAccepts(give, get),
    });
  }
  return offers;
}

/* Apply an accepted offer: reassign the slots and remember counterparties
   so the eventual picks read "via trade with X". No-op on unfair offers. */
export function executeTrade(sim, offer) {
  if (!offer || !offer.fair) return sim;
  const slots = Object.assign({}, sim.slots);
  const via = Object.assign({}, sim.via);
  for (const p of offer.userGives) {
    slots[p] = offer.team;
    via[p] = sim.userTeam;
  }
  for (const p of offer.userGets) {
    slots[p] = sim.userTeam;
    via[p] = offer.team;
  }
  return Object.assign({}, sim, {
    slots,
    via,
    tradeLog: (sim.tradeLog || []).concat({
      at: currentPickNumber(sim),
      dir: offer.dir,
      team: offer.team,
      gives: offer.userGives.slice(),
      gets: offer.userGets.slice(),
    }),
  });
}

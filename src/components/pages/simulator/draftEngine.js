/* Draft engine — pure logic for the interactive Mock Draft Simulator.
   No React, no fetch: everything is deterministic given its inputs
   (jitter comes from a seed carried in the sim object), so the whole
   engine can be sanity-checked from node. */

export const ROUND_PICKS = 32;

/* Placeholder 2027 first-round order — reverse of a plausible 2026
   standings guess. Labeled "projected" in the UI; the user can reorder. */
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

/* /api/prospects rows → draftable pool: draft_class 2027 when the field
   exists (fallback: everyone), best board order first. */
export function buildPool(rows) {
  const list = Array.isArray(rows) ? rows : [];
  const cls = list.filter((p) => Number(p.draft_class) === 2027);
  const pool = (cls.length >= ROUND_PICKS ? cls : list).slice();
  pool.sort(boardSort);
  return pool;
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

/* ── CPU pick model ──────────────────────────────────────────────────── */

/* Mild positional value: QBs go early, RBs slide. */
const POS_VALUE = { QB: 5, DL: 2.5, OL: 2.5, WR: 1.5, DB: 1.5, RB: -2 };

const CPU_WINDOW = 14; // best-available candidates considered per pick

/* Returns an index into `pool` (board-sorted, best first), or -1. Scores
   the top of the board on (a) success probability + grade, (b) positional
   need — repeat picks of a position group are dampened via posCounts —
   and (c) small deterministic jitter. */
export function cpuSelect(pool, posCounts, pickNumber, seed) {
  if (!pool || pool.length === 0) return -1;
  const n = Math.min(CPU_WINDOW, pool.length);
  let best = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < n; i++) {
    const p = pool[i];
    const group = posGroup(p.position);
    const taken = (posCounts && posCounts[group]) || 0;
    const jitter = (pickJitter(seed, pickNumber * 131 + i) - 0.5) * 8;
    const score =
      (Number(p.success_probability) || 0) +
      (9 - gradeRank(p.grade)) * 1.5 -
      i * 0.6 + // slight respect for board order
      (POS_VALUE[group] || 0) -
      taken * 9 + // positional-need dampening
      jitter;
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best;
}

/* ── Sim state (immutable transitions) ───────────────────────────────── */

export function createDraft({ order, rounds, userTeam, pool, seed }) {
  const ord = order && order.length ? order : DRAFT_ORDER_2027;
  return {
    order: ord.slice(),
    rounds,
    userTeam, // team name, or CONTROL_ALL
    seed: seed >>> 0,
    pool: pool.slice(),
    picks: [],
    posCounts: {},
    totalPicks: Math.min(rounds * ord.length, pool.length),
  };
}

export function isDone(sim) {
  return sim.picks.length >= sim.totalPicks;
}

export function currentPickNumber(sim) {
  return sim.picks.length + 1;
}

export function currentTeam(sim) {
  return sim.order[sim.picks.length % sim.order.length];
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
  return Object.assign({}, sim, {
    pool: sim.pool.filter((_, i) => i !== poolIndex),
    picks: sim.picks.concat({
      pick: pickNo,
      round: Math.ceil(pickNo / sim.order.length),
      team,
      player,
    }),
    posCounts: Object.assign({}, sim.posCounts, { [team]: teamCounts }),
  });
}

export function advanceCpu(sim) {
  if (isDone(sim)) return sim;
  const team = currentTeam(sim);
  const idx = cpuSelect(
    sim.pool,
    sim.posCounts[team],
    currentPickNumber(sim),
    sim.seed
  );
  return commitPick(sim, idx);
}

/* CPU-picks everything that remains (user team included) — "sim to end"
   core, and the node-side sanity harness. */
export function simulateToEnd(sim) {
  let s = sim;
  while (!isDone(s)) s = advanceCpu(s);
  return s;
}

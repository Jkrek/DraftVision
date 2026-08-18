/* Node-side sanity assertions for the draft engine — no build step, no
   test runner. Run from anywhere:

     node src/components/pages/simulator/engineCheck.node.mjs

   draftEngine.js is authored as ESM but lives in a CJS package (CRA), so
   the harness copies it to a temp .mjs and dynamic-imports that. Covers:
   determinism, pool ranking, steal/reach tagging + value letters, the
   Jimmy Johnson chart approximation, and trade down/up offer math. */

import { mkdtempSync, copyFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import assert from 'node:assert/strict';

function makePool(E) {
  const POS = ['QB', 'WR', 'DL', 'OL', 'DB', 'RB', 'LB', 'TE'];
  const GRADES = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C'];
  const rows = [];
  for (let i = 0; i < 40; i++) {
    rows.push({
      name: `Player ${i + 1}`,
      position: POS[i % POS.length],
      team: `School ${i + 1}`,
      grade: GRADES[Math.min(GRADES.length - 1, Math.floor(i / 5))],
      success_probability: 99 - i,
      draft_class: 2027,
    });
  }
  // exercise the big-board path: 5 owner-ordered rows, then the rest
  return E.buildPoolFromBoard(rows.slice(0, 5), rows.slice(5));
}

async function main() {
  const here = dirname(fileURLToPath(import.meta.url));
  const dir = mkdtempSync(join(tmpdir(), 'dv-engine-'));
  const mod = join(dir, 'draftEngine.mjs');
  copyFileSync(join(here, 'draftEngine.js'), mod);
  const E = await import(pathToFileURL(mod).href);

  /* ── Jimmy Johnson chart approximation ── */
  assert.equal(E.chartValue(1), 3000);
  assert.equal(E.chartValue(2), 2700);
  assert.equal(E.chartValue(3), 2430);
  assert.equal(E.chartValue(200), 1, 'chart floors at 1 point');
  for (let p = 1; p < 100; p++) {
    assert.ok(E.chartValue(p) >= E.chartValue(p + 1), 'chart is monotonic');
  }

  /* ── ±15% CPU acceptance window ── */
  assert.ok(E.cpuAccepts(850, 1000));
  assert.ok(!E.cpuAccepts(849, 1000));
  assert.ok(E.cpuAccepts(1150, 1000));
  assert.ok(!E.cpuAccepts(1151, 1000));

  /* ── Steal / reach tagging + letters ── */
  assert.equal(E.valueTag(12), 'STEAL');
  assert.equal(E.valueTag(11), null);
  assert.equal(E.valueTag(-12), 'REACH');
  assert.equal(E.valueTag(-11), null);
  assert.equal(E.valueTag(null), null);
  assert.equal(E.deltaLetter(40), 'A');
  assert.equal(E.deltaLetter(12), 'A');
  assert.equal(E.deltaLetter(4), 'B');
  assert.equal(E.deltaLetter(3), 'C');
  assert.equal(E.deltaLetter(-3), 'C');
  assert.equal(E.deltaLetter(-4), 'D');
  assert.equal(E.deltaLetter(-11), 'D');
  assert.equal(E.deltaLetter(-12), 'F');
  assert.equal(E.deltaLetter(null), null);

  /* ── Pool ranking ── */
  const pool = makePool(E);
  assert.equal(pool.length, 40);
  pool.forEach((p, i) => assert.equal(p.boardRank, i + 1));
  assert.equal(pool[0].name, 'Player 1', 'owner board order preserved');
  const modelPool = E.buildPool(
    pool.map(({ boardRank, ...p }) => p) // eslint-disable-line no-unused-vars
  );
  modelPool.forEach((p, i) => assert.equal(p.boardRank, i + 1));

  const TEAMS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'];
  const mk = (userTeam, seed) =>
    E.createDraft({ order: TEAMS, rounds: 2, userTeam, pool, seed });

  /* ── Determinism: same seed → identical draft ── */
  const a = E.simulateToEnd(mk('T3', 42));
  const b = E.simulateToEnd(mk('T3', 42));
  assert.deepEqual(
    a.picks.map((pk) => pk.player.name),
    b.picks.map((pk) => pk.player.name),
    'same seed reproduces the draft'
  );

  /* ── Every pick carries delta = rank − pick and a consistent tag ── */
  for (const pk of a.picks) {
    assert.equal(pk.delta, pk.player.boardRank - pk.pick);
    assert.equal(pk.tag, E.valueTag(pk.delta));
  }

  /* ── Trade down (user on the clock at pick 3) ── */
  let sim = mk('T3', 7);
  sim = E.advanceCpu(E.advanceCpu(sim));
  assert.ok(E.isUserOnClock(sim));
  assert.equal(E.currentPickNumber(sim), 3);
  const down = E.tradeDownOffers(sim);
  assert.ok(down.length > 0 && down.length <= 3);
  for (const o of down) {
    assert.equal(o.give, E.chartValue(3));
    assert.equal(
      o.get,
      o.userGets.reduce((s, p) => s + E.chartValue(p), 0)
    );
    assert.equal(o.fair, E.cpuAccepts(o.give, o.get));
  }
  const fairDown = down.find((o) => o.fair);
  assert.ok(fairDown, 'an adjacent-pick trade down is chart-fair');
  const afterDown = E.executeTrade(sim, fairDown);
  assert.ok(!E.isUserOnClock(afterDown), 'user surrendered the pick');
  assert.equal(E.pickTeam(afterDown, 3), fairDown.team);
  for (const p of fairDown.userGets) {
    assert.equal(E.pickTeam(afterDown, p), 'T3');
  }
  const doneDown = E.simulateToEnd(afterDown);
  const swapped = doneDown.picks.find((pk) => pk.pick === fairDown.userGets[0]);
  assert.equal(swapped.team, 'T3');
  assert.equal(swapped.via, fairDown.team, 'board logs "via trade with X"');
  assert.equal(doneDown.picks[2].via, 'T3');

  /* ── Unfair offers cannot be executed ── */
  const unfair = down.find((o) => !o.fair);
  if (unfair) assert.equal(E.executeTrade(sim, unfair), sim);

  /* ── Trade up (CPU on the clock, user pick still ahead) ── */
  const up0 = mk('T5', 11); // user picks at 5 and 11; pick 1 on the clock
  const up = E.tradeUpOffers(up0);
  assert.deepEqual(
    up.map((o) => o.userGets[0]),
    [2, 3, 4],
    'targets the 3 CPU picks closest ahead of the user slot'
  );
  for (const o of up) {
    assert.equal(o.get, E.chartValue(o.userGets[0]));
    assert.equal(
      o.give,
      o.userGives.reduce((s, p) => s + E.chartValue(p), 0)
    );
    assert.equal(o.fair, E.cpuAccepts(o.give, o.get));
  }
  const adjUp = up.find((o) => o.userGets[0] === 4);
  assert.ok(
    adjUp.fair && adjUp.userGives.length === 1,
    'one-slot jump is a fair straight swap'
  );
  const deepUp = up.find((o) => o.userGets[0] === 2);
  assert.equal(
    deepUp.userGives.length,
    2,
    'multi-slot jump adds the future-round sweetener'
  );
  const afterUp = E.executeTrade(up0, adjUp);
  assert.equal(E.pickTeam(afterUp, 4), 'T5');
  assert.equal(E.pickTeam(afterUp, 5), adjUp.team);
  let s2 = afterUp;
  while (!E.isUserOnClock(s2)) s2 = E.advanceCpu(s2);
  assert.equal(
    E.currentPickNumber(s2),
    4,
    'user comes on the clock at the acquired pick'
  );

  /* ── Trades preserve determinism ── */
  const t1 = E.simulateToEnd(E.executeTrade(E.advanceCpu(E.advanceCpu(mk('T3', 42))), fairDown));
  const t2 = E.simulateToEnd(E.executeTrade(E.advanceCpu(E.advanceCpu(mk('T3', 42))), fairDown));
  assert.deepEqual(
    t1.picks.map((pk) => `${pk.team}:${pk.player.name}`),
    t2.picks.map((pk) => `${pk.team}:${pk.player.name}`)
  );

  rmSync(dir, { recursive: true, force: true });
  console.log('engineCheck: all assertions passed');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

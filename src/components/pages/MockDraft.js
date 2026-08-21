import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { anonFetch } from '../../lib/api';
import '../../App.css';
import './MockDraft.css';
import { nflLogoUrl } from '../nflTeams';
import MockRow from './simulator/MockRow';
import ImportMock from './simulator/ImportMock';
import {
  DRAFT_ORDER_2027,
  CONTROL_ALL,
  POS_TABS,
  posGroup,
  buildPool,
  buildPoolFromBoard,
  createDraft,
  advanceCpu,
  commitPick,
  isDone,
  isUserOnClock,
  currentTeam,
  currentPickNumber,
  deltaLetter,
  tradeDownOffers,
  tradeUpOffers,
  executeTrade,
} from './simulator/draftEngine';

/* v2: picks carry delta/tag/via, results carry orderLen — old v1 saves
   are simply ignored rather than loaded into the new shape. */
const STORE_KEY = 'dv_mock_sim_v3';
const SCHEMA_V = 3;

/* Draft classes with proper eligibility bucketing: rows carry draft_class
   (2027 = current seniors/juniors … 2030 = true freshmen). */
const DRAFT_YEARS = [2027, 2028, 2029, 2030];

const HERO_IMG = process.env.PUBLIC_URL + '/images/CFB Content/lamar.png';

/* "San Francisco 49ers" → "49ers" — compact chip label. */
const nick = (t) => (t || '').split(' ').pop();

const fmtProb = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? `${Math.round(n)}%` : '—';
};

function newSeed() {
  return (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0;
}

/* ── localStorage persistence (last completed sim) ───────────────────── */

function loadSaved() {
  try {
    const data = JSON.parse(localStorage.getItem(STORE_KEY));
    return data &&
      data.v === SCHEMA_V &&
      Array.isArray(data.picks) &&
      data.picks.length
      ? data
      : null;
  } catch {
    return null;
  }
}

function slimPicks(picks) {
  return picks.map((pk) => ({
    pick: pk.pick,
    round: pk.round,
    team: pk.team,
    delta: Number.isFinite(pk.delta) ? pk.delta : null,
    tag: pk.tag || null,
    via: pk.via || null,
    player: {
      name: pk.player.name,
      position: pk.player.position,
      team: pk.player.team,
      grade: pk.player.grade,
      success_probability: pk.player.success_probability,
      espn_team_id: pk.player.espn_team_id,
    },
  }));
}

/* Plaintext export: "round.pick team player pos school grade" lines. */
function draftText(picks, orderLen) {
  const len = orderLen || 32;
  return picks
    .map((pk) => {
      const inRound = pk.pick - (pk.round - 1) * len;
      return [
        `${pk.round}.${String(inRound).padStart(2, '0')}`,
        pk.team,
        pk.player.name,
        pk.player.position || '',
        pk.player.team || '',
        pk.player.grade || '',
      ]
        .filter(Boolean)
        .join(' ');
    })
    .join('\n');
}

async function copyToClipboard(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path */
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    return true;
  } catch {
    return false;
  }
}

function persistResult(result) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(result));
  } catch {
    /* private mode etc. — the sim still works, it just won't persist */
  }
}

/* ─────────────────────────────────────────────────────────────────────── */

export default function MockDraft() {
  // phase: 'prompt' (saved sim found) | 'setup' | 'live' | 'done'
  const [phase, setPhase] = useState(() => (loadSaved() ? 'prompt' : 'setup'));
  const [saved] = useState(loadSaved);

  // player pool — null while loading, [] on failure
  const [poolRows, setPoolRows] = useState(null);
  // true when the pool came from the owner-curated '27 big board
  const [boardPowered, setBoardPowered] = useState(false);

  // clipboard feedback — 'draft' | 'haul' | null
  const [copied, setCopied] = useState(null);

  // setup
  const [draftYear, setDraftYear]   = useState(2027);
  const [rounds, setRounds]         = useState(1);
  const [userTeam, setUserTeam]     = useState(null);
  const [controlAll, setControlAll] = useState(false);
  const [order, setOrder]           = useState(() => DRAFT_ORDER_2027.slice());

  // live sim
  const [sim, setSim]         = useState(null);
  const [simming, setSimming] = useState(null); // null | 'user' | 'end'
  const [search, setSearch]   = useState('');
  const [posFilter, setPosFilter] = useState('ALL');

  // completed result being displayed (from a finished sim or localStorage)
  const [view, setView] = useState(null);

  /* ── Player pool for the selected class: that year's big board when it
     exists (owner order first, model-ranked rest after), else the
     server-filtered eligible class board-sorted. ── */
  useEffect(() => {
    let alive = true;
    setPoolRows(null);
    setBoardPowered(false);
    (async () => {
      try {
        const r = await anonFetch(`/api/big-board?class=${draftYear}`);
        if (r.ok) {
          const data = await r.json();
          if (data && Array.isArray(data.board) && data.board.length) {
            const pool = buildPoolFromBoard(data.board, data.rest);
            if (pool.length && alive) {
              setPoolRows(pool);
              setBoardPowered(true);
              return;
            }
          }
        }
      } catch {
        /* 404 / network — fall back to the model board */
      }
      try {
        const r = await anonFetch(`/api/prospects?draft_class=${draftYear}&limit=2000`);
        const data = await r.json();
        if (alive) setPoolRows(buildPool(data.prospects, draftYear));
      } catch {
        if (alive) setPoolRows([]);
      }
    })();
    return () => { alive = false; };
  }, [draftYear]);

  /* ── Start / restart ── */
  const startDraft = useCallback((cfg) => {
    if (!poolRows || poolRows.length === 0) return;
    const team = cfg.userTeam;
    const s = createDraft({
      order: cfg.order || DRAFT_ORDER_2027,
      rounds: cfg.rounds,
      userTeam: team,
      pool: poolRows,
      seed: newSeed(),
    });
    setSim(s);
    setView(null);
    setSearch('');
    setPosFilter('ALL');
    setPhase('live');
    // CPU teams run until the user is on the clock; in control-every-pick
    // mode the user is on the clock immediately.
    setSimming(team === CONTROL_ALL ? null : 'user');
  }, [poolRows]);

  /* ── Sim loop: one CPU pick per tick while "simming" ── */
  useEffect(() => {
    if (!simming || !sim) return undefined;
    if (isDone(sim) || (simming === 'user' && isUserOnClock(sim))) {
      setSimming(null);
      return undefined;
    }
    const reduced =
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const delay = reduced ? 10 : simming === 'end' ? 45 : 90;
    const t = setTimeout(() => setSim((s) => (s ? advanceCpu(s) : s)), delay);
    return () => clearTimeout(t);
  }, [simming, sim]);

  /* ── Completion: freeze the result + persist ── */
  useEffect(() => {
    if (phase !== 'live' || !sim || !isDone(sim)) return;
    const result = {
      v: SCHEMA_V,
      year: draftYear,
      rounds: sim.rounds,
      userTeam: sim.userTeam,
      orderLen: sim.order.length,
      completedAt: new Date().toISOString(),
      picks: slimPicks(sim.picks),
    };
    setView(result);
    persistResult(result);
    setSimming(null);
    setPhase('done');
  }, [sim, phase, draftYear]);

  /* ── User pick ── */
  const draftPlayer = useCallback((poolIdx) => {
    if (!sim || !isUserOnClock(sim)) return;
    setSim(commitPick(sim, poolIdx));
    setSearch('');
    setPosFilter('ALL');
    // CPU resumes toward the next user pick; in control-every-pick mode
    // the user stays on the clock for every selection.
    if (sim.userTeam !== CONTROL_ALL) setSimming('user');
  }, [sim]);

  /* ── Available players for the on-the-clock panel ── */
  const available = useMemo(() => {
    if (!sim) return [];
    let list = sim.pool.map((p, i) => ({ p, i }));
    if (posFilter !== 'ALL') {
      list = list.filter((x) => posGroup(x.p.position) === posFilter);
    }
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (x) =>
          (x.p.name || '').toLowerCase().includes(q) ||
          (x.p.team || '').toLowerCase().includes(q)
      );
    }
    return list.slice(0, 40);
  }, [sim, posFilter, search]);

  /* ── Board data (live sim or a stored result) ── */
  const boardPicks = useMemo(
    () => (sim ? sim.picks : view ? view.picks : []),
    [sim, view]
  );
  const boardUserTeam = sim ? sim.userTeam : view ? view.userTeam : null;

  const roundGroups = useMemo(() => {
    const map = new Map();
    for (const pk of boardPicks) {
      if (!map.has(pk.round)) map.set(pk.round, []);
      map.get(pk.round).push(pk);
    }
    return [...map.entries()];
  }, [boardPicks]);

  const haul = useMemo(() => {
    if (!boardUserTeam || boardUserTeam === CONTROL_ALL) return [];
    return boardPicks.filter((pk) => pk.team === boardUserTeam);
  }, [boardPicks, boardUserTeam]);

  const onClock = sim && phase === 'live' && !simming && isUserOnClock(sim);
  const clockTeam = sim && !isDone(sim) ? currentTeam(sim) : null;
  const pickNo = sim ? currentPickNumber(sim) : 0;

  /* ── Trades ── */
  const downOffers = useMemo(
    () => (onClock ? tradeDownOffers(sim) : []),
    [onClock, sim]
  );
  const upOffers = useMemo(
    () =>
      sim && phase === 'live' && !simming && !onClock && !isDone(sim)
        ? tradeUpOffers(sim)
        : [],
    [sim, phase, simming, onClock]
  );

  const acceptOffer = useCallback((offer) => {
    setSim((s) => (s ? executeTrade(s, offer) : s));
    // trading down surrenders the current pick — resume toward the next one
    if (offer.dir === 'down') setSimming('user');
  }, []);

  /* ── Net value of the user's haul (sum of pick deltas) ── */
  const haulValue = useMemo(
    () =>
      haul.reduce(
        (s, pk) => s + (Number.isFinite(pk.delta) ? pk.delta : 0),
        0
      ),
    [haul]
  );

  /* ── Overall haul grade — the end-of-draft reveal (avg pick delta) ── */
  const haulGrade = useMemo(() => {
    const deltas = haul.map((pk) => pk.delta).filter(Number.isFinite);
    if (deltas.length === 0) return null;
    return deltaLetter(deltas.reduce((s, d) => s + d, 0) / deltas.length);
  }, [haul]);

  /* ── Plaintext exports ── */
  const exportOrderLen = sim
    ? sim.order.length
    : (view && view.orderLen) || DRAFT_ORDER_2027.length;

  const copyPicks = useCallback(
    async (picks, which) => {
      const ok = await copyToClipboard(draftText(picks, exportOrderLen));
      if (ok) {
        setCopied(which);
        setTimeout(() => setCopied(null), 1800);
      }
    },
    [exportOrderLen]
  );

  const moveTeam = useCallback((i, dir) => {
    setOrder((o) => {
      const j = i + dir;
      if (j < 0 || j >= o.length) return o;
      const next = o.slice();
      const tmp = next[i];
      next[i] = next[j];
      next[j] = tmp;
      return next;
    });
  }, []);

  const canStart = poolRows && poolRows.length > 0 && (controlAll || userTeam);

  const fmtSlots = (nums) => nums.map((n) => `#${n}`).join(' + ');

  const renderOffer = (offer) => (
    <div className="sim-trade-row" key={`${offer.dir}-${offer.team}-${offer.userGets[0]}`}>
      <img
        className="sim-trade-logo"
        src={nflLogoUrl(offer.team, 500)}
        alt=""
        loading="lazy"
      />
      <span className="sim-trade-team">{nick(offer.team)}</span>
      <span className="sim-trade-detail">
        send {fmtSlots(offer.userGives)} · get {fmtSlots(offer.userGets)}
      </span>
      <span className="sim-trade-val">
        {offer.give.toLocaleString()} → {offer.get.toLocaleString()} pts
      </span>
      <button
        className="sim-btn sim-btn-accent sim-trade-accept"
        disabled={!offer.fair}
        title={offer.fair ? undefined : 'Not close enough to chart value'}
        onClick={() => acceptOffer(offer)}
      >
        {offer.fair ? 'Accept' : 'Declined'}
      </button>
    </div>
  );

  /* ─────────────────────────────────────────────────────────────────── */

  return (
    <div className="mock-page">

      {/* ── Cinematic photo header (Leaderboard's lb-hero pattern) ── */}
      <header className="mock-hero">
        <div className="mock-hero-media" aria-hidden="true">
          <img src={HERO_IMG} alt="" />
        </div>
        <div className="mock-hero-scrim-x" aria-hidden="true" />
        <div className="mock-hero-scrim-y" aria-hidden="true" />
        <div className="mock-hero-inner">
          <div className="eyebrow">Mock draft</div>
          <h1 className="mock-title">Mock Draft Simulator</h1>
          <p className="mock-sub">
            Run a {draftYear} NFL mock against the DraftVision board — CPU
            picks weigh success probability, grade, and positional need.
            {draftYear >= 2029 &&
              ' Pools for this class are projected eligibility — early declares shift classes.'}
          </p>
          {boardPowered && (
            <span className="mock-board-badge">
              Powered by the ’{String(draftYear).slice(2)} Big Board
            </span>
          )}
          {phase === 'live' && sim && (
            <p className="mock-meta">
              Pick {Math.min(pickNo, sim.totalPicks)} of {sim.totalPicks}
              {' · '}Round {Math.ceil(Math.min(pickNo, sim.totalPicks) / sim.order.length)}
            </p>
          )}
          {phase === 'done' && view && (
            <p className="mock-meta">
              Draft complete · {view.picks.length} picks
              {view.completedAt && ` · ${new Date(view.completedAt).toLocaleDateString()}`}
            </p>
          )}
        </div>
      </header>

      <div className="mock-inner">

        {/* ── Saved-sim prompt ── */}
        {phase === 'prompt' && saved && (
          <div className="sim-resume">
            <p className="sim-resume-text">
              You finished a {saved.rounds}-round {saved.year || 2027} mock
              {saved.userTeam && saved.userTeam !== CONTROL_ALL && ` as the ${saved.userTeam}`}
              {saved.completedAt && ` on ${new Date(saved.completedAt).toLocaleDateString()}`}.
            </p>
            <div className="sim-btn-row">
              <button
                className="sim-btn sim-btn-accent"
                onClick={() => { setView(saved); setPhase('done'); }}
              >
                View last mock
              </button>
              <button className="sim-btn" onClick={() => setPhase('setup')}>
                Start a new draft
              </button>
            </div>
          </div>
        )}

        {/* ── Setup panel ── */}
        {phase === 'setup' && (
          <div className="sim-setup">
            <div className="sim-setup-row">
              <span className="sim-label">Class</span>
              <div className="sim-seg">
                {DRAFT_YEARS.map((y) => (
                  <button
                    key={y}
                    className={`sim-seg-btn${draftYear === y ? ' active' : ''}`}
                    onClick={() => setDraftYear(y)}
                  >
                    ’{String(y).slice(2)}
                  </button>
                ))}
              </div>
              {poolRows === null && (
                <span className="sim-hint">loading class pool…</span>
              )}
              {poolRows && (
                <span className="sim-hint">
                  {poolRows.length.toLocaleString()} eligible players
                </span>
              )}
            </div>
            <div className="sim-setup-row">
              <span className="sim-label">Rounds</span>
              <div className="sim-seg">
                {[1, 2, 3].map((r) => (
                  <button
                    key={r}
                    className={`sim-seg-btn${rounds === r ? ' active' : ''}`}
                    onClick={() => setRounds(r)}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>

            <div className="sim-setup-row">
              <span className="sim-label">Your team</span>
              <button
                className={`sim-chip${controlAll ? ' active' : ''}`}
                onClick={() => setControlAll((v) => !v)}
              >
                Control every pick
              </button>
            </div>

            {!controlAll && (
              <div className="sim-teams">
                {order.slice().sort((a, b) => a.localeCompare(b)).map((t) => (
                  <button
                    key={t}
                    className={`sim-team${userTeam === t ? ' active' : ''}`}
                    onClick={() => setUserTeam((prev) => (prev === t ? null : t))}
                    title={t}
                  >
                    <img src={nflLogoUrl(t, 500)} alt="" loading="lazy" />
                    <span>{nick(t)}</span>
                  </button>
                ))}
              </div>
            )}

            <details className="sim-order">
              <summary className="sim-order-summary">
                <span className="mock-import-chev">›</span>
                Draft order — projected order, use the arrows to reorder
              </summary>
              <div className="sim-order-list">
                {order.map((t, i) => (
                  <div key={t} className="sim-order-row">
                    <span className="sim-order-num">{i + 1}</span>
                    <img className="sim-order-logo" src={nflLogoUrl(t, 500)} alt="" loading="lazy" />
                    <span className="sim-order-name">{t}</span>
                    <span className="sim-order-arrows">
                      <button
                        className="sim-arrow"
                        aria-label={`Move ${t} up`}
                        disabled={i === 0}
                        onClick={() => moveTeam(i, -1)}
                      >
                        ↑
                      </button>
                      <button
                        className="sim-arrow"
                        aria-label={`Move ${t} down`}
                        disabled={i === order.length - 1}
                        onClick={() => moveTeam(i, 1)}
                      >
                        ↓
                      </button>
                    </span>
                  </div>
                ))}
              </div>
              <button
                className="sim-btn sim-order-reset"
                onClick={() => setOrder(DRAFT_ORDER_2027.slice())}
              >
                Reset order
              </button>
            </details>

            <div className="sim-btn-row">
              <button
                className="sim-btn sim-btn-accent sim-start sim-primary"
                disabled={!canStart}
                onClick={() =>
                  startDraft({
                    order,
                    rounds,
                    userTeam: controlAll ? CONTROL_ALL : userTeam,
                  })
                }
              >
                Start draft
              </button>
              {poolRows === null && (
                <span className="mock-status"><span className="mock-dot" />Loading player pool</span>
              )}
              {poolRows && poolRows.length === 0 && (
                <span className="mock-error">Player pool unavailable — try again shortly.</span>
              )}
              {poolRows && poolRows.length > 0 && !controlAll && !userTeam && (
                <span className="sim-hint">Pick a team (or control every pick) to start.</span>
              )}
            </div>
          </div>
        )}

        {/* ── Live controls bar ── */}
        {phase === 'live' && sim && (
          <div className="sim-bar">
            <div className="sim-bar-status">
              {clockTeam && nflLogoUrl(clockTeam) && (
                <img className="sim-bar-logo" src={nflLogoUrl(clockTeam, 500)} alt="" />
              )}
              {onClock ? (
                <span className="sim-bar-onclock">You're on the clock — pick {pickNo}</span>
              ) : simming ? (
                <span className="mock-status"><span className="mock-dot" />Simulating · {clockTeam} on the clock</span>
              ) : (
                <span className="sim-bar-idle">{clockTeam} on the clock — pick {pickNo}</span>
              )}
            </div>
            <div className="sim-btn-row">
              {simming ? (
                <button className="sim-btn" onClick={() => setSimming(null)}>Pause</button>
              ) : (
                <>
                  {sim.userTeam !== CONTROL_ALL && !onClock && (
                    <button className="sim-btn sim-btn-accent" onClick={() => setSimming('user')}>
                      Sim to my pick
                    </button>
                  )}
                  <button className="sim-btn" onClick={() => setSimming('end')}>Sim to end</button>
                </>
              )}
              <button
                className="sim-btn"
                onClick={() =>
                  startDraft({ order: sim.order, rounds: sim.rounds, userTeam: sim.userTeam })
                }
              >
                Restart
              </button>
            </div>
          </div>
        )}

        {/* ── Trade up (paused, CPU on the clock ahead of your pick) ── */}
        {upOffers.length > 0 && (
          <details className="sim-trades">
            <summary className="sim-trades-summary">
              <span className="mock-import-chev">›</span>
              Trade up — {upOffers.length} pick{upOffers.length === 1 ? '' : 's'} ahead of yours
            </summary>
            <div className="sim-trade-list">{upOffers.map(renderOffer)}</div>
          </details>
        )}

        {/* ── On the clock: searchable best-available ── */}
        {onClock && (
          <div className="sim-clock">
            {downOffers.length > 0 && (
              <details className="sim-trades sim-trades-clock">
                <summary className="sim-trades-summary">
                  <span className="mock-import-chev">›</span>
                  Trade down — offers for pick {pickNo}
                </summary>
                <div className="sim-trade-list">{downOffers.map(renderOffer)}</div>
              </details>
            )}
            <div className="sim-clock-controls">
              <input
                className="sim-search"
                placeholder="Search best available…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              <div className="sim-chips">
                {POS_TABS.map((t) => (
                  <button
                    key={t}
                    className={`sim-chip${posFilter === t ? ' active' : ''}`}
                    onClick={() => setPosFilter(t)}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            <div className="sim-av">
              {available.map(({ p, i }) => (
                <button key={`${p.name}-${i}`} className="sim-av-row" onClick={() => draftPlayer(i)}>
                  {p.espn_team_id ? (
                    <img
                      className="sim-av-logo"
                      src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${p.espn_team_id}.png`}
                      alt=""
                      loading="lazy"
                      onError={(e) => { e.target.style.visibility = 'hidden'; }}
                    />
                  ) : (
                    <span className="sim-av-logo" />
                  )}
                  <span className="sim-av-name">
                    <span className="sim-av-player">{p.name}</span>
                    <span className="sim-av-school">{p.team}</span>
                  </span>
                  <span className="sim-av-pos">{p.position}</span>
                  <span className="sim-av-grade"><span>{p.grade || '—'}</span></span>
                  <span className="sim-av-prob">{fmtProb(p.success_probability)}</span>
                  <span className="sim-av-draft">Draft</span>
                </button>
              ))}
              {available.length === 0 && (
                <p className="sim-av-empty">No available players match that filter.</p>
              )}
            </div>
          </div>
        )}

        {/* ── Result summary ── */}
        {phase === 'done' && view && (
          <div className="sim-summary">
            {boardUserTeam && boardUserTeam !== CONTROL_ALL ? (
              <>
                <div className="sim-summary-head">
                  {nflLogoUrl(boardUserTeam) && (
                    <img className="sim-summary-logo" src={nflLogoUrl(boardUserTeam, 500)} alt="" />
                  )}
                  <div>
                    <div className="sim-summary-title">Your haul — {boardUserTeam}</div>
                    <div className="sim-summary-sub">
                      {haul.length} pick{haul.length === 1 ? '' : 's'} over {view.rounds} round{view.rounds === 1 ? '' : 's'}
                      {' · '}net value {haulValue >= 0 ? '+' : ''}{haulValue}
                    </div>
                  </div>
                  {haulGrade && (
                    <div
                      className="sim-haul-overall"
                      title="Average pick value vs the DraftVision board"
                    >
                      <span className="sim-haul-overall-label">Draft grade</span>
                      <span className={`sim-haul-overall-letter is-${haulGrade.toLowerCase()}`}>
                        {haulGrade}
                      </span>
                    </div>
                  )}
                </div>
                <div className="sim-summary-rows">
                  {haul.map((pk) => {
                    const letter = deltaLetter(pk.delta);
                    return (
                      <div key={pk.pick} className="sim-summary-row">
                        <span className="sim-summary-pick">{pk.pick}</span>
                        <span className="sim-summary-name">
                          {pk.player.name}
                          {pk.tag && (
                            <span className={`mock-tag ${pk.tag === 'STEAL' ? 'is-steal' : 'is-reach'}`}>
                              {pk.tag}
                            </span>
                          )}
                        </span>
                        <span className="sim-summary-school">{pk.player.team}</span>
                        <span className="sim-av-pos">{pk.player.position}</span>
                        <span className="sim-av-grade"><span>{pk.player.grade || '—'}</span></span>
                        <span
                          className={`sim-haul-letter${letter ? ` is-${letter.toLowerCase()}` : ''}`}
                          title={Number.isFinite(pk.delta)
                            ? `Value ${pk.delta >= 0 ? '+' : ''}${pk.delta} vs board rank`
                            : undefined}
                        >
                          {letter || '—'}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <div className="sim-summary-head">
                <div>
                  <div className="sim-summary-title">Draft complete</div>
                  <div className="sim-summary-sub">
                    You called all {view.picks.length} picks.
                  </div>
                </div>
              </div>
            )}
            <div className="sim-btn-row">
              <button
                className="sim-btn sim-btn-accent sim-primary"
                disabled={!poolRows || poolRows.length === 0}
                onClick={() =>
                  startDraft({
                    order: sim ? sim.order : order,
                    rounds: view.rounds,
                    userTeam: view.userTeam,
                  })
                }
              >
                Run it back
              </button>
              <button
                className="sim-btn"
                onClick={() => { setSim(null); setView(null); setPhase('setup'); }}
              >
                New draft
              </button>
              <button
                className="sim-btn"
                onClick={() => copyPicks(view.picks, 'draft')}
              >
                {copied === 'draft' ? 'Copied ✓' : 'Copy draft'}
              </button>
              {haul.length > 0 && (
                <button
                  className="sim-btn"
                  onClick={() => copyPicks(haul, 'haul')}
                >
                  {copied === 'haul' ? 'Copied ✓' : 'Copy my haul'}
                </button>
              )}
            </div>
          </div>
        )}

        {/* ── Board ── */}
        {roundGroups.map(([rnd, picks]) => (
          <section key={rnd} className="mock-round">
            <div className="mock-round-head">
              <span className="mock-round-label">Round {rnd}</span>
              <span className="mock-round-rule" />
              <span className="mock-round-count">{picks.length} picks</span>
            </div>
            <div className="mock-board">
              {picks.map((pk) => (
                <MockRow
                  key={pk.pick}
                  pick={pk.pick}
                  name={pk.player.name}
                  school={pk.player.team}
                  nflTeam={pk.team}
                  position={pk.player.position}
                  grade={pk.player.grade}
                  tag={pk.tag}
                  via={pk.via}
                  highlight={
                    boardUserTeam !== CONTROL_ALL && pk.team === boardUserTeam
                  }
                />
              ))}
            </div>
          </section>
        ))}

        {/* ── Legacy image-import flow (collapsed) ── */}
        <ImportMock />
      </div>
    </div>
  );
}

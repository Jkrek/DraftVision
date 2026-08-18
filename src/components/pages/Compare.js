import React, {
  useCallback, useEffect, useMemo, useRef, useState,
} from 'react';
import { useSearchParams } from 'react-router-dom';
import { anonFetch } from '../../lib/api';
import './Compare.css';

// ── Helpers ────────────────────────────────────────────────────────────────

// Shareable slug: "name-team" lowercased, spaces→'-', strip non-alnum except dash.
function slugify(name, team) {
  return `${name || ''}-${team || ''}`
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9-]/g, '');
}

// Grade letter → tier class for the mono-ramp pill tinting (a/b/c/d).
function gradeTier(grade) {
  const c = (grade || '').charAt(0).toUpperCase();
  return c === 'A' ? 'a' : c === 'B' ? 'b' : c === 'C' ? 'c' : c === 'D' ? 'd' : '';
}

const clampPct = (v) => Math.max(0, Math.min(100, Number(v) || 0));

const logoUrl = (espnTeamId) => (
  espnTeamId ? `https://a.espncdn.com/i/teamlogos/ncaa/500/${espnTeamId}.png` : null
);

// Resolve a URL slug back to a college search row. The search endpoint
// substring-matches raw names ("jamarr" won't hit "Ja'Marr"), so try each
// slug token in turn until one surfaces a row whose slug matches exactly.
async function resolveSlug(slug) {
  const tokens = slug.split('-').filter((t) => t.length >= 3);
  const queries = tokens.length > 0 ? tokens : slug.split('-').filter((t) => t.length >= 2);
  for (const q of queries.slice(0, 6)) {
    try {
      const res = await anonFetch(`/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      const rows = Array.isArray(data.players) ? data.players : [];
      const hit = rows.find((p) => p.kind === 'college' && slugify(p.name, p.team) === slug);
      if (hit) return hit;
    } catch {
      // network hiccup on one token — keep trying the rest
    }
  }
  return null;
}

const emptySlot = () => ({
  query: '',
  results: [],
  open: false,
  player: null,      // selected /search row (kind: "college")
  prediction: null,  // /predict payload
  predicting: false,
  error: null,
});

// ── Main component ─────────────────────────────────────────────────────────
export default function Compare() {
  const [slots, setSlots] = useState({ a: emptySlot(), b: emptySlot() });
  const [hydrating, setHydrating] = useState(false);

  const [searchParams, setSearchParams] = useSearchParams();
  const didHydrate = useRef(false);
  const boxRefs = { a: useRef(null), b: useRef(null) };
  const acTimers = useRef({ a: null, b: null });

  const setSlot = useCallback((side, patch) => {
    setSlots((s) => ({ ...s, [side]: { ...s[side], ...(typeof patch === 'function' ? patch(s[side]) : patch) } }));
  }, []);

  // ── prediction runner ─────────────────────────────────────────
  const predictFor = useCallback(async (side, player) => {
    setSlot(side, { predicting: true, prediction: null, error: null });
    try {
      const res = await anonFetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: player.name,
          ...(player.position && player.position !== 'UNK' ? { position: player.position } : {}),
          ...(player.team ? { team: player.team } : {}),
          // espn_id (when the search row carries it) unlocks verified ESPN stats
          ...(player.espn_id ? { espn_id: player.espn_id } : {}),
        }),
        signal: AbortSignal.timeout(20000),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed.');
      // Ignore a stale response if the slot changed players meanwhile.
      setSlots((s) => (
        s[side].player && s[side].player.name === player.name
          ? { ...s, [side]: { ...s[side], prediction: data, predicting: false } }
          : s
      ));
    } catch (err) {
      setSlots((s) => (
        s[side].player && s[side].player.name === player.name
          ? { ...s, [side]: { ...s[side], predicting: false, error: err.message || 'Could not reach backend.' } }
          : s
      ));
    }
  }, [setSlot]);

  const pickPlayer = useCallback((side, player) => {
    setSlots((s) => ({ ...s, [side]: { ...emptySlot(), player } }));
    predictFor(side, player);
  }, [predictFor]);

  const clearSlot = useCallback((side) => {
    setSlots((s) => ({ ...s, [side]: emptySlot() }));
  }, []);

  // ── hydrate once from /compare?a=&b= ─────────────────────────
  useEffect(() => {
    if (didHydrate.current) return;
    didHydrate.current = true;
    const a = (searchParams.get('a') || '').trim();
    const b = (searchParams.get('b') || '').trim();
    if (!a && !b) return;
    setHydrating(true);
    (async () => {
      const [pa, pb] = await Promise.all([
        a ? resolveSlug(a) : null,
        b ? resolveSlug(b) : null,
      ]);
      if (pa) pickPlayer('a', pa);
      if (pb) pickPlayer('b', pb);
      setHydrating(false);
    })();
  }, [searchParams, pickPlayer]);

  // ── keep the URL in sync with the current selection ──────────
  useEffect(() => {
    if (!didHydrate.current || hydrating) return;
    const next = new URLSearchParams();
    if (slots.a.player) next.set('a', slugify(slots.a.player.name, slots.a.player.team));
    if (slots.b.player) next.set('b', slugify(slots.b.player.name, slots.b.player.team));
    if (next.toString() !== searchParams.toString()) {
      setSearchParams(next, { replace: true });
    }
  }, [slots.a.player, slots.b.player, hydrating, searchParams, setSearchParams]);

  // ── autocomplete ─────────────────────────────────────────────
  const handleQuery = (side) => (e) => {
    const val = e.target.value;
    setSlot(side, { query: val });
    clearTimeout(acTimers.current[side]);
    if (val.length < 2) { setSlot(side, { results: [], open: false }); return; }
    acTimers.current[side] = setTimeout(async () => {
      try {
        const res = await anonFetch(`/search?q=${encodeURIComponent(val)}`);
        const data = await res.json();
        setSlot(side, { results: Array.isArray(data.players) ? data.players : [], open: true });
      } catch { setSlot(side, { results: [] }); }
    }, 180);
  };

  // close dropdowns on outside click
  const refA = boxRefs.a; const refB = boxRefs.b;
  useEffect(() => {
    const handler = (e) => {
      if (refA.current && !refA.current.contains(e.target)) {
        setSlots((s) => (s.a.open ? { ...s, a: { ...s.a, open: false } } : s));
      }
      if (refB.current && !refB.current.contains(e.target)) {
        setSlots((s) => (s.b.open ? { ...s, b: { ...s.b, open: false } } : s));
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [refA, refB]);

  // ── derived: verdict + aligned factor rows ───────────────────
  const predA = slots.a.prediction;
  const predB = slots.b.prediction;

  const probOf = (slot) => {
    const p = slot.prediction?.success_probability;
    if (typeof p === 'number') return p;
    const q = slot.player?.success_probability;
    return typeof q === 'number' ? q : null;
  };
  const verdict = useMemo(() => {
    if (!predA || !predB) return null;
    const pa = predA.success_probability;
    const pb = predB.success_probability;
    if (typeof pa !== 'number' || typeof pb !== 'number') return null;
    const delta = pa - pb;
    if (Math.abs(delta) < 0.05) {
      return { even: true, text: 'Model rates this matchup even.' };
    }
    const winner = delta > 0
      ? (predA.resolved_name || slots.a.player?.name)
      : (predB.resolved_name || slots.b.player?.name);
    return {
      even: false,
      winner,
      delta: Math.abs(delta),
      side: delta > 0 ? 'a' : 'b',
    };
  }, [predA, predB, slots.a.player, slots.b.player]);

  // Shared factor rows, aligned by feature name; ordered by the larger of the
  // two importances so the loudest signals sit on top.
  const factorRows = useMemo(() => {
    if (!predA && !predB) return [];
    const map = new Map();
    (Array.isArray(predA?.top_factors) ? predA.top_factors : []).forEach((f) => {
      map.set(f.feature, { feature: f.feature, a: f, b: null });
    });
    (Array.isArray(predB?.top_factors) ? predB.top_factors : []).forEach((f) => {
      const row = map.get(f.feature) || { feature: f.feature, a: null, b: null };
      row.b = f;
      map.set(f.feature, row);
    });
    return [...map.values()].sort((x, y) => (
      Math.max(y.a?.importance || 0, y.b?.importance || 0)
      - Math.max(x.a?.importance || 0, x.b?.importance || 0)
    ));
  }, [predA, predB]);

  const bothPicked = Boolean(slots.a.player && slots.b.player);

  // ── render pieces ────────────────────────────────────────────
  const renderSearch = (side) => {
    const slot = slots[side];
    const label = side === 'a' ? 'Player A' : 'Player B';
    if (slot.player) {
      const p = slot.player;
      return (
        <div className="cmp-picked" ref={boxRefs[side]}>
          {p.espn_team_id && (
            <img className="cmp-picked-logo" src={logoUrl(p.espn_team_id)} alt="" loading="lazy" />
          )}
          <span className="cmp-picked-main">
            <span className="cmp-picked-name">{p.name}</span>
            <span className="cmp-picked-sub">{[p.position, p.team].filter(Boolean).join(' · ')}</span>
          </span>
          <button type="button" className="cmp-change" onClick={() => clearSlot(side)}>
            Change
          </button>
        </div>
      );
    }
    return (
      <div className="cmp-search-box" ref={boxRefs[side]}>
        <input
          type="text"
          className="cmp-input"
          value={slot.query}
          onChange={handleQuery(side)}
          placeholder={`${label} — search FBS players`}
          aria-label={label}
        />
        {slot.open && slot.results.length > 0 && (
          <div className="cmp-ac" role="listbox">
            {slot.results.map((p, i) => {
              const selectable = p.kind === 'college';
              const sub = [p.position, p.kind === 'hs' ? p.school : p.team].filter(Boolean).join(' · ');
              const offNote = p.kind === 'hs' ? 'High school' : 'Pro / legacy';
              return (
                <button
                  type="button"
                  key={`${side}-${p.kind}-${p.name}-${p.team || p.school || i}`}
                  className={`cmp-ac-row${selectable ? '' : ' cmp-ac-row-off'}`}
                  disabled={!selectable}
                  title={selectable ? undefined : `${offNote} — only college prospects can be compared.`}
                  onMouseDown={() => { if (selectable) pickPlayer(side, p); }}
                >
                  <span className="cmp-ac-main">
                    <span className="cmp-ac-name">{p.name}</span>
                    <span className="cmp-ac-sub">
                      {sub}
                      {selectable ? '' : ` · ${offNote}`}
                    </span>
                  </span>
                  {p.grade && <span className="cmp-ac-grade">{p.grade}</span>}
                </button>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  const renderColumn = (side) => {
    const slot = slots[side];
    const p = slot.player;
    if (!p) return null;
    const pred = slot.prediction;
    const name = pred?.resolved_name || p.name;
    const pos = pred?.predicted_position || p.position || '—';
    const team = pred?.stats?.team || p.team || '';
    const grade = pred?.prospect_grade || p.grade || null;
    const tier = gradeTier(grade);
    const prob = probOf(slot);
    const stats = pred?.stats || {};

    const bars = [
      {
        label: 'Production',
        raw: stats.production_score,
        fmt: (v) => `${Number(v).toFixed(1)} / 100`,
        pct: (v) => clampPct(v),
      },
      {
        label: 'Speed',
        raw: stats.combine_speed_score,
        fmt: (v) => `${Number(v).toFixed(1)} / 100`,
        pct: (v) => clampPct(v),
      },
      {
        label: 'Games',
        raw: (Number.isFinite(Number(stats.games_played)) && Number(stats.games_played) > 0)
          ? Number(stats.games_played) : null,
        fmt: (v) => `${v} / 17`,
        pct: (v) => clampPct((v / 17) * 100),
      },
    ];

    return (
      <div className={`cmp-col${verdict && !verdict.even && verdict.side === side ? ' cmp-col-lead' : ''}`}>
        <div className="cmp-col-head">
          {p.espn_team_id ? (
            <img className="cmp-logo" src={logoUrl(p.espn_team_id)} alt="" loading="lazy" />
          ) : (
            <span className="cmp-logo cmp-logo-blank" aria-hidden="true" />
          )}
          <div className="cmp-id">
            <div className="cmp-name">{name}</div>
            <div className="cmp-sub">{[pos, team].filter(Boolean).join(' · ')}</div>
          </div>
          {grade && <span className={`cmp-grade${tier ? ` cmp-grade-${tier}` : ''}`}>{grade}</span>}
        </div>

        <div className="cmp-prob-block">
          <div className="cmp-metric-label">Success probability</div>
          <div className="cmp-prob">
            <span className="cmp-prob-value">{prob !== null ? prob : '—'}</span>
            {prob !== null && <span className="cmp-prob-unit">%</span>}
          </div>
        </div>

        <div className="cmp-proj">
          <span className="cmp-metric-label">Projection</span>
          <span className="cmp-proj-value">
            {slot.predicting ? '…' : (pred?.draft_grade || '—')}
          </span>
        </div>

        <div className="cmp-minis">
          {bars.map((b) => (
            <div className="cmp-mini" key={b.label}>
              <div className="cmp-mini-head">
                <span>{b.label}</span>
                {b.raw != null
                  ? <span className="cmp-mini-val">{b.fmt(b.raw)}</span>
                  : <span className="cmp-mini-none">No verified data</span>}
              </div>
              <span className="cmp-mini-track">
                {b.raw != null && (
                  <span className="cmp-mini-bar" style={{ width: `${b.pct(b.raw)}%` }} />
                )}
              </span>
            </div>
          ))}
        </div>

        {slot.predicting && (
          <div className="cmp-status" role="status">
            <span className="cmp-status-dot" aria-hidden="true" />
            Running the model…
          </div>
        )}
        {slot.error && !slot.predicting && (
          <div className="cmp-error" role="alert">
            {slot.error}{' '}
            <button type="button" className="cmp-retry" onClick={() => predictFor(side, p)}>
              Retry
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderFactorSide = (f, sideCls) => {
    if (!f) {
      return <div className={`cmp-fside ${sideCls}`}><span className="cmp-fval cmp-fval-none">—</span></div>;
    }
    const neg = f.direction === 'negative'
      || (typeof f.contribution === 'number' && f.contribution < 0);
    const signed = typeof f.contribution === 'number';
    const sign = signed ? (neg ? '−' : '+') : '';
    return (
      <div className={`cmp-fside ${sideCls}`}>
        <span className={`cmp-fval${signed ? (neg ? ' neg' : ' pos') : ''}`}>
          {sign}{f.importance}%
        </span>
        <span className="cmp-ftrack">
          <span
            className={`cmp-fbar${signed ? (neg ? ' neg' : ' pos') : ''}`}
            style={{ width: `${clampPct(f.importance)}%` }}
          />
        </span>
      </div>
    );
  };

  const renderComps = (side) => {
    const slot = slots[side];
    const comps = Array.isArray(slot.prediction?.historical_comps)
      ? slot.prediction.historical_comps.slice(0, 3) : [];
    return (
      <div className="cmp-comp-col">
        <div className="cmp-panel-label">
          {slot.player ? `${slot.player.name} — closest comps` : 'Closest comps'}
        </div>
        {comps.length > 0 ? comps.map((c, i) => (
          <div className="cmp-comp-row" key={`${side}-${c.name}-${i}`}>
            <div>
              <div className="cmp-comp-name">{c.name}</div>
              <div className="cmp-comp-note">{[c.position, c.outcome].filter(Boolean).join(' · ')}</div>
            </div>
            <span className="cmp-comp-sim">{c.similarity}%</span>
          </div>
        )) : (
          <p className="cmp-note">
            {slot.predicting ? 'Running the model…' : 'No historical comps were returned.'}
          </p>
        )}
      </div>
    );
  };

  // ── render ────────────────────────────────────────────────────
  return (
    <main className="cmp-page">
      <div className="cmp-shell">
        <div className="eyebrow">Head to head</div>
        <h1 className="cmp-title">Compare</h1>
        <p className="cmp-lede">
          Pick two college prospects and the ensemble scores them side by side —
          probability, projection, factor attributions and historical comps.
        </p>

        {/* ── Two search boxes ── */}
        <div className="cmp-search-grid">
          {renderSearch('a')}
          <div className="cmp-vs" aria-hidden="true">vs</div>
          {renderSearch('b')}
        </div>

        {hydrating && (
          <div className="cmp-status" role="status">
            <span className="cmp-status-dot" aria-hidden="true" />
            Loading shared comparison…
          </div>
        )}

        {!bothPicked && !hydrating && (
          <p className="cmp-empty">
            {slots.a.player || slots.b.player
              ? 'Pick the second player to run the comparison.'
              : 'Only college prospects are comparable — high-school and pro rows are shown for context but can’t be selected.'}
          </p>
        )}

        {/* ── Board ── */}
        {bothPicked && (
          <div className="cmp-board">
            <div className="cmp-cols">
              {renderColumn('a')}
              {renderColumn('b')}
            </div>

            {/* Verdict strip */}
            {verdict && (
              <div className="cmp-verdict" role="status">
                {verdict.even ? (
                  <span>{verdict.text}</span>
                ) : (
                  <span>
                    Model favors <strong>{verdict.winner}</strong> by{' '}
                    <span className="cmp-verdict-delta">{verdict.delta.toFixed(1)} points</span>.
                  </span>
                )}
              </div>
            )}

            {/* Aligned factor rows */}
            {factorRows.length > 0 && (
              <div className="cmp-factors">
                <div className="cmp-panel-label">Top prediction factors — aligned by feature</div>
                {factorRows.map((f) => (
                  <div className="cmp-factor-row" key={f.feature}>
                    {renderFactorSide(f.a, 'cmp-fside-a')}
                    <div className="cmp-flabel">{f.feature}</div>
                    {renderFactorSide(f.b, 'cmp-fside-b')}
                  </div>
                ))}
                <p className="cmp-note">
                  Signed bars are per-player SHAP attributions — + pushes toward
                  NFL success, − pushes away. Unsigned bars fall back to global
                  model importances.
                </p>
              </div>
            )}

            {/* Historical comps, per column */}
            {(predA || predB) && (
              <div className="cmp-comps">
                {renderComps('a')}
                {renderComps('b')}
              </div>
            )}
          </div>
        )}

        {bothPicked && (
          <div className="cmp-meta">
            <span>
              {(predA?.model_used || predB?.model_used)
                ? 'XGBoost · two-model ML ensemble'
                : 'Rule-based fallback'}
            </span>
            <button
              type="button"
              className="cmp-clear-btn"
              onClick={() => { clearSlot('a'); clearSlot('b'); }}
            >
              Clear comparison
            </button>
          </div>
        )}
      </div>
    </main>
  );
}

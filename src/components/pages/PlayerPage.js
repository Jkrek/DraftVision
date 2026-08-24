import React, { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { anonFetch } from '../../lib/api';
import InfoTip from '../InfoTip';
import './PlayerPage.css';

// ── Shared slug helper ───────────────────────────────────────────────────
// `${name}-${team}` lowercased, spaces→'-', strip non-alphanumerics-except-
// dash. The list pages (Leaderboard / College Stars) inline this exact
// expression in their name-cell links — keep the encoding identical.
export function playerSlug(name, team) {
  return `${name}-${team}`
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

// Optional "trend": {"delta_prob", "delta_rank"} on /api/prospects rows.
function trendOf(p) {
  const t = p && p.trend;
  if (!t || typeof t !== 'object') return null;
  const delta = Number(t.delta_prob);
  if (!Number.isFinite(delta) || delta === 0) return null;
  return delta;
}

// "’28" from 2028; null when the class is unknown.
function shortClass(year) {
  const y = Number(year);
  return Number.isFinite(y) && y > 2000 ? `’${String(y).slice(2)}` : null;
}

// The slug is lossy (punctuation stripped), so we can't just un-dash it.
// Query /api/prospects with a few progressively looser guesses pulled from
// the slug's tokens, then exact-match rows by re-encoding name+team.
async function resolveFromSlug(slug) {
  const tokens = slug.split('-').filter(Boolean);
  const longest = [...tokens].sort((a, b) => b.length - a.length)[0];
  const attempts = [
    tokens.length >= 2 ? [`${tokens[0]} ${tokens[1]}`, 20] : null, // "first last"
    tokens.length >= 2 ? [tokens[1], 20] : null,                   // surname
    tokens[0] ? [tokens[0], 20] : null,                            // first name
    longest ? [longest, 20] : null,                                // most selective token
    // Last resort — punctuated names ("Que'Sean") lose characters in the slug
    // and stop substring-matching. The slug's LAST token is a team word (teams
    // are punctuation-free), so a wide query on it always contains the player.
    tokens.length ? [tokens[tokens.length - 1], 2000] : null,
  ];
  const seen = new Set();
  for (const attempt of attempts) {
    if (!attempt) continue;
    const [q, limit] = attempt;
    const key = `${q}|${limit}`;
    if (seen.has(key)) continue;
    seen.add(key);
    try {
      const r = await anonFetch(`/api/prospects?q=${encodeURIComponent(q)}&limit=${limit}`);
      if (!r.ok) continue;
      const d = await r.json();
      const rows = Array.isArray(d.prospects) ? d.prospects : [];
      const hit = rows.find((p) => playerSlug(p.name || '', p.team || '') === slug);
      if (hit) return hit;
    } catch {
      /* try the next guess */
    }
  }
  return null;
}

export default function PlayerPage() {
  const { slug } = useParams();

  const [player, setPlayer] = useState(null);
  const [status, setStatus] = useState('loading'); // loading | ready | notfound
  const [why, setWhy] = useState({ loading: true, factors: null, comps: null, error: false });
  const [boardRank, setBoardRank] = useState(null);

  // Resolve the player from the slug.
  useEffect(() => {
    let cancelled = false;
    setStatus('loading');
    setPlayer(null);
    setBoardRank(null);
    resolveFromSlug(slug || '').then((hit) => {
      if (cancelled) return;
      if (hit) {
        setPlayer(hit);
        setStatus('ready');
      } else {
        setStatus('notfound');
      }
    });
    return () => { cancelled = true; };
  }, [slug]);

  // "Why this grade" — POST /predict for SHAP factors + historical comps.
  useEffect(() => {
    if (!player) return;
    let cancelled = false;
    setWhy({ loading: true, factors: null, comps: null, error: false });
    anonFetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: player.name, position: player.position, team: player.team }),
    })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('predict failed'))))
      .then((d) => {
        if (cancelled) return;
        setWhy({
          loading: false,
          factors: Array.isArray(d.top_factors) ? d.top_factors : [],
          comps: Array.isArray(d.historical_comps) ? d.historical_comps : [],
          error: false,
        });
      })
      .catch(() => {
        if (!cancelled) setWhy({ loading: false, factors: null, comps: null, error: true });
      });
    return () => { cancelled = true; };
  }, [player]);

  // Big-board rank — only when the player is on the owner board for their class.
  useEffect(() => {
    if (!player) return;
    const cls = Number(player.draft_class);
    if (!Number.isFinite(cls) || !cls) return;
    let cancelled = false;
    anonFetch(`/api/big-board?class=${cls}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no board'))))
      .then((d) => {
        if (cancelled) return;
        const board = Array.isArray(d.board) ? d.board : [];
        const me = `${(player.name || '').toLowerCase()}|${(player.team || '').toLowerCase()}`;
        const idx = board.findIndex(
          (p) => `${(p.name || '').toLowerCase()}|${(p.team || '').toLowerCase()}` === me,
        );
        if (idx >= 0) setBoardRank(idx + 1);
      })
      .catch(() => { /* no owner board for this class — chip simply absent */ });
    return () => { cancelled = true; };
  }, [player]);

  // Stat bars — null means "no verified data", never a fake bar.
  const statBars = useMemo(() => {
    if (!player) return [];
    const gp = Number(player.games_played);
    return [
      {
        label: 'Production score',
        value: player.production_score != null ? `${Number(player.production_score).toFixed(1)} / 100` : null,
        pct: player.production_score != null ? clampPct(player.production_score) : null,
      },
      {
        label: 'Combine speed score',
        value: player.combine_speed_score != null ? `${Number(player.combine_speed_score).toFixed(1)} / 100` : null,
        pct: player.combine_speed_score != null ? clampPct(player.combine_speed_score) : null,
      },
      {
        label: 'Games played',
        value: Number.isFinite(gp) && player.games_played != null ? `${gp} / 17` : null,
        pct: Number.isFinite(gp) && player.games_played != null ? clampPct((gp / 17) * 100) : null,
      },
    ];
  }, [player]);

  if (status === 'loading') {
    return (
      <div className="pp-page">
        <div className="pp-state">
          <div className="pp-shimmer pp-shimmer-title" />
          <div className="pp-shimmer pp-shimmer-line" />
          <div className="pp-shimmer pp-shimmer-line short" />
        </div>
      </div>
    );
  }

  if (status === 'notfound') {
    return (
      <div className="pp-page">
        <div className="pp-state">
          <div className="pp-eyebrow">Player profile</div>
          <h1 className="pp-nf-title">Not found</h1>
          <p className="pp-nf-body">
            No graded prospect matches this link — the board may have been rebuilt since it was shared.
          </p>
          <Link className="pp-outline-btn" to="/leaderboard">Back to the leaderboard</Link>
        </div>
      </div>
    );
  }

  const trend = trendOf(player);
  const sp = player.success_probability;
  const tier = gradeTier(player.grade);
  const cls = shortClass(player.draft_class);
  const isEst = player.data_source && player.data_source !== 'espn_live';

  return (
    <div className="pp-page">

      {/* ── Cinematic header ── */}
      <header className="pp-hero">
        {player.espn_team_id && (
          <div className="pp-hero-logo-bg" aria-hidden="true">
            <img
              src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${player.espn_team_id}.png`}
              alt=""
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>
        )}
        <div className="pp-hero-scrim-x" aria-hidden="true" />
        <div className="pp-hero-scrim-y" aria-hidden="true" />
        <div className="pp-hero-inner">
          {/* ESPN headshot when published; the team logo is the fallback
              (and stays as the underline mark either way) */}
          {player.espn_id && (
            <img
              className="pp-hero-headshot"
              src={`https://a.espncdn.com/i/headshots/college-football/players/full/${player.espn_id}.png`}
              alt=""
              loading="lazy"
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          )}
          {player.espn_team_id && (
            <img
              className="pp-hero-logo"
              src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${player.espn_team_id}.png`}
              alt={`${player.team || ''} logo`}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          )}
          <div className="pp-hero-text">
            <div className="pp-eyebrow">Player profile</div>
            <h1 className="pp-name">
              {player.name}
              {isEst && (
                <span className="pp-est" title="No verified season stats — profile is estimated">EST</span>
              )}
            </h1>
            <div className="pp-meta">
              <span className="pp-meta-pos">{(player.position || '?').toUpperCase()}</span>
              <span className="pp-meta-sep" aria-hidden="true">·</span>
              <span>{player.team || 'Unknown school'}</span>
              {player.class_year && (
                <>
                  <span className="pp-meta-sep" aria-hidden="true">·</span>
                  <span>{player.class_year}</span>
                </>
              )}
              {cls && <span className="pp-class-chip">{cls} draft class</span>}
              {boardRank != null && cls && (
                <span className="pp-board-chip">№{boardRank} on the {cls} Board</span>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="pp-main">

        {/* ── Grade card ── */}
        <section className="pp-grade-card">
          <div className="pp-grade-cell">
            <div className="pp-cell-label">
              Grade
              <InfoTip text="Letter grade from percentile cutoffs of success probability across the whole board — A+ is the top 2%, A- the top 10%, C+ sits near the median. Graded on the curve of this class." />
            </div>
            <span className={`pp-grade-pill${tier ? ` pp-grade-${tier}` : ''}`}>
              {player.grade || '—'}
            </span>
          </div>
          <div className="pp-grade-cell">
            <div className="pp-cell-label">
              Success probability
              <InfoTip text="The ML ensemble’s calibrated chance this player becomes an NFL success — defined in training as a Pro Bowl, 3+ seasons as a starter, or 30+ career Approximate Value." />
            </div>
            <div className="pp-prob">
              {sp != null ? (
                <>
                  {Number(sp).toFixed(1)}<span className="pp-prob-unit">%</span>
                </>
              ) : '—'}
              {trend !== null && (
                <span className={`pp-trend ${trend > 0 ? 'up' : 'down'}`}>
                  {trend > 0 ? '▲' : '▼'} {Math.abs(trend).toFixed(1)} vs last board
                </span>
              )}
            </div>
          </div>
          <div className="pp-grade-cell">
            <div className="pp-cell-label">
              Draft projection
              <InfoTip text="Projected draft range from the ensemble’s draft-round classifier, trained on the 2000–2023 draft classes." place="top-left" />
            </div>
            <div className="pp-proj">{player.draft_grade || '—'}</div>
          </div>
        </section>

        {/* ── Stat bars ── */}
        <section className="pp-section">
          <h2 className="pp-section-title">Profile</h2>
          <div className="pp-stats">
            {statBars.map((s) => (
              <div className="pp-stat" key={s.label}>
                <div className="pp-stat-head">
                  <span>{s.label}</span>
                  {s.value != null && <span className="pp-stat-val">{s.value}</span>}
                </div>
                {s.pct != null ? (
                  <div className="pp-stat-track">
                    <span className="pp-stat-bar" style={{ width: `${s.pct}%` }} />
                  </div>
                ) : (
                  <div className="pp-stat-empty">no verified data</div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* ── Why this grade ── */}
        <section className="pp-section">
          <h2 className="pp-section-title">Why this grade</h2>
          {why.loading ? (
            <div className="pp-why-loading" aria-label="Computing model attribution">
              <div className="pp-shimmer pp-shimmer-line" />
              <div className="pp-shimmer pp-shimmer-line" />
              <div className="pp-shimmer pp-shimmer-line short" />
            </div>
          ) : why.error ? (
            <p className="pp-muted">
              Model attribution is unavailable right now — the grade above still stands.
            </p>
          ) : (
            <div className="pp-why">
              {why.factors && why.factors.length > 0 && (
                <div className="pp-factors">
                  {why.factors.map((f) => {
                    const signed = typeof f.contribution === 'number';
                    const negative = signed
                      ? f.contribution < 0
                      : f.direction === 'negative';
                    const width = clampPct(f.importance) / 2; // half-track each side
                    return (
                      <div className="pp-factor" key={f.feature}>
                        <span className="pp-factor-name">{f.feature}</span>
                        <span className="pp-factor-track">
                          <span className="pp-factor-axis" aria-hidden="true" />
                          <span
                            className={`pp-factor-bar ${negative ? 'neg' : 'pos'}`}
                            style={{ width: `${Math.max(1.5, width)}%` }}
                          />
                        </span>
                        <span className={`pp-factor-val ${negative ? 'neg' : 'pos'}`}>
                          {signed
                            ? `${f.contribution >= 0 ? '+' : '−'}${Math.abs(f.contribution).toFixed(2)}`
                            : `${Number(f.importance || 0).toFixed(0)}%`}
                        </span>
                      </div>
                    );
                  })}
                  <p className="pp-factor-note">
                    Signed model attributions — bars right of the axis pushed the grade up, left pushed it down.
                  </p>
                </div>
              )}
              {why.comps && why.comps.length > 0 && (
                <div className="pp-comps">
                  <div className="pp-cell-label">
                    Historical comps
                    <InfoTip text="Nearest real prospects from the 2000–2023 classes at the same position group — statistical distance over height, weight, speed, production and pedigree. Capped at 99%: no comp is a clone." />
                  </div>
                  {why.comps.map((c) => (
                    <div className="pp-comp" key={c.name}>
                      <span className="pp-comp-name">{c.name}</span>
                      <span className="pp-comp-pos">{c.position}</span>
                      <span className="pp-comp-sim">{c.similarity}% match</span>
                      {c.outcome && <span className="pp-comp-outcome">{c.outcome}</span>}
                    </div>
                  ))}
                </div>
              )}
              {(!why.factors || why.factors.length === 0) &&
                (!why.comps || why.comps.length === 0) && (
                  <p className="pp-muted">The model returned no attribution detail for this player.</p>
                )}
            </div>
          )}
        </section>

        {/* ── Film ── */}
        <section className="pp-section">
          <h2 className="pp-section-title">Film</h2>
          <a
            className="pp-outline-btn"
            href={`https://www.youtube.com/@jkrek/search?query=${encodeURIComponent(player.name || '')}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            Watch film on JK Football ▶
          </a>
        </section>

      </div>
    </div>
  );
}

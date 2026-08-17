import React, { useState, useEffect } from 'react';
import { anonFetch } from '../lib/api';
import './Ticker.css';

function toItem(p, i) {
  if (!p || typeof p !== 'object') return null;
  const name = p.name || p.player_name || p.player;
  if (!name) return null;
  const prob = p.success_probability ?? p.probability ?? p.prob;
  return {
    rank: String(i + 1).padStart(2, '0'),
    name,
    pos: p.position || p.pos || '',
    team: p.team || p.school || '',
    prob: typeof prob === 'number' ? prob.toFixed(1) : (prob || null),
  };
}

// /api/movers row -> ticker item; null when malformed or delta is zero/absent.
function toMover(m) {
  if (!m || typeof m !== 'object' || !m.name) return null;
  const delta = Number(m.delta_prob);
  if (!Number.isFinite(delta) || delta === 0) return null;
  return {
    name: m.name,
    pos: m.position || '',
    team: m.team || '',
    delta,
  };
}

function Ticker() {
  // mode: 'movers' (real deltas) | 'top' (fallback: top prospects)
  const [state, setState] = useState({ mode: null, items: [] });

  useEffect(() => {
    let cancelled = false;

    const loadFallback = () => {
      anonFetch('/api/prospects?limit=12')
        .then(r => (r.ok ? r.json() : Promise.reject(new Error('bad status'))))
        .then(data => {
          if (cancelled) return;
          const list = Array.isArray(data?.prospects)
            ? data.prospects
            : Array.isArray(data)
              ? data
              : [];
          const items = list.map(toItem).filter(Boolean);
          if (items.length) setState({ mode: 'top', items });
        })
        .catch(() => { /* render nothing on failure */ });
    };

    anonFetch('/api/movers')
      .then(r => (r.ok ? r.json() : Promise.reject(new Error('bad status'))))
      .then(data => {
        if (cancelled) return;
        const risers  = Array.isArray(data?.risers)  ? data.risers  : [];
        const fallers = Array.isArray(data?.fallers) ? data.fallers : [];
        const movers = [...risers, ...fallers]
          .map(toMover)
          .filter(Boolean)
          .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
        if (movers.length) {
          setState({ mode: 'movers', items: movers });
        } else {
          // No prior snapshot yet ("since": null) — graceful fallback.
          loadFallback();
        }
      })
      .catch(() => { if (!cancelled) loadFallback(); });

    return () => { cancelled = true; };
  }, []);

  const { mode, items } = state;
  if (!items.length) return null;

  const renderRow = (hidden) => (
    <div className="ticker-row" aria-hidden={hidden || undefined}>
      {items.map((t, i) => (
        <div className="ticker-item" key={(hidden ? 'b' : 'a') + i}>
          {mode === 'top' && <span className="ticker-rank">{t.rank}</span>}
          <span className="ticker-name">{t.name}</span>
          {t.pos && <span className="ticker-pos">{t.pos}</span>}
          {t.team && <span className="ticker-team">{t.team}</span>}
          {mode === 'movers' ? (
            <span className={`ticker-delta ${t.delta > 0 ? 'up' : 'down'}`}>
              {t.delta > 0
                ? `▲ +${t.delta.toFixed(1)}`
                : `▼ ${Math.abs(t.delta).toFixed(1)}`}
            </span>
          ) : (
            t.prob != null && <span className="ticker-prob">{t.prob}%</span>
          )}
        </div>
      ))}
    </div>
  );

  return (
    <div className="ticker">
      <div className="ticker-label">
        <span className="ticker-dot" />
        <span className="ticker-label-text">
          {mode === 'movers' ? 'Board movers' : 'Top prospects'}
        </span>
      </div>
      <div className="ticker-viewport">
        <div className="ticker-track">
          {renderRow(false)}
          {renderRow(true)}
        </div>
      </div>
    </div>
  );
}

export default Ticker;

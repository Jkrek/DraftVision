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
    grade: p.grade || null,
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
  // mode: 'board' (owner big board + elite grades) | 'movers' | 'top'
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

    const loadMovers = () => {
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
            loadFallback();
          }
        })
        .catch(() => { if (!cancelled) loadFallback(); });
    };

    // Primary: the owner's Big Board (engagement picks) padded with elite
    // grades not on it. Falls back to movers, then top prospects.
    anonFetch('/api/big-board?class=2027')
      .then(r => (r.ok ? r.json() : Promise.reject(new Error('bad status'))))
      .then(data => {
        if (cancelled) return;
        const board = Array.isArray(data?.board) ? data.board : [];
        if (!board.length) { loadMovers(); return; }
        const onBoard = new Set(
          board.map(p => `${(p.name || '').toLowerCase()}|${(p.team || '').toLowerCase()}`)
        );
        const boardItems = board.slice(0, 20).map((p, i) => {
          const it = toItem(p, i);
          if (it) it.boardRank = i + 1;
          return it;
        }).filter(Boolean);
        const elite = (Array.isArray(data?.rest) ? data.rest : [])
          .filter(p =>
            (p.grade === 'A+' || p.grade === 'A') &&
            !onBoard.has(`${(p.name || '').toLowerCase()}|${(p.team || '').toLowerCase()}`)
          )
          .slice(0, 10)
          .map(toItem)
          .filter(Boolean);
        setState({ mode: 'board', items: [...boardItems, ...elite] });
      })
      .catch(() => { if (!cancelled) loadMovers(); });

    return () => { cancelled = true; };
  }, []);

  const { mode, items } = state;
  if (!items.length) return null;

  // One full loop scrolls one copy of the list (track is duplicated for the
  // seamless wrap), so pace by single-copy length: ~8s per entry, clamped.
  // Only sets duration — the animation itself still lives behind the
  // prefers-reduced-motion media query in CSS, so reduced-motion stays off.
  const loopSeconds = Math.min(Math.max(items.length * 8, 90), 200);

  const renderRow = (hidden) => (
    <div className="ticker-row" aria-hidden={hidden || undefined}>
      {items.map((t, i) => (
        <div className="ticker-item" key={(hidden ? 'b' : 'a') + i}>
          {mode === 'board' && (
            t.boardRank
              ? <span className="ticker-boardrank">№{t.boardRank}</span>
              : <span className="ticker-gradechip">{t.grade}</span>
          )}
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
          {mode === 'board' ? 'The board' : mode === 'movers' ? 'Board movers' : 'Top prospects'}
        </span>
      </div>
      <div className="ticker-viewport">
        <div
          className="ticker-track"
          style={{ animationDuration: `${loopSeconds}s` }}
        >
          {renderRow(false)}
          {renderRow(true)}
        </div>
      </div>
    </div>
  );
}

export default Ticker;

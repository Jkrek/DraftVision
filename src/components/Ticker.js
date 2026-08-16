import React, { useState, useEffect } from 'react';
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

function Ticker() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/prospects?limit=12')
      .then(r => (r.ok ? r.json() : Promise.reject(new Error('bad status'))))
      .then(data => {
        if (cancelled) return;
        const list = Array.isArray(data?.prospects)
          ? data.prospects
          : Array.isArray(data)
            ? data
            : [];
        setItems(list.map(toItem).filter(Boolean));
      })
      .catch(() => { /* render nothing on failure */ });
    return () => { cancelled = true; };
  }, []);

  if (!items.length) return null;

  const renderRow = (hidden) => (
    <div className="ticker-row" aria-hidden={hidden || undefined}>
      {items.map((t, i) => (
        <div className="ticker-item" key={(hidden ? 'b' : 'a') + i}>
          <span className="ticker-rank">{t.rank}</span>
          <span className="ticker-name">{t.name}</span>
          {t.pos && <span className="ticker-pos">{t.pos}</span>}
          {t.team && <span className="ticker-team">{t.team}</span>}
          {t.prob != null && <span className="ticker-prob">{t.prob}%</span>}
        </div>
      ))}
    </div>
  );

  return (
    <div className="ticker">
      <div className="ticker-label">
        <span className="ticker-dot" />
        <span className="ticker-label-text">Board movers</span>
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

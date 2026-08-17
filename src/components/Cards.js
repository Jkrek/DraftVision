import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import CardItem from './CardItem';
import { anonFetch } from '../lib/api';
import './Cards.css';

/*
 * Cards — two home-page sections (Nocturne):
 *   1. "Top of the board" — 6-cell prospect grid from /api/prospects?limit=6
 *   2. CTA band over lamar.png
 */

/* Static fallback so the grid never looks broken if the fetch fails. */
const FALLBACK = [
  { rank: '01', name: 'Caden Story', pos: 'DL', team: 'Clemson', grade: 'A+', prob: '90.5' },
  { rank: '02', name: 'TJ Guy', pos: 'EDGE', team: 'Michigan', grade: 'A+', prob: '90.5' },
  { rank: '03', name: 'Karson Kiesewetter', pos: 'S', team: 'Penn State', grade: 'A+', prob: '90.3' },
  { rank: '04', name: 'Joshua Nichols', pos: 'DB', team: 'Michigan', grade: 'A+', prob: '89.5' },
  { rank: '05', name: 'Max Heffner', pos: 'DB', team: 'Penn State', grade: 'A+', prob: '89.4' },
  { rank: '06', name: 'Trajen Odom', pos: 'DL', team: 'Ohio State', grade: 'A+', prob: '89.3' },
];

/* Map an /api/prospects row into the cell shape (defensive, same field
   fallbacks the Ticker and Leaderboard use). */
function toCell(p, i) {
  if (!p || typeof p !== 'object') return null;
  const name = p.name || p.player_name || p.player;
  if (!name) return null;
  const prob = p.success_probability ?? p.probability ?? p.prob;
  return {
    rank: String(i + 1).padStart(2, '0'),
    name,
    pos: p.position || p.pos || '—',
    team: p.team || p.school || '—',
    grade: p.grade || p.draft_grade || null,
    prob: typeof prob === 'number' ? prob.toFixed(1) : prob || '—',
  };
}

function Cards() {
  const [cells, setCells] = useState(FALLBACK);

  useEffect(() => {
    let cancelled = false;
    anonFetch('/api/prospects?limit=6')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('bad status'))))
      .then((data) => {
        if (cancelled) return;
        const list = Array.isArray(data?.prospects)
          ? data.prospects
          : Array.isArray(data)
            ? data
            : [];
        const mapped = list.map(toCell).filter(Boolean).slice(0, 6);
        if (mapped.length) setCells(mapped);
      })
      .catch(() => { /* keep the static fallback */ });
    return () => { cancelled = true; };
  }, []);

  return (
    <>
      {/* ── Top of the board ── */}
      <section className="board">
        <div className="board-head">
          <h2 className="board-heading">Top of the board</h2>
          <Link className="board-link" to="/leaderboard">
            All 9,033 prospects →
          </Link>
        </div>
        <div className="board-grid">
          {cells.map((c) => (
            <CardItem key={c.rank + c.name} {...c} />
          ))}
        </div>
      </section>

      {/* ── CTA band ── */}
      <section className="ctaband">
        <div className="ctaband-media" aria-hidden="true">
          <img
            src={process.env.PUBLIC_URL + '/images/CFB Content/lamar.png'}
            alt=""
            loading="lazy"
          />
        </div>
        <div className="ctaband-scrim" aria-hidden="true" />
        <div className="ctaband-inner">
          <div className="ctaband-copy">
            <h2 className="ctaband-heading">Stay ahead of the draft</h2>
            <p className="ctaband-sub">
              Free while in beta. Built at the University of Cincinnati.
            </p>
          </div>
          <div className="ctaband-ctas">
            <Link to="/predict" className="dv-cta">Run a prediction</Link>
            <Link to="/leaderboard" className="dv-cta dv-cta-ghost">Browse the board</Link>
          </div>
        </div>
      </section>
    </>
  );
}

export default Cards;

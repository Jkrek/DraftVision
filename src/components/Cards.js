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

/* Until the fetch resolves the grid shows six empty skeleton cells — no
   names, no numbers (a static list goes stale, and it is what a crawler
   reads). If the fetch fails or comes back empty the section is hidden. */
const SKELETON_CELLS = [0, 1, 2, 3, 4, 5];

function SkeletonCell() {
  return (
    <div className="board-cell board-cell-skeleton" aria-hidden="true">
      <div className="pp-shimmer board-skel-rank" />
      <div className="pp-shimmer board-skel-name" />
      <div className="pp-shimmer board-skel-meta" />
      <div className="pp-shimmer board-skel-prob" />
    </div>
  );
}

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
  const [cells, setCells] = useState(null); // null = loading
  const [failed, setFailed] = useState(false);

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
        else setFailed(true);
      })
      .catch(() => { if (!cancelled) setFailed(true); });
    return () => { cancelled = true; };
  }, []);

  return (
    <>
      {/* ── Top of the board ── */}
      {!failed && (
        <section className="board">
          <div className="board-head" data-reveal="">
            <h2 className="board-heading">Top of the board</h2>
            <Link className="board-link" to="/leaderboard">
              The full board →
            </Link>
          </div>
          <div className="board-grid" data-reveal="2" aria-busy={cells ? undefined : true}>
            {cells
              ? cells.map((c) => <CardItem key={c.rank + c.name} {...c} />)
              : SKELETON_CELLS.map((i) => <SkeletonCell key={i} />)}
          </div>
        </section>
      )}

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
        <div className="ctaband-inner" data-reveal="">
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

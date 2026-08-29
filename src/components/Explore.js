import React from 'react';
import { Link } from 'react-router-dom';
import './Explore.css';

/*
 * Explore — home-page destination grid (Nocturne). The top nav lists the
 * pages; this section sells them. One card per destination with the
 * one-line pitch the nav has no room for.
 */

const DESTINATIONS = [
  {
    to: '/predict',
    title: 'Predict',
    pitch: 'Run the model on any FBS player — grade, draft projection, comps and the why behind the number.',
  },
  {
    to: '/leaderboard',
    title: 'Model Board',
    pitch: 'The model’s full board, untouched by human hands: every FBS player graded and ranked.',
  },
  {
    to: '/big-board',
    title: 'Big Board',
    pitch: 'Curated class-by-class draft boards — Jared’s rankings, the model’s, and one you build yourself.',
  },
  {
    to: '/hs-prospects',
    title: 'HS Prospects',
    pitch: 'The recruiting pipeline: composite-ranked high-school classes with commitments as they land.',
  },
  {
    to: '/compare',
    title: 'Compare',
    pitch: 'Two prospects head-to-head — probabilities, factor attributions and historical comps, aligned.',
  },
  {
    to: '/mock-draft',
    title: 'Mock Draft',
    pitch: 'Play out draft night against the model’s projections, pick by pick.',
  },
  {
    to: '/backtest',
    title: 'Backtest',
    pitch: 'The receipts: what the model would have said in 2019–20 — hits and misses, side by side.',
  },
];

function Explore() {
  return (
    <section className="explore" aria-label="Explore the site">
      <div className="explore-head">
        <div className="explore-eyebrow">The toolkit</div>
        <h2 className="explore-heading">Everything on the site</h2>
      </div>
      <div className="explore-grid">
        {DESTINATIONS.map((d, i) => (
          <Link className="explore-card" to={d.to} key={d.to}>
            <span className="explore-num">{String(i + 1).padStart(2, '0')}</span>
            <span className="explore-body">
              <span className="explore-title">
                {d.title}
                <span className="explore-arrow" aria-hidden="true">→</span>
              </span>
              <span className="explore-pitch">{d.pitch}</span>
            </span>
          </Link>
        ))}
      </div>
    </section>
  );
}

export default Explore;

import React from 'react';
import { Link } from 'react-router-dom';
import './Explore.css';

/*
 * Explore — home-page destination grid (Nocturne). Mirrors the nav's five
 * groups one for one: a card per thing a visitor might want to do, with the
 * one-line pitch the nav has no room for and the pages inside the group.
 * A new page joins a card; it does not add one.
 */

const DESTINATIONS = [
  {
    to: '/leaderboard',
    title: 'Boards',
    pitch: 'Who the model rates, from high-school recruits to this year’s draft class.',
    more: [
      { to: '/leaderboard',  label: 'Model Board' },
      { to: '/big-board',    label: 'Big Board' },
      { to: '/hs-prospects', label: 'HS Prospects' },
      { to: '/services',     label: 'College Stars' },
    ],
  },
  {
    to: '/predict',
    title: 'Players',
    pitch: 'Run the model on any FBS player: grade, projected pick, comps and the reasons behind the number.',
    more: [
      { to: '/predict', label: 'Predict' },
      { to: '/compare', label: 'Compare two' },
    ],
  },
  {
    to: '/mock-draft',
    title: 'Mock Draft',
    pitch: 'Play out draft night against the model’s projections, pick by pick.',
  },
  {
    to: '/futures',
    title: 'Futures',
    pitch: 'Kalshi’s 2027 draft markets lined up against the model, with every disagreement recorded in public.',
  },
  {
    to: '/backtest',
    title: 'Track Record',
    pitch: 'What the model said about drafts it never saw, hits and misses side by side.',
    more: [
      { to: '/backtest',       label: 'Backtest' },
      { to: '/futures#record', label: 'Futures record' },
    ],
  },
];

function Explore() {
  return (
    <section className="explore" aria-label="Explore the site">
      <div className="explore-head" data-reveal="">
        <div className="explore-eyebrow">The toolkit</div>
        <h2 className="explore-heading">Where to go</h2>
      </div>
      <div className="explore-grid">
        {DESTINATIONS.map((d, i) => (
          <div className="explore-card" key={d.to} data-reveal={(i % 4) + 1}>
            <span className="explore-num">{String(i + 1).padStart(2, '0')}</span>
            <span className="explore-body">
              {/* The title link stretches over the whole card (::after), so
                  the card is one target without nesting links inside a link. */}
              <Link className="explore-title" to={d.to}>
                {d.title}
                <span className="explore-arrow" aria-hidden="true">→</span>
              </Link>
              <span className="explore-pitch">{d.pitch}</span>
              {d.more && (
                <span className="explore-more">
                  {d.more.map((m) => (
                    <Link className="explore-more-link" to={m.to} key={m.to}>{m.label}</Link>
                  ))}
                </span>
              )}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

export default Explore;

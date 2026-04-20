import React from 'react';
import { Link } from 'react-router-dom';
import './Cards.css';

const FEATURES = [
  {
    icon: '🔮',
    title: 'ML Prediction',
    desc: 'XGBoost + CatBoost ensemble evaluates production, athleticism, conference tier, and accolades to score any FBS prospect.',
    cta: 'Run a Prediction',
    path: '/predict',
    color: '#6366f1',
  },
  {
    icon: '📊',
    title: 'Prospect Leaderboard',
    desc: 'Every Power 5 player ranked and graded. Filter by position, grade, or school. Click any name to pull their full scouting report.',
    cta: 'Browse Rankings',
    path: '/leaderboard',
    color: '#a855f7',
  },
  {
    icon: '🏈',
    title: 'Mock Draft Board',
    desc: 'Upload your PFF mock draft CSV and get a polished, round-by-round draft board with team colors and one-click player previews.',
    cta: 'View Mock Draft',
    path: '/mock-draft',
    color: '#f59e0b',
  },
  {
    icon: '🌟',
    title: 'HS Prospect Rankings',
    desc: '247Sports composite ratings for the top high school recruits — 5★ down to 3★ across every class year, with commit tracking.',
    cta: 'See Recruits',
    path: '/hs-prospects',
    color: '#10b981',
  },
  {
    icon: '⚖️',
    title: 'Compare Players',
    desc: 'Stack multiple prospects side-by-side across every metric — success probability, draft projection, combine athleticism, and more.',
    cta: 'Compare Now',
    path: '/services',
    color: '#f43f5e',
  },
  {
    icon: '🎓',
    title: 'College Stars',
    desc: 'Browse synced rosters from every major program. Search by name or school to discover talent before the draft spotlight hits.',
    cta: 'Explore Rosters',
    path: '/products',
    color: '#0ea5e9',
  },
];

function Cards() {
  return (
    <section className="features-section">
      <div className="features-inner">
        <p className="features-eyebrow">
          <span className="features-eyebrow-line" />
          Everything in one place
        </p>
        <h2 className="features-heading">The Complete Scouting Suite</h2>
        <p className="features-intro">
          From high school recruits to NFL draft picks — DraftVision covers every level with ML-powered analysis and live data.
        </p>

        <div className="features-grid">
          {FEATURES.map((f) => (
            <Link
              key={f.title}
              to={f.path}
              className="feature-card"
              style={{ '--card-color': f.color }}
            >
              <div className="feature-card-icon">{f.icon}</div>
              <h3 className="feature-card-title">{f.title}</h3>
              <p className="feature-card-desc">{f.desc}</p>
              <span className="feature-card-cta">
                {f.cta}
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
              </span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export default Cards;

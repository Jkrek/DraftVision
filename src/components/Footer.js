import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

const NAV_COLS = [
  {
    heading: 'DraftVision',
    links: [
      { label: 'Predict',       to: '/predict'      },
      { label: 'Leaderboard',   to: '/leaderboard'  },
      { label: 'Mock Draft',    to: '/mock-draft'   },
      { label: 'HS Prospects',  to: '/hs-prospects' },
      { label: 'Compare',       to: '/services'     },
      { label: 'College Stars', to: '/products'     },
    ],
  },
  {
    heading: 'The Model',
    links: [
      { label: 'XGBoost + CatBoost', to: '/predict' },
      { label: 'Feature Importance',  to: '/predict' },
      { label: 'Conference Tiers',    to: '/predict' },
      { label: 'Success Criteria',    to: '/predict' },
      { label: 'Draft Grade Logic',   to: '/predict' },
    ],
  },
  {
    heading: 'About',
    links: [
      { label: 'Jared Krekeler',        to: '/' },
      { label: 'University of Cincinnati', to: '/' },
      { label: 'Computer Science',      to: '/' },
      { label: 'Spring 2025',           to: '/' },
    ],
  },
  {
    heading: 'Tech Stack',
    links: [
      { label: 'XGBoost / CatBoost', to: '/' },
      { label: 'Flask + Python',     to: '/' },
      { label: 'React 18',           to: '/' },
      { label: 'ESPN CFB API',       to: '/' },
      { label: 'CFBD API',           to: '/' },
      { label: 'Fly.io',             to: '/' },
    ],
  },
];

function Footer() {
  return (
    <footer className="footer">
      {/* CTA band */}
      <div className="footer-cta">
        <div className="footer-cta-inner">
          <div>
            <h3 className="footer-cta-heading">Stay ahead of the draft</h3>
            <p className="footer-cta-sub">ML-powered NFL prospect predictions · Built at the University of Cincinnati</p>
          </div>
          <Link to="/predict" className="footer-cta-btn">Run a Prediction</Link>
        </div>
      </div>

      {/* Links grid */}
      <div className="footer-links-grid">
        {NAV_COLS.map((col) => (
          <div key={col.heading} className="footer-col">
            <h4 className="footer-col-heading">{col.heading}</h4>
            {col.links.map((l) => (
              <Link key={l.label} to={l.to} className="footer-link">{l.label}</Link>
            ))}
          </div>
        ))}
      </div>

      {/* Bottom bar */}
      <div className="footer-bottom">
        <Link to="/" className="footer-bottom-logo">
          🏈 <span>DraftVision</span>
        </Link>
        <span className="footer-bottom-copy">© 2025 Jared Krekeler · University of Cincinnati</span>
        <div className="footer-bottom-socials">
          <a href="https://github.com" target="_blank" rel="noreferrer" aria-label="GitHub" className="footer-social">
            <i className="fab fa-github" />
          </a>
          <a href="https://linkedin.com" target="_blank" rel="noreferrer" aria-label="LinkedIn" className="footer-social">
            <i className="fab fa-linkedin" />
          </a>
          <a href="https://twitter.com" target="_blank" rel="noreferrer" aria-label="Twitter" className="footer-social">
            <i className="fab fa-twitter" />
          </a>
        </div>
      </div>
    </footer>
  );
}

export default Footer;

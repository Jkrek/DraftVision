import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

const FOOTER_LINKS = [
  { label: 'Predict',     to: '/predict'     },
  { label: 'Leaderboard', to: '/leaderboard' },
  { label: 'Mock Draft',  to: '/mock-draft'  },
];

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <svg
            className="footer-logo"
            width="22"
            height="22"
            viewBox="0 0 32 32"
            fill="none"
            aria-hidden="true"
          >
            <path d="M4 6 L16 26 L28 6" stroke="var(--color-neutral-600)" strokeWidth="2.2" strokeLinecap="square" />
            <path d="M10.5 6 L16 15 L21.5 6" stroke="var(--color-neutral-800)" strokeWidth="2.2" strokeLinecap="square" />
            <circle cx="16" cy="26" r="2.6" fill="var(--color-neutral-600)" />
            <circle cx="16" cy="26" r="6.5" stroke="var(--color-neutral-700)" strokeWidth="1" opacity=".5" />
          </svg>
          <span className="footer-copy">© 2026 Jared Krekeler · University of Cincinnati</span>
        </div>

        <div className="footer-links">
          {FOOTER_LINKS.map(({ label, to }) => (
            <Link key={to} to={to} className="footer-link">{label}</Link>
          ))}
          <a
            href="https://github.com/"
            target="_blank"
            rel="noreferrer"
            className="footer-link"
          >
            GitHub
          </a>
        </div>

        <div className="footer-disclaimer">
          Data from ESPN and the College Football Data API. Not affiliated with the NFL, NFLPA, or any team.
        </div>
      </div>
    </footer>
  );
}

export default Footer;

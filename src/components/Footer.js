import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';
import LogoMark from './Logo';

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
          <LogoMark size={22} muted className="footer-logo" />
          <span className="footer-copy">© 2026 Jared Krekeler · JK Football</span>
        </div>

        <div className="footer-links">
          {FOOTER_LINKS.map(({ label, to }) => (
            <Link key={to} to={to} className="footer-link">{label}</Link>
          ))}
          <a
            href="https://www.youtube.com/@jkrek"
            target="_blank"
            rel="noreferrer"
            className="footer-link footer-link-yt"
          >
            ▶ YouTube
          </a>
          <a
            href="https://github.com/Jkrek/DraftVision"
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

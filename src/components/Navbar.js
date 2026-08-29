import React, { useState, useEffect, useCallback } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import './Navbar.css';
import AuthButton from './AuthButton';
import LogoMark from './Logo';
import { getTheme, toggleTheme } from '../theme';

const NAV_LINKS = [
  { to: '/',             label: 'Overview',      end: true },
  { to: '/services',     label: 'College Stars'  },
  { to: '/hs-prospects', label: 'HS Prospects'   },
  { to: '/mock-draft',   label: 'Mock Draft'     },
  { to: '/leaderboard',  label: 'Model Board'    },
  { to: '/big-board',    label: 'Big Board'      },
  { to: '/predict',      label: 'Predict'        },
];

function ThemeToggle() {
  const [theme, setTheme] = useState(getTheme);
  const flip = () => setTheme(toggleTheme());
  const dark = theme === 'dark';
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={flip}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      title={dark ? 'Light theme' : 'Dark theme'}
    >
      <i className={dark ? 'fas fa-sun' : 'fas fa-moon'} aria-hidden="true" />
    </button>
  );
}

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [condensed, setCondensed] = useState(false);
  const [progress, setProgress] = useState(0);
  const location = useLocation();

  const close = useCallback(() => setMenuOpen(false), []);

  // Close menu on route change
  useEffect(() => { close(); }, [location.pathname, close]);

  // Prevent body scroll when menu is open
  useEffect(() => {
    document.body.style.overflow = menuOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [menuOpen]);

  // Scroll: condense the bar + drive the progress line (cheap, rAF-throttled)
  useEffect(() => {
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(() => {
        const y = window.scrollY || 0;
        const max = document.documentElement.scrollHeight - window.innerHeight;
        setCondensed(y > 40);
        setProgress(max > 0 ? Math.min(1, y / max) : 0);
        ticking = false;
      });
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, []);

  const linkClass = ({ isActive }) =>
    isActive ? 'nav-link nav-link-active' : 'nav-link';

  return (
    <nav className={condensed ? 'navbar navbar-condensed' : 'navbar'}>
      <div
        className="navbar-progress"
        style={{ transform: `scaleX(${progress})` }}
        aria-hidden="true"
      />
      <div className="navbar-container">

        {/* Logo */}
        <Link to="/" className="navbar-logo" onClick={close}>
          <LogoMark size={26} className="navbar-logo-mark" />
          <span className="navbar-logo-text">
            Draft<span className="navbar-logo-accent">Vision</span>
          </span>
        </Link>

        {/* Desktop links */}
        <div className="nav-menu-desktop">
          {NAV_LINKS.map(({ to, label, end }) => (
            <NavLink key={to} to={to} end={end} className={linkClass}>
              {label}
            </NavLink>
          ))}
        </div>

        {/* Right side */}
        <div className="nav-right">
          <a
            href="https://www.youtube.com/@jkrek"
            target="_blank"
            rel="noreferrer"
            className="nav-yt"
            title="JK Football on YouTube"
          >
            ▶ YouTube
          </a>
          <ThemeToggle />
          <AuthButton />
          <Link to="/sign-up" className="nav-cta">Get access</Link>
        </div>

        {/* Hamburger */}
        <button
          type="button"
          className="menu-icon"
          onClick={() => setMenuOpen(o => !o)}
          aria-label={menuOpen ? 'Close menu' : 'Open menu'}
          aria-expanded={menuOpen}
        >
          <i className={menuOpen ? 'fas fa-times' : 'fas fa-bars'} />
        </button>

      </div>

      {/* Mobile slide-down panel */}
      <div className={menuOpen ? 'nav-menu-mobile open' : 'nav-menu-mobile'}>
        {NAV_LINKS.map(({ to, label, end }) => (
          <NavLink key={to} to={to} end={end} className={linkClass} onClick={close}>
            {label}
          </NavLink>
        ))}
        <div className="nav-menu-mobile-auth">
          <ThemeToggle />
          <AuthButton />
          <Link to="/sign-up" className="nav-cta" onClick={close}>Get access</Link>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;

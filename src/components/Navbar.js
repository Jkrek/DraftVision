import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import './Navbar.css';
import AuthButton from './AuthButton';
import LogoMark from './Logo';
import { getTheme, toggleTheme } from '../theme';
import { getSpotlight } from '../lib/edgeData';
import { PALETTE_OPEN_EVENT } from './CommandPalette';

// Five entries, grouped by what a visitor is trying to do. A new page joins
// an existing group; it does not get a sixth top-level slot. An entry with
// `items` is a disclosure menu; one with `to` is a plain link.
const NAV = [
  {
    label: 'Boards',
    items: [
      { to: '/leaderboard',  label: 'Model Board',   hint: 'Every FBS player, ranked by the model' },
      { to: '/big-board',    label: 'Big Board',     hint: 'Draft boards by class' },
      { to: '/hs-prospects', label: 'HS Prospects',  hint: 'Recruiting classes' },
      { to: '/services',     label: 'College Stars', hint: 'The model’s highest-rated players' },
    ],
  },
  {
    label: 'Players',
    items: [
      { to: '/predict', label: 'Predict', hint: 'Run the model on one player' },
      { to: '/compare', label: 'Compare', hint: 'Two players side by side' },
    ],
  },
  {
    label: 'Draft',
    items: [
      { to: '/team-needs', label: 'Team Needs', hint: 'Depth charts, contracts and draft slots' },
      { to: '/mock-draft', label: 'Mock Draft', hint: 'Play out draft night' },
    ],
  },
  { to: '/futures',    label: 'Futures', live: true },
  {
    label: 'Track Record',
    items: [
      { to: '/backtest',       label: 'Backtest',       hint: 'Past drafts the model never saw' },
      { to: '/futures#record', label: 'Futures record', hint: 'Every recorded disagreement' },
    ],
  },
];

const LIVE_LABEL = 'Futures — live disagreements on the board';

// The live dot shows only when the market cache is healthy AND at least one
// disagreement is on the board (the server's count). No badge count,
// no dot while warming / seasonal / stale. One shared cache (edgeData): the
// nav reads getSpotlight(3) — the same promise the home FuturesBand awaits —
// so a home visit produces ONE /api/edge/spotlight request and the dot never
// adds a second /api/edge call on top of the page's own.
function liveFromSpotlight(d) {
  if (!d || !d.discovery || d.discovery.state !== 'ok' || !d.summary) return false;
  const s = d.summary;
  const n = s.disagreements != null ? s.disagreements
    : s.eligible_over_threshold != null ? s.eligible_over_threshold
      : s.edges_over_threshold;
  return Number(n) > 0;
}

// Desktop-only affordance: the palette answers ⌘K / Ctrl-K on its own, this
// is just the visible hint that the shortcut exists. It dispatches the same
// window event the palette listens for, so no shared state crosses the tree.
function PaletteHint() {
  const ua = typeof navigator === 'undefined' ? '' : (navigator.platform || navigator.userAgent || '');
  const mac = /Mac|iPhone|iPad|iPod/.test(ua);
  return (
    <button
      type="button"
      className="nav-palette"
      onClick={() => window.dispatchEvent(new Event(PALETTE_OPEN_EVENT))}
      aria-label="Search players, pages and commands"
      title={mac ? 'Search — ⌘K' : 'Search — Ctrl K'}
    >
      <i className="fas fa-search" aria-hidden="true" />
      <span className="nav-palette-key">{mac ? '⌘K' : 'Ctrl K'}</span>
    </button>
  );
}

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

// Disclosure menu: click or keyboard toggles it; Escape, an outside click, a
// focus move elsewhere, or a route change closes it.
function NavGroup({ group, pathname }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  // A hash entry (/futures#record) is a section of a page that has its own
  // top-level link, so it never lights the group up.
  const active = group.items.some((it) => !it.to.includes('#')
    && (it.to === pathname || pathname.startsWith(`${it.to}/`)));

  useEffect(() => { setOpen(false); }, [pathname]);

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      setOpen(false);
      const btn = ref.current && ref.current.querySelector('button');
      if (btn) btn.focus();
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const onBlur = (e) => {
    if (ref.current && !ref.current.contains(e.relatedTarget)) setOpen(false);
  };

  return (
    <div className="nav-group" ref={ref} onBlur={onBlur}>
      <button
        type="button"
        className={active ? 'nav-link nav-group-btn nav-link-active' : 'nav-link nav-group-btn'}
        aria-expanded={open}
        aria-haspopup="true"
        onClick={() => setOpen((o) => !o)}
      >
        {group.label}
        <span className="nav-caret" aria-hidden="true" />
      </button>
      {open && (
        <div className="nav-drop">
          {group.items.map((it) => (
            <Link
              key={it.to}
              to={it.to}
              className={it.to === pathname ? 'nav-drop-item nav-drop-item-active' : 'nav-drop-item'}
              onClick={() => setOpen(false)}
            >
              <span className="nav-drop-label">{it.label}</span>
              <span className="nav-drop-hint">{it.hint}</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [condensed, setCondensed] = useState(false);
  const [progress, setProgress] = useState(0);
  const location = useLocation();

  const [live, setLive] = useState(false);

  const close = useCallback(() => setMenuOpen(false), []);

  // Re-ask on every route change: the cache makes it free within its TTL and
  // lets the dot appear once a warming first pass has finished.
  useEffect(() => {
    let alive = true;
    getSpotlight(3).then((d) => { if (alive) setLive(liveFromSpotlight(d)); });
    return () => { alive = false; };
  }, [location.pathname]);

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
          {NAV.map((entry) => (entry.items ? (
            <NavGroup key={entry.label} group={entry} pathname={location.pathname} />
          ) : (
            <NavLink
              key={entry.to}
              to={entry.to}
              className={linkClass}
              aria-label={entry.live && live ? LIVE_LABEL : undefined}
            >
              {entry.label}
              {entry.live && live && <span className="nav-live-dot" aria-hidden="true" />}
            </NavLink>
          )))}
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
          <PaletteHint />
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
        {NAV.map((entry) => (entry.items ? (
          <div className="nav-mobile-group" key={entry.label}>
            <div className="nav-mobile-heading">{entry.label}</div>
            {entry.items.map((it) => (
              <Link
                key={it.to}
                to={it.to}
                className={it.to === location.pathname ? 'nav-link nav-link-active' : 'nav-link'}
                onClick={close}
              >
                {it.label}
              </Link>
            ))}
          </div>
        ) : (
          <NavLink
            key={entry.to}
            to={entry.to}
            className={linkClass}
            onClick={close}
            aria-label={entry.live && live ? LIVE_LABEL : undefined}
          >
            {entry.label}
            {entry.live && live && <span className="nav-live-dot" aria-hidden="true" />}
          </NavLink>
        )))}
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

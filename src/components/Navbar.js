import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';
import AuthButton from './AuthButton';

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  const close = () => setMenuOpen(false);

  // Close menu on route change
  useEffect(() => { close(); }, [location.pathname]);

  // Prevent body scroll when menu is open
  useEffect(() => {
    document.body.style.overflow = menuOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [menuOpen]);

  const isActive = (path) => location.pathname === path ? 'nav-links active' : 'nav-links';

  const navLinks = [
    { to: '/predict',      label: 'Predict'       },
    { to: '/leaderboard',  label: 'Leaderboard'   },
    { to: '/mock-draft',   label: 'Mock Draft'     },
    { to: '/hs-prospects', label: 'HS Prospects'  },
    { to: '/services',     label: 'Compare'        },
    { to: '/products',     label: 'College Stars'  },
  ];

  return (
    <nav className="navbar">
      <div className="navbar-container">

        {/* Logo */}
        <Link to="/" className="navbar-logo" onClick={close}>
          <div className="navbar-logo-mark">🏈</div>
          <span className="navbar-logo-text">DraftVision</span>
        </Link>

        {/* Nav links */}
        <ul className={menuOpen ? 'nav-menu active' : 'nav-menu'}>
          {navLinks.map(({ to, label }) => (
            <li className="nav-item" key={to}>
              <Link to={to} className={isActive(to)} onClick={close}>
                {label}
              </Link>
            </li>
          ))}
          {/* Auth — shown inside mobile menu */}
          <li className="nav-item" style={{ padding: '1rem 2rem' }}>
            <AuthButton />
          </li>
        </ul>

        {/* Right side */}
        <div className="nav-right">
          <AuthButton />
          <div className="nav-divider" />
          <Link to="/sign-up" className="nav-cta">
            Get Access
          </Link>
        </div>

        {/* Hamburger */}
        <div className="menu-icon" onClick={() => setMenuOpen(o => !o)} aria-label="Menu">
          <i className={menuOpen ? 'fas fa-times' : 'fas fa-bars'} />
        </div>

      </div>
    </nav>
  );
}

export default Navbar;

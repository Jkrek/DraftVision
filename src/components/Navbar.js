import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';
import { Button } from './button';
import AuthButton from './AuthButton';

function Navbar() {
  const [click, setClick] = useState(false);

  const handleClick = () => setClick(!click);
  const closeMobileMenu = () => setClick(false);
  const [button, setButton] = useState(true);

  const showButton = () => {
    if (window.innerWidth <= 960) {
      setButton(false);
    } else {
      setButton(true);
    }
  };

  useEffect(() => {
    showButton();
    window.addEventListener('resize', showButton);
    return () => window.removeEventListener('resize', showButton);
  }, []);

  return (
    <>
      <nav className="navbar">
        <div className="navbar-container">
          <Link to="/" className="navbar-logo" onClick={closeMobileMenu}>
            🏈 DraftVision
          </Link>
          <div className="menu-icon" onClick={handleClick}>
            <i className={click ? 'fas fa-times' : 'fas fa-bars'} />
          </div>
          <ul className={click ? 'nav-menu active' : 'nav-menu'}>
            <li className="nav-item">
              <Link to="/" className="nav-links" onClick={closeMobileMenu}>
                Home
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/predict" className="nav-links" onClick={closeMobileMenu}>
                Predict
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/services" className="nav-links" onClick={closeMobileMenu}>
                Compare
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/products" className="nav-links" onClick={closeMobileMenu}>
                College Stars
              </Link>
            </li>
            <li className="nav-item">
              <Link to="/sign-up" className="nav-links" onClick={closeMobileMenu}>
                Sign Up
              </Link>
            </li>
            <li className="nav-item">
              <AuthButton />
            </li>
          </ul>
          {button && <Button buttonStyle="btn--outline" to="/sign-up">SIGN UP</Button>}
        </div>
      </nav>
    </>
  );
}

export default Navbar;

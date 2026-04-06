import React from 'react';
import './Footer.css';
import { Link } from 'react-router-dom';

function Footer() {
  return (
    <div className='footer-container'>
      <section className='footer-subscription'>
        <p className='footer-subscription-heading'>
          Stay ahead of the draft with DraftVision
        </p>
        <p className='footer-subscription-text'>
          ML-powered NFL prospect predictions · Built at the University of Cincinnati
        </p>
        <div className='input-areas'>
          <form onSubmit={e => e.preventDefault()}>
            <input
              className='footer-input'
              name='email'
              type='email'
              placeholder='Your Email'
            />
            <Link to='/sign-up'>
              <button type='button' className='btn btn--outline btn--medium' style={{ marginLeft: '8px' }}>
                Get Access
              </button>
            </Link>
          </form>
        </div>
      </section>

      <div className='footer-links'>
        <div className='footer-link-wrapper'>
          <div className='footer-link-items'>
            <h2>DraftVision</h2>
            <Link to='/'>Home</Link>
            <Link to='/predict'>Predict Prospects</Link>
            <Link to='/services'>Compare Players</Link>
            <Link to='/products'>College Stars</Link>
            <Link to='/sign-up'>Sign Up</Link>
          </div>
          <div className='footer-link-items'>
            <h2>The Model</h2>
            <Link to='/predict'>How it Works</Link>
            <Link to='/predict'>Feature Importance</Link>
            <Link to='/predict'>Position Analysis</Link>
            <Link to='/predict'>Success Criteria</Link>
          </div>
        </div>
        <div className='footer-link-wrapper'>
          <div className='footer-link-items'>
            <h2>About</h2>
            <Link to='/'>Jared Krekeler</Link>
            <Link to='/'>University of Cincinnati</Link>
            <Link to='/'>Computer Science</Link>
            <Link to='/'>Spring 2025</Link>
          </div>
          <div className='footer-link-items'>
            <h2>Tech Stack</h2>
            <Link to='/'>XGBoost ML</Link>
            <Link to='/'>Flask API</Link>
            <Link to='/'>React 18</Link>
            <Link to='/'>ESPN CFB API</Link>
          </div>
        </div>
      </div>

      <section className='social-media'>
        <div className='social-media-wrap'>
          <div className='footer-logo'>
            <Link to='/' className='social-logo' style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '1.3rem' }}>🏈</span>
              DraftVision
            </Link>
          </div>
          <small className='website-rights'>DraftVision · Jared Krekeler · UC © 2025</small>
          <div className='social-icons'>
            <Link className='social-icon-link github' to='/' target='_blank' aria-label='GitHub'>
              <i className='fab fa-github' />
            </Link>
            <Link className='social-icon-link linkedin' to='/' target='_blank' aria-label='LinkedIn'>
              <i className='fab fa-linkedin' />
            </Link>
            <Link className='social-icon-link twitter' to='/' target='_blank' aria-label='Twitter'>
              <i className='fab fa-twitter' />
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Footer;

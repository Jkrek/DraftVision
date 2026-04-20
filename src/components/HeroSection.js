import React from 'react';
import { Link } from 'react-router-dom';
import './HeroSection.css';

const STATS = [
  { value: '4,000+',  label: 'FBS Prospects'   },
  { value: '3-Model', label: 'ML Ensemble'      },
  { value: '10-Tier', label: 'Conference Scale' },
  { value: 'Live',    label: 'ESPN Data'        },
];

function HeroSection() {
  return (
    <div className="hero-container">
      <div
        className="hero-background-image"
        style={{ backgroundImage: "url('/images/cfbstars2.jpeg')" }}
        aria-hidden="true"
      />

      {/* Live pill */}
      <div className="hero-pill">
        <span className="hero-pill-dot" />
        ML-Powered Draft Intelligence
      </div>

      {/* Title */}
      <h1 className="hero-title">DRAFTVISION</h1>

      {/* Subtitle */}
      <p className="hero-sub">
        Predict NFL prospect success with machine learning — powered by XGBoost, CatBoost, and live ESPN roster data.
      </p>
      <p className="hero-tagline">
        Browse every Power 5 prospect · Run instant scouting reports · Compare players side-by-side
      </p>

      {/* Stats strip */}
      <div className="hero-stats">
        {STATS.map(({ value, label }) => (
          <div className="hero-stat" key={label}>
            <span className="hero-stat-value">{value}</span>
            <span className="hero-stat-label">{label}</span>
          </div>
        ))}
      </div>

      {/* CTAs */}
      <div className="hero-btns">
        <Link to="/predict" className="hero-btn-primary">
          Run a Prediction
        </Link>
        <Link to="/leaderboard" className="hero-btn-secondary">
          Browse Prospects
        </Link>
      </div>

      {/* Scroll cue */}
      <div className="hero-scroll-cue" aria-hidden="true">
        <span>scroll</span>
        <div className="hero-scroll-line" />
      </div>
    </div>
  );
}

export default HeroSection;

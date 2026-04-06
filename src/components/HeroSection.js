import React from 'react';
import '../App.css';
import { Button } from './button';
import './HeroSection.css';

function HeroSection() {
  const heroBackgroundStyle = {
    backgroundImage: "url('/images/cfbstars2.jpeg')",
  };

  return (
    <div className='hero-container'>
      <div className='hero-background-image' style={heroBackgroundStyle} aria-hidden='true' />
      <h1>DRAFTVISION</h1>
      <p>ML-Powered NFL Prospect Success Prediction</p>
      <p className="subtitle">Browse 4,000+ college prospects · Run the XGBoost model · Get instant scouting reports</p>
      <div className='hero-btns'>
        <Button
          className='btns'
          buttonStyle='btn--outline'
          buttonSize='btn--large'
          to='/predict'
        >
          START PREDICTION
        </Button>
        <Button
          className='btns'
          buttonStyle='btn--primary'
          buttonSize='btn--large'
          to='/services'
        >
          VIEW FEATURES <i className='far fa-play-circle' />
        </Button>
      </div>
    </div>
  );
}

export default HeroSection;

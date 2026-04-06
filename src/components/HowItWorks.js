import React from 'react';
import './HowItWorks.css';

const steps = [
  {
    title: '1. Browse Prospects',
    detail: 'Filter 4,000+ synced college prospects by position, team, or name. Every major school and conference is covered via live ESPN roster data.',
  },
  {
    title: '2. Run the ML Model',
    detail: 'Click any prospect card to trigger the DraftVision XGBoost classifier. It evaluates 15 features — production, draft capital, combine athleticism, and college tier.',
  },
  {
    title: '3. Read the Scouting Report',
    detail: 'Get a success probability gauge, scout profile card, top prediction factors, and a full stat breakdown — instantly and clearly explained.',
  },
];

function HowItWorks() {
  return (
    <section className='how-it-works'>
      <div className='how-it-works__inner'>
        <p className='how-it-works__eyebrow'>Machine Learning · Draft Analytics</p>
        <h2>How DraftVision Works</h2>
        <p className='how-it-works__intro'>
          DraftVision uses a trained XGBoost classifier and live college roster data to predict
          whether any prospect has what it takes to succeed in the NFL — in seconds.
        </p>

        <div className='how-it-works__grid'>
          {steps.map((step) => (
            <article className='how-it-works__card' key={step.title}>
              <h3>{step.title}</h3>
              <p>{step.detail}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

export default HowItWorks;

import React from 'react';
import './HowItWorks.css';

const steps = [
  {
    number: '01',
    title: 'Browse Prospects',
    detail: 'Filter 4,000+ synced college prospects by position, team, or name. Every major school and conference is covered via live ESPN roster data.',
    color: '#6366f1',
  },
  {
    number: '02',
    title: 'Run the ML Model',
    detail: 'Click any prospect card to trigger the DraftVision XGBoost classifier. It evaluates 15 features — production, draft capital, combine athleticism, and college tier.',
    color: '#a855f7',
  },
  {
    number: '03',
    title: 'Read the Scouting Report',
    detail: 'Get a success probability gauge, scout profile card, top prediction factors, and a full stat breakdown — instantly and clearly explained.',
    color: '#ec4899',
  },
];

function HowItWorks() {
  return (
    <section className='how-it-works'>
      <div className='how-it-works__inner'>
        <p className='how-it-works__eyebrow'>Machine Learning · Draft Analytics</p>
        <h2 className='how-it-works__heading'>How DraftVision Works</h2>
        <p className='how-it-works__intro'>
          DraftVision uses a trained XGBoost classifier and live college roster data to predict
          whether any prospect has what it takes to succeed in the NFL — in seconds.
        </p>

        <div className='how-it-works__grid'>
          {steps.map((step) => (
            <article
              className='how-it-works__card'
              key={step.number}
              style={{ '--step-color': step.color }}
            >
              <div className='how-it-works__step-number'>{step.number}</div>
              <h3 className='how-it-works__card-title'>{step.title}</h3>
              <p className='how-it-works__card-text'>{step.detail}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

export default HowItWorks;

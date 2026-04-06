import React from 'react';
import './Cards.css';
import CardItem from './CardItem';

const featureCards = [
  {
    src: '/images/Top-NFL-Players.jpeg',
    text: 'Add multiple prospects side-by-side and compare their ML success probabilities, draft projections, and production scores.',
    label: 'Compare Players',
    path: '/services',
  },
  {
    src: '/images/cfbstars2.jpeg',
    text: 'Browse 4,000+ synced college prospects. Search by name or school to discover rising talent before the draft.',
    label: 'College Stars',
    path: '/products',
  },
  {
    src: '/images/collegestars.jpeg',
    text: 'Sign up to get early access to new DraftVision features, including position-specific models and real draft data.',
    label: 'Get Early Access',
    path: '/sign-up',
  },
];

function Cards() {
  const firstRow = featureCards.slice(0, 2);
  const secondRow = featureCards.slice(2);

  return (
    <div className='cards'>
      <h1>Explore scouting views</h1>
      <div className='cards__container'>
        <div className='cards__wrapper'>
          <ul className='cards__items'>
            {firstRow.map((card) => (
              <CardItem key={card.label} {...card} />
            ))}
          </ul>
          <ul className='cards__items'>
            {secondRow.map((card) => (
              <CardItem key={card.label} {...card} />
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Cards;

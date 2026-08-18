import React from 'react';
import { useNavigate } from 'react-router-dom';
import { nflLogoUrl } from '../../nflTeams';

/* "San Francisco 49ers" → "49ers" — compact trade label. */
const nick = (t) => (t || '').split(' ').pop();

/* One board row in the mock-draft visual language. Shared by the live
   simulator board and the imported-image board — both render into the
   same .mock-row grid from MockDraft.css. */
export default function MockRow({
  pick,      // pick number
  name,      // player name
  school,    // college
  nflTeam,   // NFL team (free text — logo matched by nflLogoUrl)
  position,
  grade,     // letter grade or PFF grade — rendered as a pill
  highlight, // true → user's own pick (accent edge)
  tag,       // 'STEAL' | 'REACH' | null — value-delta chip
  via,       // team the slot was acquired from ("via trade with X")
}) {
  const navigate = useNavigate();
  const logo = nflLogoUrl(nflTeam);

  return (
    <div className={`mock-row${highlight ? ' is-user' : ''}`}>
      <span className="mock-row-pick">{pick}</span>

      <div className="mock-row-player">
        <div className="mock-row-nameline">
          <span
            className={`mock-row-name${name ? ' is-link' : ''}`}
            onClick={() => name && navigate(`/predict?name=${encodeURIComponent(name)}`)}
          >
            {name || '—'}
          </span>
          {tag && (
            <span className={`mock-tag ${tag === 'STEAL' ? 'is-steal' : 'is-reach'}`}>
              {tag}
            </span>
          )}
        </div>
        {(school || via) && (
          <div className="mock-row-school">
            {school}
            {via && (
              <span className="mock-row-via">
                {school ? ' · ' : ''}via trade with {nick(via)}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="mock-row-team">
        {logo && (
          <img
            className="mock-row-logo"
            src={nflLogoUrl(nflTeam, 500)}
            alt=""
            loading="lazy"
            onError={(e) => { e.target.style.display = 'none'; }}
          />
        )}
        <span>{nflTeam || '—'}</span>
      </div>

      {position ? <span className="mock-row-pos">{position}</span> : <span />}

      {grade ? <span className="mock-row-grade"><span>{grade}</span></span> : <span />}

      {name ? (
        <button
          className="mock-row-btn"
          onClick={() => navigate(`/predict?name=${encodeURIComponent(name)}`)}
        >
          Predict
        </button>
      ) : (
        <span />
      )}
    </div>
  );
}

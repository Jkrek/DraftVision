import React from 'react';
import { useNavigate } from 'react-router-dom';
import { nflLogoUrl } from '../../nflTeams';

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
}) {
  const navigate = useNavigate();
  const logo = nflLogoUrl(nflTeam);

  return (
    <div className={`mock-row${highlight ? ' is-user' : ''}`}>
      <span className="mock-row-pick">{pick}</span>

      <div className="mock-row-player">
        <div
          className={`mock-row-name${name ? ' is-link' : ''}`}
          onClick={() => name && navigate(`/predict?name=${encodeURIComponent(name)}`)}
        >
          {name || '—'}
        </div>
        {school && <div className="mock-row-school">{school}</div>}
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

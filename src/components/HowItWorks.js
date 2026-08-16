import React from 'react';
import './HowItWorks.css';

/*
 * HowItWorks — three home-page sections (Nocturne):
 *   1. "How the number gets made" — heading + intro, 3 numbered hover rows
 *   2. Image strip — 4 flexible photo panels
 *   3. "Under the hood" — usc.jpeg backdrop + feature-importance bars
 */

const img = (file) => process.env.PUBLIC_URL + '/images/CFB Content/' + file;

const STEPS = [
  {
    number: '01',
    title: 'Browse prospects',
    detail:
      'Filter every synced college prospect by position, team or name. Rosters and stats come straight from live ESPN and College Football Data feeds, pre-cached so the board loads instantly.',
  },
  {
    number: '02',
    title: 'Run the ensemble',
    detail:
      'One click engineers sixteen features on the fly — a position-normalized production score, a 1-10 conference tier, combine athleticism, award flags — and runs them through calibrated XGBoost, CatBoost and a rule-based fallback.',
  },
  {
    number: '03',
    title: 'Read the report',
    detail:
      'You get a success probability, a projected draft grade, the factors that moved the number most, and the historical NFL players whose feature vectors sit closest to this one.',
  },
];

const STRIP = [
  { file: 'img-1.jpeg', mod: 'a' },
  { file: 'malachi-toney.webp', mod: 'b' },
  { file: '5000.avif', mod: 'c' },
  { file: 'ahmad-hardy.webp', mod: 'd' },
];

const IMPORTANCES = [
  { label: 'Production score', pct: 24, tone: 'hi' },
  { label: 'Conference tier', pct: 19, tone: 'hi' },
  { label: 'Combine speed score', pct: 17, tone: 'mid' },
  { label: 'Position group', pct: 14, tone: 'mid' },
  { label: 'Games / snaps played', pct: 13, tone: 'low' },
  { label: 'Award & All-America flags', pct: 13, tone: 'low' },
];

function HowItWorks() {
  return (
    <>
      {/* ── 1. How the number gets made ── */}
      <section className="hiw">
        <div className="hiw-head">
          <h2 className="hiw-heading">How the number gets made</h2>
          <p className="hiw-intro">
            Three steps, no black box. Every prediction ships with the factors
            that moved it and the historical players it resembles.
          </p>
        </div>
        {STEPS.map((st) => (
          <div className="hiw-step" key={st.number}>
            <div className="hiw-step-num">{st.number}</div>
            <h3 className="hiw-step-title">{st.title}</h3>
            <p className="hiw-step-detail">{st.detail}</p>
          </div>
        ))}
        <div className="hiw-endrule" aria-hidden="true" />
      </section>

      {/* ── 2. Image strip ── */}
      <section className="strip" aria-hidden="true">
        {STRIP.map((p) => (
          <div className={`strip-panel strip-panel--${p.mod}`} key={p.file}>
            <img src={img(p.file)} alt="" loading="lazy" />
            <div className="strip-scrim" />
          </div>
        ))}
      </section>

      {/* ── 3. Under the hood ── */}
      <section className="uth">
        <div className="uth-media" aria-hidden="true">
          <img src={img('usc.jpeg')} alt="" loading="lazy" />
        </div>
        <div className="uth-tint" aria-hidden="true" />
        <div className="uth-scrim-h" aria-hidden="true" />
        <div className="uth-scrim-v" aria-hidden="true" />
        <div className="uth-inner">
          <div className="uth-content">
            <div className="uth-eyebrow">Under the hood</div>
            <h2 className="uth-heading">
              Sixteen features, three models, one calibrated score
            </h2>
            <p className="uth-lede">
              A calibrated XGBoost classifier and a CatBoost model are averaged
              against a rule-based fallback, so the probability stays stable
              when a prospect&rsquo;s data is thin.
            </p>
            {IMPORTANCES.map((im, i) => (
              <div className="uth-bar" key={im.label}>
                <div className="uth-bar-head">
                  <span>{im.label}</span>
                  <span className="uth-bar-pct">{im.pct}%</span>
                </div>
                <div className="uth-bar-track">
                  <div
                    className={`uth-bar-fill uth-bar-fill--${im.tone}`}
                    style={{
                      width: `${Math.min(100, im.pct * 3.6)}%`,
                      animationDelay: `${(i * 0.06).toFixed(2)}s`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}

export default HowItWorks;

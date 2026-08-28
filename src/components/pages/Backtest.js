import React, { useEffect, useMemo, useState } from 'react';
import './Backtest.css';

/* Backtest — the receipts page. Renders public/data/backtest.json:
   what the production ensemble would have said about the 2019-2020 draft
   classes, which were held out of training entirely. Misses are shown on
   purpose — that is the point of the page. */

const TABS = [
  {
    key: 'top20',
    label: 'Top 20',
    blurb: "The model's 20 highest success probabilities across the whole 2019–20 holdout — its board, hits and misses together.",
  },
  {
    key: 'hit',
    label: 'Hits',
    blurb: 'High predicted probability, and the player actually hit — Pro Bowls or multiple seasons as a primary starter.',
  },
  {
    key: 'miss',
    label: 'Misses',
    blurb: 'The model was confident and wrong. These players got some of its highest probabilities and did not pan out.',
  },
  {
    key: 'steal',
    label: 'Steals',
    blurb: 'The model graded these players as Top-50 picks; the league let them slide to Day 2 or later — and they hit.',
  },
  {
    key: 'fade',
    label: 'Fades',
    blurb: 'The lowest probabilities the model handed out to players who indeed busted — correct thumbs-downs.',
  },
];

function fmtPct(p) {
  return `${(p * 100).toFixed(1)}%`;
}

function MetricRow({ label, model, baseline, better, digits = 4 }) {
  return (
    <tr>
      <td className="bt-metric-label">
        {label}
        <span className="bt-metric-note">{better}</span>
      </td>
      <td className="bt-num bt-num-model">{model.toFixed(digits)}</td>
      <td className="bt-num">{baseline.toFixed(digits)}</td>
    </tr>
  );
}

export default function Backtest() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('top20');

  useEffect(() => {
    let alive = true;
    fetch(`${process.env.PUBLIC_URL || ''}/data/backtest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json) => { if (alive) setData(json); })
      .catch((e) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, []);

  const rows = useMemo(() => {
    if (!data) return [];
    const list = data.players.filter((p) => p.categories.includes(tab));
    if (tab === 'fade') return [...list].sort((a, b) => a.pred_success_prob - b.pred_success_prob);
    return list; // already sorted by predicted probability, descending
  }, [data, tab]);

  const active = TABS.find((t) => t.key === tab);

  if (error) {
    return (
      <div className="bt-page">
        <div className="bt-main">
          <h1 className="bt-title">Backtest</h1>
          <p className="bt-lede">Could not load backtest data ({error}). Run scripts/generate_backtest.py to produce public/data/backtest.json.</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bt-page">
        <div className="bt-main"><p className="bt-lede">Loading receipts&hellip;</p></div>
      </div>
    );
  }

  const m = data.metrics;
  const b = m.baseline;

  return (
    <div className="bt-page">
      <header className="bt-hero">
        <div className="bt-hero-inner">
          <div className="bt-eyebrow">Receipts &mdash; held-out draft classes</div>
          <h1 className="bt-title">What the model would have said in 2019&ndash;20</h1>
          <div className="bt-lede-block">
            <p className="bt-lede">
              A <strong>holdout</strong> is data the model was never allowed to learn from.
              These models were trained on the 2000&ndash;2017 draft classes and calibrated
              on 2018; the <strong>2019 and 2020 classes were kept completely out of
              training</strong>. Everything below is the model meeting those {m.holdout_rows} players
              cold, scored with the exact training recipe behind the models running in
              production today &mdash; then compared against how their careers actually went.
            </p>
            <p className="bt-lede">
              The misses are shown on purpose. A projection page that only shows its wins
              is marketing; this page is the model&apos;s actual track record, good calls and
              bad ones side by side.
            </p>
            <p className="bt-lede bt-lede-honest">
              Honest read on the headline number: an AUC of {m.auc.toFixed(2)} means that if
              you hand the model one eventual NFL success and one bust at random, it ranks
              the success higher about {Math.round(m.auc * 100)}% of the time &mdash; meaningfully
              better than a coin flip at 50%, and a long way from certainty. Draft outcomes
              are mostly noise; treat every probability here as a lean, not a verdict.
            </p>
          </div>
        </div>
      </header>

      <main className="bt-main">
        <section aria-label="Headline metrics">
          <div className="bt-divider">Model vs. rule-based baseline &mdash; same {m.holdout_rows} holdout players</div>
          <div className="bt-metrics-wrap">
            <table className="bt-metrics">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="bt-num-h">ML ensemble</th>
                  <th className="bt-num-h">Rule-based baseline</th>
                </tr>
              </thead>
              <tbody>
                <MetricRow label="AUC (success)" better="higher is better" model={m.auc} baseline={b.auc} />
                <MetricRow label="Brier score (success)" better="lower is better" model={m.brier} baseline={b.brier} />
                <MetricRow label="Draft-grade accuracy" better="higher is better" model={m.accuracy} baseline={b.accuracy} />
              </tbody>
            </table>
            <p className="bt-metrics-foot">
              Baseline = the app&apos;s hand-written heuristic fallback, scored on the same
              holdout. Base rate: only {fmtPct(m.holdout_success_rate)} of these {m.holdout_rows} players
              became NFL successes at all.
            </p>
          </div>

          {m.pick && (
            <div className="bt-metrics-wrap">
              <div className="bt-divider">Draft-pick projection &mdash; where the model slots each player</div>
              <table className="bt-metrics">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th className="bt-num-h">Served projection</th>
                    <th className="bt-num-h">4-bucket baseline</th>
                  </tr>
                </thead>
                <tbody>
                  <MetricRow label="Avg. miss on drafted players" better="picks off, lower is better" digits={1}
                    model={m.pick.served_blend.mae_picks_drafted} baseline={m.pick.classifier_baseline.mae_picks_drafted} />
                  <MetricRow label="Rank correlation, top 64" better="orders the top two rounds, higher is better"
                    model={m.pick.served_blend.spearman_top64} baseline={m.pick.classifier_baseline.spearman_top64} />
                  <MetricRow label="1st-rounders flagged by pick 45" better="share of actual R1s, higher is better"
                    model={m.pick.served_blend.r1_recall_within_45} baseline={m.pick.classifier_baseline.r1_recall_within_45} />
                </tbody>
              </table>
              <p className="bt-metrics-foot">
                The served projection blends a pick regressor with the 4-bucket grade
                head; the baseline is the 4-bucket head alone. An average miss of
                ~{Math.round(m.pick.served_blend.mae_picks_drafted)} picks is honest scale for
                this problem &mdash; the projections are round-level leans, not slot calls.
              </p>
            </div>
          )}
        </section>

        <section aria-label="Notable calls">
          <div className="bt-divider">The receipts</div>

          <div className="bt-tabs" role="tablist" aria-label="Call categories">
            {TABS.map((t) => (
              <button
                key={t.key}
                role="tab"
                aria-selected={tab === t.key}
                className={`bt-tab${tab === t.key ? ' bt-tab-active' : ''}`}
                onClick={() => setTab(t.key)}
              >
                {t.label}
              </button>
            ))}
          </div>
          <p className="bt-tab-blurb">{active.blurb}</p>

          <div className="bt-table" role="table">
            <div className="bt-row bt-row-head" role="row">
              <span className="bt-c-name">Player</span>
              <span className="bt-c-year">Class</span>
              <span className="bt-c-prob">Pred. success</span>
              <span className="bt-c-bucket">Pred. bucket</span>
              <span className="bt-c-pick">Pred. pick</span>
              <span className="bt-c-bucket">Actual draft slot</span>
              <span className="bt-c-outcome">Outcome</span>
              <span className="bt-c-note">Career</span>
            </div>
            {rows.map((p) => (
              <div className="bt-row" role="row" key={`${p.name}-${p.draft_year}`}>
                <span className="bt-c-name">
                  <span className="bt-name">{p.name}</span>
                  <span className="bt-sub">{p.position}{p.college ? ` · ${p.college}` : ''}</span>
                </span>
                <span className="bt-c-year">{p.draft_year}</span>
                <span className="bt-c-prob">{fmtPct(p.pred_success_prob)}</span>
                <span className="bt-c-bucket">{p.pred_grade_bucket}</span>
                <span className="bt-c-pick">{p.pred_pick ? `~#${p.pred_pick}` : '—'}</span>
                <span className="bt-c-bucket">
                  {p.actual_round_bucket}
                  {p.actual_round ? (
                    <span className="bt-sub"> (Rd {p.actual_round}{p.actual_pick ? `, #${p.actual_pick}` : ''})</span>
                  ) : null}
                </span>
                <span className={`bt-c-outcome ${p.actual_success ? 'bt-hit' : 'bt-bust'}`}>
                  {p.actual_success ? 'Hit' : 'Bust'}
                </span>
                <span className="bt-c-note">{p.career_note}</span>
              </div>
            ))}
          </div>
        </section>

        <p className="bt-foot">
          Generated {new Date(data.generated_at).toLocaleDateString()}. {data.holdout_note}{' '}
          &ldquo;Hit&rdquo; = a Pro Bowl, 3+ seasons as a primary starter, or equivalent
          career value.
        </p>
      </main>
    </div>
  );
}

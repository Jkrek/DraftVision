import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
// /player/<slug> = name-team, matching PlayerPage's canonical helper.
import { playerSlug } from './PlayerPage';
import './Edge.css';

// Market Edge — read-only board comparing DraftVision model probabilities to
// public prices on Kalshi (a CFTC-regulated prediction-market exchange).
// We never take or facilitate bets; this page is analysis, and the paper
// ledger below exists to build a verifiable public track record BEFORE anyone
// is asked to trust an edge.

const fmtCents = (c) => (c == null ? '—' : `${c}¢`);
const fmtProb = (p) => (p == null ? '—' : `${Number(p).toFixed(1)}%`);

function EdgeChip({ edge }) {
  if (edge == null) return <span className="edge-chip edge-chip-none">—</span>;
  const sign = edge > 0 ? '+' : '';
  const cls = edge > 0 ? 'edge-chip-pos' : 'edge-chip-neg';
  return (
    <span className={`edge-chip ${cls}`}>
      {sign}
      {Number(edge).toFixed(1)} pts
    </span>
  );
}

function PlayerCell({ name, team }) {
  if (!name) return <span className="edge-muted">—</span>;
  if (!team) return <span>{name}</span>;
  return (
    <Link className="edge-player-link" to={`/player/${playerSlug(name, team)}`}>
      {name}
    </Link>
  );
}

export default function Edge() {
  const [data, setData] = useState(null);
  const [ledger, setLedger] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let alive = true;
    fetch('/api/edge')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('http'))))
      .then((d) => alive && setData(d))
      .catch(() => alive && setError(true));
    fetch('/api/edge/ledger')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('http'))))
      .then((d) => alive && setLedger(d))
      .catch(() => alive && setLedger({ entries: [] }));
    return () => {
      alive = false;
    };
  }, []);

  const markets = data?.markets || [];
  const seasonal = error || (data && markets.length === 0);
  const entries = ledger?.entries || [];

  return (
    <div className="edge-page">
      <div className="edge-inner">
        {/* ── Intro ── */}
        <header className="edge-header">
          <p className="eyebrow">Market Edge</p>
          <h1 className="edge-title">Where the model and the market disagree</h1>
          <p className="edge-sub">
            This board lines up DraftVision model probabilities against live
            prices on{' '}
            <a href="https://kalshi.com" target="_blank" rel="noopener noreferrer">
              Kalshi
            </a>
            , a CFTC-regulated prediction-market exchange. It is read-only
            analysis: we take no positions and facilitate no bets. An edge is
            shown only when a market&rsquo;s question maps directly onto
            something the model genuinely answers — every other relevant market
            is listed without a model number.
          </p>
          <p className="edge-disclaimer">
            Not investment advice. Prediction markets involve risk and are
            restricted to adults 18+. All market links open kalshi.com.
          </p>
        </header>

        {/* ── Edge board ── */}
        {!data && !error ? (
          <p className="edge-loading">Reading the market…</p>
        ) : seasonal ? (
          <div className="edge-empty">
            <p className="edge-empty-title">No draft markets on the board.</p>
            <p className="edge-empty-sub">
              Draft markets are thickest January–April — the model is watching.
            </p>
          </div>
        ) : (
          <div className="edge-table-wrap">
            <table className="edge-table">
              <thead>
                <tr>
                  <th className="edge-th-market">Market</th>
                  <th>Price</th>
                  <th>Model</th>
                  <th>Edge</th>
                  <th>Player</th>
                  <th aria-label="Kalshi link" />
                </tr>
              </thead>
              <tbody>
                {markets.map((m) => (
                  <tr key={m.ticker}>
                    <td className="edge-td-title">{m.title}</td>
                    <td className="edge-td-num">{fmtCents(m.yes_price_cents)}</td>
                    <td className="edge-td-num">{fmtProb(m.model_prob)}</td>
                    <td>
                      <EdgeChip edge={m.edge} />
                    </td>
                    <td>
                      <PlayerCell name={m.matched_player} team={m.matched_team} />
                    </td>
                    <td>
                      <a
                        className="edge-out-link"
                        href={m.url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        View ↗
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {data?.note && data.note !== 'seasonal' && (
              <p className="edge-note">{data.note}</p>
            )}
          </div>
        )}

        {/* ── Paper ledger ── */}
        <section className="edge-ledger">
          <h2 className="edge-ledger-title">Paper ledger</h2>
          <p className="edge-ledger-sub">
            Whenever the model and the market disagree by 10+ points, the call
            is recorded here automatically — date, price, and model number,
            frozen at the moment of disagreement. No money moves. The ledger
            exists to build a verifiable public track record <em>before</em>{' '}
            anyone is asked to trust an edge: if the model can&rsquo;t beat the
            market on paper, it has no business claiming an edge at all.
          </p>
          {entries.length === 0 ? (
            <p className="edge-muted">
              No recorded calls yet — entries appear when a 10+ point
              disagreement is first observed.
            </p>
          ) : (
            <div className="edge-table-wrap">
              <table className="edge-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th className="edge-th-market">Market</th>
                    <th>Player</th>
                    <th>Price</th>
                    <th>Model</th>
                    <th>Edge</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map((e) => (
                    <tr key={`${e.ticker}-${e.date}`}>
                      <td className="edge-td-num">{e.date}</td>
                      <td className="edge-td-title">{e.title}</td>
                      <td>
                        <PlayerCell name={e.player} />
                      </td>
                      <td className="edge-td-num">{fmtCents(e.market_price_cents)}</td>
                      <td className="edge-td-num">{fmtProb(e.model_prob)}</td>
                      <td>
                        <EdgeChip edge={e.edge} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

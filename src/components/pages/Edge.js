import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
// /player/<slug> = name-team, matching PlayerPage's canonical helper.
import { playerSlug } from './PlayerPage';
import './Edge.css';

// Market Edge — read-only board comparing DraftVision model-implied draft
// probabilities to public prices on Kalshi (a CFTC-regulated prediction-market
// exchange). We never take or facilitate bets; this page is analysis, and the
// paper ledger below exists to build a verifiable public track record BEFORE
// anyone is asked to trust an edge.
//
// Payload states from /api/edge (always HTTP 200):
//   note === 'warming'  — the server's background discovery has not finished
//                         its first pass yet; poll again shortly.
//   note === 'seasonal' — no relevant markets on the book (or upstream down).
//   otherwise           — markets[], summary{}, discovery{}.

const fmtCents = (c) => (c == null ? '—' : `${c}¢`);
const fmtProb = (p) => (p == null ? '—' : `${Number(p).toFixed(1)}%`);
const SOURCE_LABEL = {
  last: 'last',
  mid: 'mid',
  mid_wide: 'mid (wide)', // bid/ask midpoint on a wide, untraded book — shown, never ledgered
  ask: 'ask only',
  bid: 'bid only',
};
// discovery.state from /api/edge: 'ok' | 'stale' (last pass > 2×TTL ago,
// upstream failing since) | 'error' (no successful pass yet) | 'warming'.
const DISCOVERY_LABEL = { ok: 'live', stale: 'stale', error: 'upstream error', warming: 'warming' };

const WARMING_POLL_MS = 4000;
const WARMING_MAX_POLLS = 10;

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

function PlayerCell({ name, team, ambiguous }) {
  if (!name) return <span className="edge-muted">—</span>;
  const body = team ? (
    <Link className="edge-player-link" to={`/player/${playerSlug(name, team)}`}>
      {name}
    </Link>
  ) : (
    <span>{name}</span>
  );
  return (
    <>
      {body}
      {ambiguous && (
        <span
          className="edge-flag"
          title="Two prospects share this name; matched by the market's team hint, then projected pick."
        >
          ?
        </span>
      )}
    </>
  );
}

function PriceCell({ cents, source }) {
  if (cents == null) return <span className="edge-muted">—</span>;
  return (
    <>
      {fmtCents(cents)}
      {source && source !== 'last' && (
        <span className="edge-price-src" title={`Quote source: ${SOURCE_LABEL[source] || source}`}>
          {SOURCE_LABEL[source] || source}
        </span>
      )}
    </>
  );
}

function ModelCell({ prob, basis }) {
  if (prob == null) {
    return (
      <span className="edge-muted" title={basis || undefined}>
        —
      </span>
    );
  }
  return (
    <span className="edge-model" title={basis || undefined}>
      {fmtProb(prob)}
    </span>
  );
}

export default function Edge() {
  const [data, setData] = useState(null);
  const [ledger, setLedger] = useState(null);
  const [error, setError] = useState(false);
  const [polls, setPolls] = useState(0);

  useEffect(() => {
    let alive = true;
    fetch('/api/edge/ledger')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('http'))))
      .then((d) => alive && setLedger(d))
      .catch(() => alive && setLedger({ entries: [] }));
    return () => {
      alive = false;
    };
  }, []);

  // Fetch the board; while the server reports 'warming', re-fetch on a short
  // timer (bounded) instead of showing the seasonal empty state.
  useEffect(() => {
    let alive = true;
    let timer = null;
    fetch('/api/edge')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('http'))))
      .then((d) => {
        if (!alive) return;
        setData(d);
        if (d && d.note === 'warming' && polls < WARMING_MAX_POLLS) {
          timer = setTimeout(() => alive && setPolls((n) => n + 1), WARMING_POLL_MS);
        }
      })
      .catch(() => alive && setError(true));
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [polls]);

  const markets = data?.markets || [];
  const warming = !error && data && data.note === 'warming' && polls < WARMING_MAX_POLLS;
  const seasonal = error || (data && !warming && markets.length === 0);
  const entries = ledger?.entries || [];
  const summary = data?.summary || null;
  const modeled = summary ? summary.modeled : markets.filter((m) => m.edge != null).length;
  const listedOnly = !seasonal && !warming && data && markets.length > 0 && modeled === 0;
  const fetchedAt = data?.discovery?.fetched_at_utc;
  const discoveryState = data?.discovery?.state;
  const discoveryError = data?.discovery?.error;
  const discoveryAge = data?.discovery?.age_s;

  return (
    <div className="edge-page">
      <div className="edge-inner">
        {/* ── Intro ── */}
        <header className="edge-header">
          <p className="eyebrow">Market Edge</p>
          <h1 className="edge-title">Where the model and the market disagree</h1>
          <p className="edge-sub">
            This board lines up DraftVision&rsquo;s model-implied draft-position
            probabilities against live prices on{' '}
            <a href="https://kalshi.com" target="_blank" rel="noopener noreferrer">
              Kalshi
            </a>
            , a CFTC-regulated prediction-market exchange. It is read-only
            analysis: we take no positions and facilitate no bets. A model number
            is shown only for draft-position threshold questions (top-N,
            over/under), derived from each player&rsquo;s conformal pick
            interval and renormalized so no event&rsquo;s top-N mass exceeds N
            &mdash; hover a number for the exact basis. Exact-pick markets
            (&ldquo;#1 overall&rdquo;) are listed without one: an exact-pick
            probability is not identifiable from an interval. Every other
            relevant market is listed without one too.
          </p>
          <p className="edge-disclaimer">
            Not investment advice. Prediction markets involve risk and are
            restricted to adults 18+. All market links open kalshi.com.
          </p>
        </header>

        {/* ── Edge board ── */}
        {(!data && !error) || warming ? (
          <p className="edge-loading">
            {warming ? 'Reading the market — first pass in progress…' : 'Reading the market…'}
          </p>
        ) : seasonal ? (
          <div className="edge-empty">
            <p className="edge-empty-title">No draft markets on the board.</p>
            <p className="edge-empty-sub">
              Kalshi opens the next cycle&rsquo;s #1-overall and top-5 books the
              week after the draft; the finer books arrive January–April. The
              model is watching.
            </p>
          </div>
        ) : (
          <>
            {summary && (
              <p className="edge-summary">
                <span>
                  <strong>{summary.markets}</strong> markets
                </span>
                <span>
                  <strong>{summary.priced}</strong> priced
                </span>
                <span>
                  <strong>{summary.matched}</strong> matched
                </span>
                <span>
                  <strong>{summary.modeled}</strong> modeled
                </span>
                <span>
                  <strong>{summary.edges_over_threshold}</strong> disagreements ≥10 pts
                </span>
                {fetchedAt && (
                  <span className="edge-summary-time">
                    prices as of {new Date(fetchedAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                    {discoveryState && (
                      <span
                        className={`edge-state edge-state-${discoveryState}`}
                        title={
                          discoveryError
                            ? `Last Kalshi pass failed: ${discoveryError}`
                            : discoveryAge != null
                              ? `Last successful Kalshi pass ${Math.round(discoveryAge / 60)} min ago`
                              : undefined
                        }
                      >
                        {DISCOVERY_LABEL[discoveryState] || discoveryState}
                      </span>
                    )}
                  </span>
                )}
                {discoveryError && discoveryState !== 'ok' && (
                  <span className="edge-summary-error" title={discoveryError}>
                    {discoveryState === 'stale'
                      ? 'Kalshi has not answered since the last successful pass; showing the last good prices.'
                      : 'Kalshi discovery is failing; prices may be missing.'}
                  </span>
                )}
              </p>
            )}
            {listedOnly && (
              <div className="edge-banner">
                <strong>Listed, not priced.</strong> These markets are on the book,
                but no row maps onto a model output right now — hover a dash in the
                Model column to see why. Nothing here is an edge.
              </div>
            )}
            <div className="edge-table-wrap">
              <table className="edge-table">
                <thead>
                  <tr>
                    <th className="edge-th-market">Market</th>
                    <th>Price</th>
                    <th>
                      Model{' '}
                      <span className="edge-th-note" title="Model-implied from the conformal pick interval, not a persisted posterior.">
                        model-implied
                      </span>
                    </th>
                    <th>Edge</th>
                    <th>Player</th>
                    <th aria-label="Kalshi link" />
                  </tr>
                </thead>
                <tbody>
                  {markets.map((m) => (
                    <tr key={m.ticker}>
                      <td className="edge-td-title">{m.title}</td>
                      <td className="edge-td-num">
                        <PriceCell cents={m.yes_price_cents} source={m.yes_price_source} />
                      </td>
                      <td className="edge-td-num">
                        <ModelCell prob={m.model_prob} basis={m.model_basis} />
                      </td>
                      <td>
                        <EdgeChip edge={m.edge} />
                      </td>
                      <td>
                        <PlayerCell
                          name={m.matched_player}
                          team={m.matched_team}
                          ambiguous={m.match_ambiguous}
                        />
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
            </div>
            {data?.note && data.note !== 'seasonal' && (
              <p className="edge-note">{data.note}</p>
            )}
          </>
        )}

        {/* ── Paper ledger ── */}
        <section className="edge-ledger">
          <h2 className="edge-ledger-title">Paper ledger</h2>
          <p className="edge-ledger-sub">
            Whenever the model-implied probability and a real two-sided market
            quote (a last trade or a midpoint on a tight or traded book — never
            a one-sided ask or a placeholder 1¢/65¢ book) disagree by 10+
            points, the call is recorded here automatically — date, price, and
            model number, frozen at the moment of disagreement.
            No money moves. The ledger exists to build a verifiable public track
            record <em>before</em> anyone is asked to trust an edge: if the model
            can&rsquo;t beat the market on paper, it has no business claiming an
            edge at all.
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
                      <td className="edge-td-num">
                        <PriceCell cents={e.market_price_cents} source={e.price_source} />
                      </td>
                      <td className="edge-td-num">
                        <ModelCell prob={e.model_prob} basis={e.model_basis} />
                      </td>
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

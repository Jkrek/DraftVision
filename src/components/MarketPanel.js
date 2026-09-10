import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import GapBar, { gapWord } from './GapBar';
import InfoTip from './InfoTip';
import { getPlayerMarkets } from '../lib/edgeData';
import './MarketPanel.css';

/*
 * MarketPanel — "The market" section on a player profile.
 *
 * Public Kalshi quotes for this one player next to what the model's conformal
 * pick interval implies, one cell per market (Top-5 → #1 overall → Heisman,
 * max 3). Read-only: DraftVision takes no positions.
 *
 * Renders NOTHING for state no_markets / warming / seasonal / error, and when
 * every matched row is ambiguous (two prospects share the name — a wrong-
 * player screenshot is worse than no panel). Never blocks the page: the
 * fetch runs after the player resolves, through the shared edgeData cache.
 *
 * Also home to the small row helpers the hero card and the predict report
 * share (question kind, "priced top-5 row") — edgeData.js is a frozen
 * interface, so they live here.
 */

const KIND_ORDER = { top_n: 0, pick_eq: 1, heisman: 2 };

/* Row → { kind, n, label }. Prefers the API's `question`; until the backend
   lands it, derives the kind from the ticker prefix / title. */
export function questionOf(row) {
  if (row && row.question && row.question.kind) return row.question;
  const ticker = String((row && (row.ticker || row.series)) || '');
  const title = String((row && row.title) || '');
  const topN = title.match(/top[ -](\d+)/i);
  if (/^KXNFLDRAFTTOP/i.test(ticker) || topN) {
    const n = topN ? Number(topN[1]) : null;
    return { kind: 'top_n', n, label: n ? `Top-${n} pick` : 'Top-N pick' };
  }
  if (/^KXHEISMAN/i.test(ticker) || /heisman/i.test(title)) {
    return { kind: 'heisman', n: null, label: 'Heisman' };
  }
  if (/^KXNFLDRAFT1(ST)?(-|$)/i.test(ticker) || /^KXNFLDRAFTPICK/i.test(ticker)
    || /#1 overall|first overall|1st overall/i.test(title)) {
    return { kind: 'pick_eq', n: 1, label: '#1 overall' };
  }
  return { kind: 'unknown', n: null, label: 'Other' };
}

export const isPriced = (row) => !!row
  && row.yes_price_cents != null && row.model_prob != null;

/* Signed gap, model − market (the API's `edge`; recomputed when absent). */
export function gapOf(row) {
  if (!row) return null;
  if (row.edge != null && Number.isFinite(Number(row.edge))) return Number(row.edge);
  if (!isPriced(row)) return null;
  return Number(row.model_prob) - Number(row.yes_price_cents);
}

/* Unambiguous rows from a /api/edge/player payload, in Top-5 → #1 → Heisman
   order; [] for every non-ok state. */
export function playerRows(payload) {
  if (!payload || payload.state !== 'ok' || !Array.isArray(payload.markets)) return [];
  return payload.markets
    .filter((r) => r && !r.match_ambiguous)
    .map((r) => ({ row: r, q: questionOf(r) }))
    .sort((a, b) => (KIND_ORDER[a.q.kind] ?? 3) - (KIND_ORDER[b.q.kind] ?? 3));
}

/* The one row the hero card / predict report care about: a priced top-5. */
export function topFiveRow(payload) {
  const hit = playerRows(payload).find(({ row, q }) => q.kind === 'top_n' && q.n === 5 && isPriced(row));
  return hit ? hit.row : null;
}

export const fmtGap = (g) => (g == null ? '—'
  : `${g < 0 ? '−' : g > 0 ? '+' : ''}${Math.abs(Math.round(g))}`);

const fmtTime = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null
    : d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
};
const fmtDate = (ymd) => {
  if (!ymd) return null;
  const d = new Date(`${ymd}T12:00:00Z`);
  return Number.isNaN(d.getTime()) ? ymd
    : d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
};

const UNPRICED_REASON = {
  pick_eq: 'exact pick isn’t identifiable from an interval',
  heisman: 'an award, not a draft slot',
};

const PANEL_TIP = 'Public Kalshi quotes for this player, next to what the model’s '
  + 'conformal pick interval implies. Read-only — DraftVision takes no positions.';

function intervalCaption(row) {
  const iv = row.model_interval;
  if (!iv || iv.lo == null || iv.hi == null) return null;
  const cov = iv.coverage != null ? `${Math.round(Number(iv.coverage) * 100)}% ` : '';
  return `from the ${cov}interval, picks ${iv.lo}–${iv.hi}`;
}

function MarketCell({ row, q, delay }) {
  const priced = isPriced(row);
  const market = row.yes_price_cents;
  const model = row.model_prob;
  const gap = gapOf(row);
  const caption = intervalCaption(row);
  return (
    <div className={`pp-grade-cell mp-cell mp-cell-${q.kind}`}>
      <div className="pp-cell-label">
        {q.label} · Kalshi
        {row.model_basis && <InfoTip text={row.model_basis} />}
      </div>

      <div className="mp-numerals">
        <div className="mp-num">
          <span className="mp-num-value">{market != null ? `${Math.round(market)}¢` : '—'}</span>
          <span className="mp-num-unit">market</span>
        </div>
        <div className="mp-num mp-num-model">
          {priced ? (
            <>
              <span className="mp-num-value">{Math.round(model)}%</span>
              <span className="mp-num-unit">model-implied</span>
            </>
          ) : (
            <>
              <span className="mp-num-value mp-num-none">—</span>
              <span className="mp-num-unit">not priced</span>
            </>
          )}
        </div>
        {priced && gap != null && (
          <div className="mp-gap">
            <span className="mp-gap-value">{fmtGap(gap)}</span>
            <span className="mp-gap-word">{gapWord(gap)}</span>
          </div>
        )}
      </div>

      <GapBar
        market={market}
        model={priced ? model : null}
        source={row.yes_price_source || 'last'}
        size="row"
        labels
        delay={delay}
      />

      {priced ? (
        <>
          {(caption || row.wide_interval) && (
            <div className="mp-caption">
              {caption}
              {row.wide_interval && (
                <span className="mp-caption-wide">
                  {caption ? ' · ' : ''}wide interval — conservative by construction
                </span>
              )}
            </div>
          )}
          <p className="mp-sentence">
            The market thinks {Math.round(market)}¢. The model says{' '}
            <span className="mp-sentence-model">{Math.round(model)}%</span>.
          </p>
        </>
      ) : (
        <p className="mp-sentence mp-sentence-muted">
          Listed, not priced — {UNPRICED_REASON[q.kind] || 'no model number for this question'}.
        </p>
      )}

      {row.url && (
        <div className="mp-source">
          <a href={row.url} target="_blank" rel="noopener noreferrer">Kalshi ↗</a>
        </div>
      )}
    </div>
  );
}

export default function MarketPanel({ name, team }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    if (!name) return undefined;
    let alive = true;
    setData(null);
    getPlayerMarkets(name, team).then((d) => { if (alive) setData(d); });
    return () => { alive = false; };
  }, [name, team]);

  const rows = playerRows(data).slice(0, 3);
  if (rows.length === 0) return null;

  const asOf = fmtTime(data.as_of);
  const since = data.on_ledger ? fmtDate(data.ledger_since) : null;

  return (
    <section className="pp-section pp-market">
      <h2 className="pp-section-title">
        The market
        <InfoTip text={PANEL_TIP} />
      </h2>
      <p className="mp-sub">
        {asOf ? `Kalshi public prices as of ${asOf} · read-only` : 'Kalshi public prices · read-only'}
      </p>
      <div className="pp-grade-card mp-card">
        {rows.map(({ row, q }, i) => (
          <MarketCell key={row.ticker || `${q.kind}-${i}`} row={row} q={q} delay={i * 120} />
        ))}
      </div>
      {data.on_ledger && (
        <p className="mp-ledger">
          <Link to="/futures#record">
            On the paper record{since ? ` since ${since}` : ''} — scored after the draft
          </Link>
        </p>
      )}
    </section>
  );
}

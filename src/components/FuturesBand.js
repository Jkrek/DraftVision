import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import GapBar, { gapWord } from './GapBar';
import { getSpotlight, playerSlug } from '../lib/edgeData';
import './FuturesBand.css';

/*
 * FuturesBand — the home page's ink band, right after the hero.
 *
 * LEFT (the story): eyebrow, one data-generated sentence — "The market says
 * 90¢. The model says 37%." — the lede naming the player, then the LADDER:
 * the three largest disagreements as a 1px-gapped board-grid of GapBars.
 * RIGHT (the scoreboard): the 2×2 hero-stats grid ('N priced' / 'N
 * disagreements ≥ 10' / 'N calls on paper' / '0 scored · settles May 2027'),
 * the ghost CTA to /futures, the quieter link to the record and the legend.
 * FOOTER strip: live/stale chip, "Kalshi public prices as of …", paper record.
 *
 * Data: getSpotlight(3) (one request per home visit through the shared
 * edgeData cache — the nav dot reads the same promise).
 *
 * Every colour inside the band is pinned hex (never a theme token): the
 * band is composed to read as a 1200×630 crop in either theme.
 *
 * States: warming / error → render nothing (the home page must not break;
 * warming keeps polling so the band appears once the first pass lands);
 * seasonal → a single ink strip with the record scoreboard.
 */

const WARMING_POLL_MS = 4000;
const WARMING_MAX_POLLS = 10;

const fmtGap = (g) => (g == null ? '—'
  : `${g < 0 ? '−' : g > 0 ? '+' : ''}${Math.abs(Math.round(g))}`);

const fmtTime = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null
    : d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
};

/* 'YYYY-MM-DD' → 'May 2027' (noon UTC so no timezone can roll the month). */
const fmtMonth = (ymd) => {
  if (!ymd) return null;
  const d = new Date(`${ymd}T12:00:00Z`);
  return Number.isNaN(d.getTime()) ? null
    : d.toLocaleDateString([], { month: 'short', year: 'numeric' });
};

const num = (v) => (v == null || v === '' ? null : Number(v));

/* ESPN full headshot → team logo → nothing. */
function Headshot({ espnId, teamId }) {
  const [stage, setStage] = useState(espnId ? 0 : teamId ? 1 : 2);
  if (stage === 2) return <span className="fb-headshot fb-headshot-empty" aria-hidden="true" />;
  const src = stage === 0
    ? `https://a.espncdn.com/i/headshots/college-football/players/full/${espnId}.png`
    : `https://a.espncdn.com/i/teamlogos/ncaa/500/${teamId}.png`;
  return (
    <img
      className={`fb-headshot${stage === 1 ? ' fb-headshot-logo' : ''}`}
      src={src}
      alt=""
      loading="lazy"
      onError={() => setStage((s) => (s === 0 && teamId ? 1 : 2))}
    />
  );
}

function questionLabel(q, year) {
  const label = (q && q.label) || 'Market';
  return year ? `${label} · ${year}` : label;
}

/* One ladder row: 40px headshot · name + meta + full-width GapBar · gap numeral. */
function LadderRow({ row, year, delay }) {
  const gap = row.gap != null ? Number(row.gap) : Number(row.model) - Number(row.market);
  const slug = row.slug || playerSlug(row.player, row.team);
  return (
    <div className="board-cell fb-cell">
      <Headshot espnId={row.espn_id} teamId={row.espn_team_id} />
      <div className="fb-cell-who">
        <Link className="fb-cell-name" to={`/player/${slug}`}>
          {row.player}
          {row.match_ambiguous && (
            <span className="fb-flag" title="Two prospects share this name">?</span>
          )}
        </Link>
        <div className="fb-cell-meta">{questionLabel(row.question, year)}</div>
      </div>
      <div className="fb-gap">
        <span className="fb-gap-value">{fmtGap(gap)}</span>
        <span className="fb-gap-word">{gapWord(gap)}</span>
      </div>
      <div className="fb-cell-bar">
        <GapBar
          market={row.market}
          model={row.model}
          source={row.price_source || 'last'}
          size="row"
          labels
          ground="ink"
          delay={delay}
        />
      </div>
    </div>
  );
}

/* The one scoreboard component (hero-stats grammar on ink). */
function Scoreboard({ cells, className }) {
  return (
    <div className={`fb-record${className ? ` ${className}` : ''}`}>
      {cells.map((c) => (
        <div className="fb-record-cell" key={c.label}>
          <div className="fb-record-value">{c.value}</div>
          <div className="fb-record-label">{c.label}</div>
        </div>
      ))}
    </div>
  );
}

/* Fourth cell: '0 scored · settles May 2027' until anything resolves, then
   'Model closer 7 of 11'. */
function scoredCell(record) {
  const r = record || {};
  const resolved = num(r.resolved) || 0;
  if (resolved > 0) {
    return { value: `${num(r.model_closer) || 0} of ${resolved}`, label: 'model closer' };
  }
  const scored = num(r.scored) != null ? num(r.scored) : resolved;
  const month = fmtMonth(r.settles_on);
  return { value: scored, label: month ? `scored · settles ${month}` : 'scored · settles draft night' };
}

function homeCells(summary, record) {
  const s = summary || {};
  const r = record || {};
  return [
    { value: num(s.modeled) != null ? num(s.modeled) : '—', label: 'priced' },
    { value: num(s.disagreements) != null ? num(s.disagreements) : '—', label: 'disagreements ≥ 10' },
    { value: num(r.calls) != null ? num(r.calls) : '—', label: 'calls on paper' },
    scoredCell(r),
  ];
}

function recordCells(record) {
  const r = record || {};
  return [
    { value: num(r.calls) != null ? num(r.calls) : '—', label: 'calls on paper' },
    { value: num(r.open) != null ? num(r.open) : '—', label: 'open' },
    scoredCell(r),
  ];
}

function SeasonalStrip({ data }) {
  const year = data.summary && data.summary.draft_year ? Number(data.summary.draft_year) + 1 : null;
  return (
    <section className="futures-band futures-band-seasonal" aria-label="Model vs. market — off-season">
      <div className="fb-inner fb-inner-seasonal">
        <div className="fb-seasonal-copy">
          <div className="fb-eyebrow">
            <span className="fb-eyebrow-text">Model vs. Market · draft futures</span>
          </div>
          <p className="fb-seasonal-line">
            Kalshi opens the {year ? `${year} books` : 'next cycle’s books'} the week after the draft.
            The model is watching.
          </p>
          <Link className="board-link fb-link" to="/futures#record">The record →</Link>
        </div>
        <Scoreboard cells={recordCells(data.record)} className="fb-record-3" />
      </div>
    </section>
  );
}

export default function FuturesBand() {
  const [data, setData] = useState(null);
  const [polls, setPolls] = useState(0);

  useEffect(() => {
    let alive = true;
    let timer = null;
    getSpotlight(3).then((d) => {
      if (!alive) return;
      setData(d);
      if (d && d.state === 'warming' && polls < WARMING_MAX_POLLS) {
        timer = setTimeout(() => alive && setPolls((n) => n + 1), WARMING_POLL_MS);
      }
    });
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [polls]);

  if (!data || data.state === 'warming' || data.state === 'error') return null;
  if (data.state === 'seasonal') return <SeasonalStrip data={data} />;
  if (data.state !== 'ok') return null;

  const rows = (Array.isArray(data.rows) ? data.rows : [])
    .filter((r) => r && r.player && !r.match_ambiguous && r.market != null && r.model != null)
    .slice(0, 3);
  const head = data.headline && !data.headline.match_ambiguous
    && data.headline.market != null && data.headline.model != null
    ? data.headline : rows[0];
  if (!head) return null;

  const summary = data.summary || {};
  const record = data.record || {};
  const year = summary.draft_year || null;
  const headGap = head.gap != null ? Number(head.gap) : Number(head.model) - Number(head.market);
  const headSlug = head.slug || playerSlug(head.player, head.team);
  const asOf = fmtTime(data.as_of);
  const stale = data.discovery && data.discovery.state === 'stale';

  return (
    <section id="futures" className="futures-band" aria-label="The model versus the market">
      <div className="fb-inner">
        {/* ── left: the story + the ladder ── */}
        <div className="fb-story">
          <div className="fb-eyebrow">
            <span className="fb-eyebrow-dot" aria-hidden="true" />
            <span className="fb-eyebrow-text">
              Model vs. Market{year ? ` · ${year} draft futures` : ' · draft futures'}
            </span>
          </div>
          <h2 className="fb-heading">
            The market says <span className="fb-num-market">{Math.round(head.market)}¢</span>.
            {' '}The model says <span className="fb-num-model">{Math.round(head.model)}%</span>.
          </h2>
          <p className="fb-lede">
            <Link className="fb-lede-name" to={`/player/${headSlug}`}>{head.player}</Link>
            {' · '}{(head.question && head.question.label) || 'Market'}
            {year ? `, ${year} draft` : ''}
            {' · gap '}<span className="fb-lede-gap">{fmtGap(headGap)}</span>
            {' · '}{gapWord(headGap)}
          </p>
          <div className="board-grid fb-grid">
            {rows.map((row, i) => (
              <LadderRow key={row.ticker || `${row.player}-${i}`} row={row} year={year} delay={200 + i * 140} />
            ))}
          </div>
        </div>

        {/* ── right: the scoreboard ── */}
        <div className="fb-side">
          <Scoreboard cells={homeCells(summary, record)} />
          <div className="fb-ctas">
            <Link className="dv-cta dv-cta-ghost fb-cta" to="/futures">See every market →</Link>
            <Link className="fb-quiet-link" to="/futures#record">The record →</Link>
          </div>
          <p className="fb-legend">
            ○ market price (Kalshi, yes-cents) · ● model-implied probability · band = the gap
          </p>
        </div>

        {/* ── footer strip ── */}
        <div className="fb-foot">
          <span className={`fb-state${stale ? ' fb-state-stale' : ''}`}>
            {!stale && <span className="fb-state-dot" aria-hidden="true" />}
            {stale ? 'stale' : 'live'}
          </span>
          <span className="fb-foot-text">
            {asOf ? `Kalshi public prices as of ${asOf}` : 'Kalshi public prices'}
            {' · read-only, no positions are taken.'}
          </span>
          <Link className="fb-foot-link" to="/futures#record">
            paper record: {record.calls != null ? record.calls : '—'} calls, scored on draft night →
          </Link>
        </div>
      </div>
    </section>
  );
}

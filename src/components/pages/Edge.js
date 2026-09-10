import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import GapBar, { gapWord } from '../GapBar';
import InfoTip from '../InfoTip';
import { getEdge, getLedger, playerSlug } from '../../lib/edgeData';
import './Edge.css';

// Futures — "The Gap". Read-only board lining up Kalshi's public 2027 draft
// futures against the model-implied probability from each player's conformal
// pick interval. Nothing here asks anyone to do anything: the market thinks
// X, the model says Y, and the paper ledger (#record) is the receipt, scored
// on draft night. Page order: hero + scoreboard → method strip → the gap
// board → listed-not-priced grids → the record → dense table fallback. Route: /futures (the component keeps its file name; the
// /api/edge* paths are unchanged).
//
// Data comes only from the shared edgeData cache (getEdge / getLedger).
// Payload states from /api/edge (always HTTP 200):
//   note === 'warming'  — first discovery pass not finished; poll (4 s × 10).
//   note === 'seasonal' — no relevant markets on the book.
//   state === 'error'   — edgeData's never-reject fallback for a failed fetch.
// Additive fields (question, matched_slug, espn ids, model_interval,
// event_scale, by_kind, resolution…) are optional: everything below derives
// a fallback from the fields that have always been there.

const HERO_IMG = process.env.PUBLIC_URL + '/images/CFB Content/cfbstars2.jpeg';
const THRESHOLD = 10;
const WARMING_POLL_MS = 4000;
const WARMING_MAX_POLLS = 10;
const UNPRICED_PREVIEW = 12;

const SOURCE_LABEL = {
  last: 'last',
  mid: 'mid',
  mid_wide: 'mid (wide)', // bid/ask midpoint on a wide, untraded book — shown, never recorded
  ask: 'ask only',
  bid: 'bid only',
};
// discovery.state from /api/edge: 'ok' | 'stale' (last pass > 2×TTL ago,
// upstream failing since) | 'error' (no successful pass yet) | 'warming'.
const DISCOVERY_LABEL = { ok: 'live', stale: 'stale', error: 'upstream error', warming: 'warming' };

const LEDGER_TIP = 'Only two-sided, tight or traded quotes are recorded; a 1¢/65¢ placeholder book is shown but never scored.';
const FLAG_TIP = 'Two prospects share this name; matched by the market’s team hint, then projected pick.';
const UNPRICED_REASON = {
  pick_eq: 'no model number — exact pick isn’t identifiable from an interval',
  heisman: 'no model number — an award, not a draft slot',
};

const headshotUrl = (id) => `https://a.espncdn.com/i/headshots/college-football/players/full/${id}.png`;
const teamLogoUrl = (id) => `https://a.espncdn.com/i/teamlogos/ncaa/500/${id}.png`;

/* ── formatting ─────────────────────────────────────────────────────────── */

const fmtCents = (c) => (c == null ? '—' : `${Math.round(c)}¢`);
const fmtPct = (p) => (p == null ? '—' : `${Math.round(p)}%`);
const fmtPct1 = (p) => (p == null ? '—' : `${Number(p).toFixed(1)}%`);
const fmtGap = (g) => {
  if (g == null || !Number.isFinite(Number(g))) return '—';
  const r = Math.round(Math.abs(g));
  if (r === 0) return '0';
  return g > 0 ? `+${r}` : `−${r}`;
};
const fmtBrier = (b) => (b == null ? '—' : Number(b).toFixed(3));
const fmtTime = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
};
// 'YYYY-MM-DD' → local Date (avoids the UTC day-shift of new Date('YYYY-MM-DD'))
const ymdToDate = (s) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(s || ''));
  return m ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3])) : null;
};
const fmtDay = (s) => {
  const d = ymdToDate(s);
  return d ? d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) : s || '—';
};
const fmtMonthYear = (s) => {
  const d = ymdToDate(s);
  return d ? d.toLocaleDateString([], { month: 'short', year: 'numeric' }) : null;
};
const daysUntil = (s) => {
  const d = ymdToDate(s);
  if (!d) return null;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return Math.max(0, Math.round((d - today) / 86400000));
};

/* ── deriving the contract's additive fields when they are absent ───────── */

const EVENT_YEAR_RE = /-(\d{2})(?:\D|$)/;
const EVENT_N_RE = /^KXNFLDRAFT(?:TOP|PICK)-\d{2}-(\d{1,3})$/i;

function questionOf(m) {
  if (m && m.question && m.question.kind) return m.question;
  const series = String(m?.series || String(m?.ticker || '').split('-')[0] || '').toUpperCase();
  const event = String(m?.event_ticker || String(m?.ticker || '').split('-').slice(0, 3).join('-')).toUpperCase();
  const title = m?.title || '';
  const n = (EVENT_N_RE.exec(event) || [])[1];
  if (series.startsWith('KXHEISMAN')) return { kind: 'heisman', n: null, label: 'Heisman' };
  if (series.startsWith('KXNFLDRAFTTOP')) {
    if (/-R1$/.test(event)) return { kind: 'top_n', n: 32, label: 'First round' };
    const k = n ? Number(n) : (/top[\s-]*(\d{1,3})/i.exec(title) || [])[1];
    return k ? { kind: 'top_n', n: Number(k), label: `Top-${k} pick` } : { kind: 'unknown', n: null, label: 'Other' };
  }
  if (series.startsWith('KXNFLDRAFTPICK')) {
    const k = n ? Number(n) : (/picked\s+(\d{1,3})/i.exec(title) || [])[1];
    return { kind: 'pick_eq', n: k ? Number(k) : null, label: `#${k || '?'} overall` };
  }
  if (/^KXNFLDRAFT(QB|RB|WR|TE|OL|EDGE|DT|LB|DB)/.test(series)) return { kind: 'nth_position', n: null, label: 'Positional order' };
  return { kind: 'unknown', n: null, label: 'Other' };
}

const chipText = (q) => {
  if (!q) return '';
  if (q.kind === 'top_n') return q.n === 32 ? 'ROUND 1' : `TOP ${q.n}`;
  if (q.kind === 'pick_eq') return `#${q.n ?? '?'} OVERALL`;
  if (q.kind === 'heisman') return 'HEISMAN';
  return String(q.label || q.kind || '').toUpperCase();
};

const BASIS_INTERVAL_RE = /(\d{1,3})%\s+(?:conformal\s+)?(?:pick\s+)?interval\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]/i;
const BASIS_SCALE_RE = /scaled\s+x\s*([\d.]+)/i;

function intervalOf(m) {
  if (m.model_interval && m.model_interval.lo != null) return m.model_interval;
  const hit = BASIS_INTERVAL_RE.exec(m.model_basis || '');
  return hit ? { lo: Number(hit[2]), hi: Number(hit[3]), coverage: Number(hit[1]) / 100 } : null;
}
function scaleOf(m) {
  if (m.event_scale != null) return Number(m.event_scale);
  const hit = BASIS_SCALE_RE.exec(m.model_basis || '');
  return hit ? Number(hit[1]) : 1;
}
function draftYearOf(m) {
  if (m.draft_year) return m.draft_year;
  const hit = EVENT_YEAR_RE.exec(String(m.event_ticker || m.ticker || '').toUpperCase());
  return hit ? 2000 + Number(hit[1]) : null;
}

function enrich(m) {
  const q = questionOf(m);
  const interval = intervalOf(m);
  const scale = scaleOf(m);
  const wide = m.wide_interval != null ? Boolean(m.wide_interval) : Boolean(interval && interval.hi > 100);
  const slug = m.matched_slug || (m.matched_player && m.matched_team ? playerSlug(m.matched_player, m.matched_team) : null);
  const gap = m.edge != null ? Number(m.edge) : null;
  return {
    ...m,
    q,
    interval,
    scale,
    wide,
    slug,
    gap,
    year: draftYearOf(m),
    priced: m.model_prob != null && m.yes_price_cents != null,
    big: gap != null && Math.abs(gap) >= THRESHOLD,
  };
}

// '72% ' (with trailing space) or '' — the coverage is read off the payload,
// never assumed: the interval's nominal coverage is a model setting.
const coverageWord = (cov) => (cov != null && Number.isFinite(Number(cov)) ? `${Math.round(Number(cov) * 100)}% ` : '');
const scaleBit = (scale) => (scale != null && Math.abs(scale - 1) > 0.005);

const basisCaption = (row) => {
  if (!row.interval) return null;
  const scale = scaleBit(row.scale) ? ` · ×${row.scale.toFixed(2)} event scale` : '';
  return `from ${coverageWord(row.interval.coverage)}interval, picks ${row.interval.lo}–${row.interval.hi}${scale}`;
};

/* ── small pieces ───────────────────────────────────────────────────────── */

// ESPN full headshot → team logo → nothing. Never a broken image.
function Headshot({ espnId, teamId, size = 36 }) {
  const [i, setI] = useState(0);
  const candidates = [espnId ? headshotUrl(espnId) : null, teamId ? teamLogoUrl(teamId) : null].filter(Boolean);
  if (i >= candidates.length) return null;
  return (
    <img
      className={`fut-headshot${i > 0 ? ' fut-headshot-logo' : ''}`}
      style={{ width: size, height: size }}
      src={candidates[i]}
      alt=""
      loading="lazy"
      onError={() => setI((n) => n + 1)}
    />
  );
}

function QChip({ q }) {
  const cls = q && q.kind === 'top_n' ? 'fut-qchip fut-qchip-top' : 'fut-qchip';
  return <span className={cls}>{chipText(q)}</span>;
}

function PlayerName({ name, slug, ambiguous, className = '' }) {
  if (!name) return <span className="fut-muted">—</span>;
  return (
    <>
      {slug ? (
        <Link className={`fut-player${className ? ` ${className}` : ''}`} to={`/player/${slug}`}>{name}</Link>
      ) : (
        <span className={`fut-player-plain${className ? ` ${className}` : ''}`}>{name}</span>
      )}
      {ambiguous && (
        <button type="button" className="fut-flag" title={FLAG_TIP} aria-label={`Ambiguous name — ${FLAG_TIP}`}>
          ?
        </button>
      )}
    </>
  );
}

function SourceLink({ url }) {
  if (!url) return <span className="fut-source fut-source-none" />;
  return (
    <a className="fut-source" href={url} target="_blank" rel="noopener noreferrer">
      Kalshi ↗
    </a>
  );
}

// The numeric triplet beside every rail: market · model · signed gap.
function Triplet({ market, source, model, basis, gap, unpricedReason }) {
  return (
    <>
      <span className="fut-num fut-num-market">
        {fmtCents(market)}
        {source && source !== 'last' && (
          <span className="fut-src-tag" title={`Quote source: ${SOURCE_LABEL[source] || source}`}>
            {SOURCE_LABEL[source] || source}
          </span>
        )}
      </span>
      <span className="fut-num fut-num-model">
        {model != null ? (
          <span className="fut-model" title={basis || undefined}>{fmtPct(model)}</span>
        ) : (
          <span className="fut-not-priced" title={basis || unpricedReason || undefined}>not priced</span>
        )}
      </span>
      <span className="fut-num fut-num-gap">
        <span className="fut-gap-val">{fmtGap(gap)}</span>
        <span className="fut-gap-word">{gap != null ? gapWord(gap, THRESHOLD) : ''}</span>
      </span>
    </>
  );
}

// The one scoreboard component: page top, ledger head (hero-stats grammar).
function Scoreboard({ cells, foot, className = '' }) {
  return (
    <div className={`fut-scoreboard${className ? ` ${className}` : ''}`}>
      <div className="fut-stats">
        {cells.map((c) => (
          <div className="fut-stat" key={c.label}>
            <div className="fut-stat-value">{c.value}</div>
            <div className="fut-stat-label">
              {c.label}
              {c.tip && <InfoTip text={c.tip} place={c.tipPlace || 'top'} />}
            </div>
            {c.sub && <div className="fut-stat-sub">{c.sub}</div>}
          </div>
        ))}
      </div>
      {foot && <div className="fut-stats-foot">{foot}</div>}
    </div>
  );
}

function DiscoveryChip({ discovery }) {
  const state = discovery?.state;
  if (!state) return null;
  const err = discovery.error;
  const age = discovery.age_s;
  const title = err
    ? `Last Kalshi pass failed: ${err}`
    : age != null ? `Last successful Kalshi pass ${Math.round(age / 60)} min ago` : undefined;
  return (
    <span className={`fut-chip fut-chip-${state}`} title={title}>
      {state === 'ok' && <span className="fut-chip-dot" aria-hidden="true" />}
      {DISCOVERY_LABEL[state] || state}
    </span>
  );
}

function DiscoveryBanner({ discovery }) {
  const state = discovery?.state;
  if (!state || state === 'ok' || state === 'warming') return null;
  return (
    <p className="fut-banner fut-banner-warn" title={discovery.error || undefined}>
      {state === 'stale'
        ? 'Kalshi has not answered since the last successful pass; showing the last good prices.'
        : 'Kalshi discovery is failing; prices may be missing.'}
    </p>
  );
}

function StatusChip({ resolution, settlesOn }) {
  const outcome = resolution?.outcome || null;
  const closer = resolution?.closer || null;
  const brier = resolution && (resolution.brier_model != null || resolution.brier_market != null)
    ? `model Brier ${fmtBrier(resolution.brier_model)} · market Brier ${fmtBrier(resolution.brier_market)}`
    : undefined;
  if (!outcome) {
    const when = fmtMonthYear(settlesOn);
    return <span className="fut-status fut-status-open">open{when ? ` · settles ${when}` : ''}</span>;
  }
  if (outcome === 'void') return <span className="fut-status fut-status-neutral" title={resolution.note || undefined}>void</span>;
  if (closer === 'model') return <span className="fut-status fut-status-model" title={brier}>model closer</span>;
  if (closer === 'market') return <span className="fut-status fut-status-market" title={brier}>market closer</span>;
  return <span className="fut-status fut-status-neutral" title={brier}>push</span>;
}

/* ── skeletons ──────────────────────────────────────────────────────────── */

function RailSkeleton({ rows = 3 }) {
  return (
    <div className="fut-skeleton" aria-label="Reading the market">
      {Array.from({ length: rows }).map((_, i) => (
        <div className="pp-shimmer fut-shimmer-rail" key={i} />
      ))}
    </div>
  );
}

/* ── sections ───────────────────────────────────────────────────────────── */

function HeadlineCard({ row, seasonal, warming, errored, listedOnly, marketCount, recordCells }) {
  if (warming) {
    return (
      <div className="fut-hero-card">
        <div className="fut-hero-card-eyebrow">Reading the market — first pass in progress</div>
        <RailSkeleton rows={3} />
      </div>
    );
  }
  if (seasonal || errored || listedOnly || !row) {
    // Four honest states share the record-scoreboard card; only the words differ.
    let eyebrow;
    let note;
    if (errored) {
      eyebrow = 'Kalshi is not answering right now';
      note = 'Prices are unavailable until the next successful pass. The paper ledger below still stands.';
    } else if (seasonal) {
      eyebrow = 'Between cycles';
      note = 'Kalshi opens the next cycle’s books the week after the draft. The model is watching.';
    } else if (listedOnly) {
      eyebrow = 'Listed, not priced';
      note = `${marketCount} market${marketCount === 1 ? '' : 's'} on the book; none maps onto a model output right now. The prices below are the market’s alone.`;
    } else {
      eyebrow = 'No headline gap today';
      note = 'Every priced row sits on a wide or one-sided quote or an ambiguous name, so none qualifies as the headline. The board below still lines them up.';
    }
    return (
      <div className="fut-hero-card">
        <div className="fut-hero-card-eyebrow">{eyebrow}</div>
        <p className="fut-hero-card-note">{note}</p>
        <Scoreboard className="fut-scoreboard-inset" cells={recordCells} />
      </div>
    );
  }
  const higher = gapWord(row.gap, THRESHOLD);
  return (
    <div className="fut-hero-card">
      <div className="fut-head-nums">
        <div className="fut-head-num">
          <span className="fut-head-kicker">The market thinks</span>
          <span className="fut-head-figure fut-head-figure-market">{fmtCents(row.yes_price_cents)}</span>
        </div>
        <span className="fut-head-hairline" aria-hidden="true" />
        <div className="fut-head-num">
          <span className="fut-head-kicker">The model says</span>
          <span className="fut-head-figure fut-head-figure-model">{fmtPct(row.model_prob)}</span>
        </div>
      </div>
      <div className="fut-head-who">
        <PlayerName name={row.matched_player} slug={row.slug} ambiguous={row.match_ambiguous} className="fut-head-name" />
      </div>
      <div className="fut-head-meta">
        {row.q.label}{row.year ? ` · ${row.year}` : ''} · gap {fmtGap(row.gap)}{higher ? ` · ${higher}` : ''}
      </div>
      <GapBar
        size="hero"
        labels
        market={row.yes_price_cents}
        model={row.model_prob}
        source={row.yes_price_source}
        threshold={THRESHOLD}
        delay={250}
      />
      <div className="fut-head-basis">
        {row.interval
          ? `model-implied P(pick ≤ ${row.q.n ?? '?'}) from the ${coverageWord(row.interval.coverage)}interval [${row.interval.lo}, ${row.interval.hi}]`
          : row.model_basis}
        {row.wide && <span className="fut-wide"> · wide interval — conservative by construction</span>}
        <InfoTip text={row.model_basis || 'Model-implied from the conformal pick interval.'} place="top-left" />
      </div>
      <div className="fut-head-foot">
        <SourceLink url={row.url} />
      </div>
    </div>
  );
}

function BoardRow({ row, rank, onLedger, delay }) {
  const dim = row.priced && !row.big;
  const caption = basisCaption(row);
  return (
    <div className={`fut-row${dim ? ' fut-row-dim' : ''}${!row.priced ? ' fut-row-unpriced' : ''}`} role="listitem">
      <span className="fut-rank">{String(rank).padStart(2, '0')}</span>
      <span className="fut-cell-shot">
        <Headshot espnId={row.matched_espn_id} teamId={row.matched_espn_team_id} size={32} />
      </span>
      <span className="fut-cell-who">
        <span className="fut-who-line">
          <PlayerName name={row.matched_player} slug={row.slug} ambiguous={row.match_ambiguous} />
          <QChip q={row.q} />
        </span>
        {!row.matched_player && <span className="fut-who-title">{row.title}</span>}
      </span>
      <span className="fut-cell-rail">
        <GapBar
          size="row"
          labels
          market={row.yes_price_cents}
          model={row.model_prob}
          source={row.yes_price_source}
          threshold={THRESHOLD}
          delay={delay}
        />
        {caption && <span className="fut-rail-caption">{caption}</span>}
      </span>
      <Triplet
        market={row.yes_price_cents}
        source={row.yes_price_source}
        model={row.model_prob}
        basis={row.model_basis}
        gap={row.gap}
        unpricedReason={UNPRICED_REASON[row.q.kind]}
      />
      <span className="fut-cell-tip">
        <InfoTip text={row.model_basis || UNPRICED_REASON[row.q.kind] || 'Listed, not priced.'} place="top-left" />
      </span>
      <span
        className={`fut-ledger-dot${onLedger ? ' fut-ledger-dot-on' : ''}`}
        role="img"
        title={onLedger ? 'On the paper ledger — frozen the first day this disagreement was seen.' : 'Not on the paper ledger.'}
        aria-label={onLedger ? 'On the paper ledger' : 'Not on the paper ledger'}
      />
      <SourceLink url={row.url} />
    </div>
  );
}

function UnpricedGrid({ title, rows, reason }) {
  const [all, setAll] = useState(false);
  const shown = all ? rows : rows.slice(0, UNPRICED_PREVIEW);
  return (
    <div className="fut-unpriced">
      <div className="fut-unpriced-head">
        <h3 className="fut-unpriced-title">{title}</h3>
        <span className="fut-unpriced-tag">listed, not priced</span>
      </div>
      {rows.length === 0 ? (
        <p className="fut-muted">Nothing on this book right now.</p>
      ) : (
        <div className="board-grid fut-ugrid">
          {shown.map((r, i) => (
            <div className="board-cell fut-ucell" key={r.ticker}>
              <span className="fut-urank">{String(i + 1).padStart(2, '0')}</span>
              <Headshot espnId={r.matched_espn_id} teamId={r.matched_espn_team_id} size={36} />
              <span className="fut-uwho">
                <PlayerName name={r.matched_player} slug={r.slug} ambiguous={r.match_ambiguous} />
                <span className="fut-umeta">{reason}</span>
              </span>
              <span className="fut-uprice">{fmtCents(r.yes_price_cents)}</span>
              <SourceLink url={r.url} />
            </div>
          ))}
        </div>
      )}
      {rows.length > UNPRICED_PREVIEW && (
        <button type="button" className="fut-quiet-btn" onClick={() => setAll((v) => !v)}>
          {all ? 'Show fewer' : `Show all ${rows.length}`}
        </button>
      )}
    </div>
  );
}

function MethodSection({ coverage, scaleExample }) {
  const covPct = coverage != null ? Math.round(coverage * 100) : null;
  const scaleWord = scaleExample != null ? `“×${scaleExample.toFixed(2)} event scale”` : '“event scale”';
  const cells = [
    {
      title: 'What is priced',
      body: 'Only draft-position threshold questions — “top 5”, “before pick 10”. Each player’s conformal pick interval is read as log-uniform mass, and P(pick ≤ N) is the share of that mass below N.',
      tip: covPct != null
        ? `Model-implied P(pick ≤ N) from the ${covPct}% conformal pick interval, log-uniform mass; the ${100 - covPct}% outside the interval is split into the two tails.`
        : 'Model-implied P(pick ≤ N) from the conformal pick interval, log-uniform mass; the mass outside the interval is split into the two tails.',
    },
    {
      title: 'What is not',
      body: 'Exact-pick markets (“#1 overall”) and awards (Heisman) are listed with their price only. An exact pick isn’t identifiable from an interval, and an award isn’t a draft slot — so no model number is printed.',
      tip: 'not priced: exact pick isn’t identifiable from an interval · not priced: an award, not a draft slot.',
    },
    {
      title: 'Why the model looks low at the top',
      body: `Every top-5 book has five slots. When the matched players’ raw P(pick ≤ 5) sum past 5, each is scaled down so the event stays coherent — that is the ${scaleWord} under a rail. Wide intervals are conservative by construction.`,
      tip: 'scaled so the event’s matched P(pick ≤ N) sums to N (raw marginals overshot). The scale is printed on every rail it touched.',
    },
    {
      title: 'How it gets scored',
      body: 'A call is written the first day a 10-point disagreement is seen on a real two-sided quote — a last trade, or a midpoint on a tight or traded book. One-sided asks and placeholder books are shown, never recorded. On draft night each call gets a Brier score for the model and for the market.',
      tip: 'Brier = (probability − outcome)². Lower is closer. model Brier vs. market Brier, per call and on average; void markets are excluded.',
    },
  ];
  return (
    <section className="fut-section fut-method" aria-labelledby="fut-method-title">
      <h2 className="fut-h2" id="fut-method-title">How the number is made</h2>
      <div className="fut-method-grid">
        {cells.map((c) => (
          <div className="fut-method-cell" key={c.title}>
            <div className="pp-cell-label fut-method-label">
              {c.title}
              <InfoTip text={c.tip} />
            </div>
            <p className="fut-method-body">{c.body}</p>
          </div>
        ))}
      </div>
      <p className="fut-readonly">
        Read-only. DraftVision takes no positions, holds no account, and links to no order flow — the source link on each row is the public market page.
      </p>
      <p className="fut-legend">○ market price · ● model-implied probability · band = the gap</p>
    </section>
  );
}

function TableFallback({ rows }) {
  return (
    <details className="fut-table-details">
      <summary>Show as table</summary>
      <div className="fut-table-wrap">
        <table className="fut-table">
          <thead>
            <tr>
              <th className="fut-th-market">Market</th>
              <th>Price</th>
              <th>Model</th>
              <th>Gap</th>
              <th>Player</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => (
              <tr key={m.ticker}>
                <td className="fut-td-title">{m.title}</td>
                <td className="fut-td-num">
                  {fmtCents(m.yes_price_cents)}
                  {m.yes_price_source && m.yes_price_source !== 'last' && (
                    <span className="fut-src-tag">{SOURCE_LABEL[m.yes_price_source] || m.yes_price_source}</span>
                  )}
                </td>
                <td className="fut-td-num">
                  {m.model_prob != null
                    ? <span className="fut-model" title={m.model_basis || undefined}>{fmtPct1(m.model_prob)}</span>
                    : <span className="fut-muted" title={m.model_basis || undefined}>—</span>}
                </td>
                <td className="fut-td-num">{m.gap != null ? `${fmtGap(m.gap)} pts` : '—'}</td>
                <td><PlayerName name={m.matched_player} slug={m.slug} ambiguous={m.match_ambiguous} /></td>
                <td><SourceLink url={m.url} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

function RecordSection({ ledger, ledgerErrored, marketMap, boardBig, boardRecorded, recordCells, settlesOn, scoring }) {
  const entries = useMemo(() => {
    const raw = ledger?.entries || [];
    return raw.map((e) => {
      const mm = marketMap.get(e.ticker) || null;
      const team = e.team || mm?.matched_team || null;
      const slug = e.slug || (team && e.player ? playerSlug(e.player, team) : mm?.slug || null);
      const q = e.question && e.question.kind ? e.question : (mm ? mm.q : questionOf({ ticker: e.ticker, title: e.title }));
      const current = e.current_price_cents != null ? e.current_price_cents : (mm ? mm.yes_price_cents : null);
      const gap = e.edge != null ? Number(e.edge) : (e.model_prob != null && e.market_price_cents != null ? e.model_prob - e.market_price_cents : null);
      return {
        ...e,
        team,
        slug,
        q,
        gap,
        current,
        espn_id: e.espn_id || mm?.matched_espn_id || null,
        espn_team_id: e.espn_team_id || mm?.matched_espn_team_id || null,
        resolution: e.resolution || null,
        ambiguous: mm ? Boolean(mm.match_ambiguous) : false,
      };
    });
  }, [ledger, marketMap]);

  const groups = useMemo(() => {
    const by = new Map();
    entries.forEach((e) => {
      const k = e.date || '—';
      if (!by.has(k)) by.set(k, []);
      by.get(k).push(e);
    });
    return [...by.entries()]
      .sort((a, b) => (a[0] < b[0] ? 1 : -1))
      .map(([date, rows]) => [date, rows.sort((a, b) => Math.abs(b.gap ?? 0) - Math.abs(a.gap ?? 0))]);
  }, [entries]);

  const days = daysUntil(settlesOn);
  const lastSync = fmtDay(scoring?.last_synced_at) !== '—' && scoring?.last_synced_at ? fmtDay(scoring.last_synced_at) : null;

  return (
    <section className="fut-section fut-record" id="record" aria-labelledby="fut-record-title">
      <h2 className="fut-h2" id="fut-record-title">The record</h2>
      <p className="fut-sub">
        Every 10-point disagreement on a real two-sided quote, frozen the day it first appears. No money moves. Scored on draft night against the settled market.
      </p>
      <Scoreboard
        cells={recordCells}
        foot={(
          <span className="fut-stats-meta">
            Ledger committed to git weekly{lastSync ? ` · last sync ${lastSync}` : ''}
            {settlesOn && days != null ? ` · settles ${fmtDay(settlesOn)} · ${days} days` : ''}
          </span>
        )}
      />

      {!ledger ? (
        <RailSkeleton rows={3} />
      ) : ledgerErrored ? (
        <p className="fut-muted fut-empty fut-ledger-err">
          The ledger is not answering right now. The calls already frozen still stand; they are committed to git weekly and will show again on the next successful pass.
        </p>
      ) : entries.length === 0 ? (
        <p className="fut-muted fut-empty">
          No recorded calls yet — a call is written the first day a 10-point disagreement is seen on a real two-sided quote.
        </p>
      ) : (
        groups.map(([date, rows]) => (
          <div className="fut-day" id={`record-${date}`} key={date}>
            <div className="fut-day-rail">
              <span className="fut-day-label">{fmtDay(date)}</span>
              <span className="hr fut-day-hr" aria-hidden="true" />
            </div>
            <div className="fut-board fut-board-record" role="list" aria-label={`Calls recorded ${fmtDay(date)}`}>
              {rows.map((e, i) => {
                const ghost = e.current != null && e.market_price_cents != null && Math.abs(e.current - e.market_price_cents) >= 3 ? e.current : null;
                const outcome = e.resolution?.outcome || null;
                return (
                  <div className="fut-row fut-row-record" key={`${e.ticker}-${e.date}`} role="listitem">
                    <span className="fut-cell-shot">
                      <Headshot espnId={e.espn_id} teamId={e.espn_team_id} size={28} />
                    </span>
                    <span className="fut-cell-who">
                      <span className="fut-who-line">
                        <PlayerName name={e.player} slug={e.slug} ambiguous={e.ambiguous} />
                        <QChip q={e.q} />
                      </span>
                    </span>
                    <span className="fut-cell-rail">
                      <GapBar
                        size="row"
                        labels
                        market={e.market_price_cents}
                        model={e.model_prob}
                        source={e.price_source}
                        threshold={THRESHOLD}
                        outcome={outcome}
                        ghost={ghost}
                        delay={i * 40}
                      />
                      {ghost != null && <span className="fut-rail-caption">now {fmtCents(ghost)}</span>}
                    </span>
                    <Triplet
                      market={e.market_price_cents}
                      source={e.price_source}
                      model={e.model_prob}
                      basis={e.model_basis}
                      gap={e.gap}
                    />
                    <span className="fut-cell-tip">
                      <InfoTip text={e.model_basis || 'Model-implied from the conformal pick interval at the time of the call.'} place="top-left" />
                    </span>
                    <StatusChip resolution={e.resolution} settlesOn={settlesOn} />
                  </div>
                );
              })}
            </div>
          </div>
        ))
      )}

      <p className="fut-foot">
        {boardBig} disagreements on the board today · {boardRecorded} eligible and recorded · the rest sit on wide or one-sided quotes and are never recorded.
      </p>
    </section>
  );
}

/* ── page ───────────────────────────────────────────────────────────────── */

const SEG = [
  { key: 'modeled', label: 'Priced' },
  { key: 'top_n', label: 'Top-5' },
  { key: 'pick_eq', label: '#1 overall' },
  { key: 'heisman', label: 'Heisman' },
  { key: 'all', label: 'All' },
];

export default function Edge() {
  const [data, setData] = useState(null);
  const [ledger, setLedger] = useState(null);
  const [polls, setPolls] = useState(0);
  const [seg, setSeg] = useState('modeled');
  const [bigOnly, setBigOnly] = useState(false);
  const [query, setQuery] = useState('');

  useEffect(() => {
    let alive = true;
    getLedger().then((d) => {
      if (!alive) return;
      // an error stays an error (never an empty ledger): the record cells print '—'
      setLedger(d || { state: 'error', error: 'fetch failed' });
    });
    return () => { alive = false; };
  }, []);

  // Fetch the board through the shared cache; while the server reports
  // 'warming', re-ask on a short timer (bounded) instead of showing the
  // seasonal empty state. edgeData caches a warming payload for only 4 s.
  useEffect(() => {
    let alive = true;
    let timer = null;
    getEdge().then((d) => {
      if (!alive) return;
      setData(d);
      if (d && d.note === 'warming' && polls < WARMING_MAX_POLLS) {
        timer = setTimeout(() => alive && setPolls((n) => n + 1), WARMING_POLL_MS);
      }
    });
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [polls]);

  const errored = Boolean(data && data.state === 'error');
  const warming = !errored && Boolean(data && data.note === 'warming' && polls < WARMING_MAX_POLLS);
  const loading = !data || warming;

  const rows = useMemo(() => (data?.markets || []).map(enrich), [data]);
  const summary = data?.summary || null;
  // a failed fetch has no discovery block — surface it as the upstream-error chip
  const discovery = data?.discovery || (errored ? { state: 'error', error: data.error || 'fetch failed' } : null);
  const seasonal = !loading && !errored && rows.length === 0;

  const modeled = useMemo(() => rows.filter((r) => r.priced).sort((a, b) => Math.abs(b.gap) - Math.abs(a.gap)), [rows]);
  // every row the model does not answer — price desc, no-quote rows ('—') last
  const unpriced = useMemo(
    () => rows.filter((r) => !r.priced).sort((a, b) => (b.yes_price_cents ?? -1) - (a.yes_price_cents ?? -1)),
    [rows],
  );
  const listedOnly = !loading && !errored && rows.length > 0 && modeled.length === 0;

  const ledgerErrored = Boolean(ledger && ledger.state === 'error');
  const ledgerTickers = useMemo(() => new Set((ledger?.entries || []).map((e) => e.ticker)), [ledger]);
  const marketMap = useMemo(() => new Map(rows.map((r) => [r.ticker, r])), [rows]);

  const eligibleBig = summary?.eligible_over_threshold ?? modeled.filter((r) => r.big && r.ledger_eligible).length;
  const onLedgerCount = summary?.on_ledger ?? rows.filter((r) => ledgerTickers.has(r.ticker) || r.call).length;
  const draftYear = summary?.draft_year ?? (rows.find((r) => r.year)?.year ?? 2027);

  // The headline: largest |gap| that is ledger-eligible and unambiguous.
  const headline = useMemo(() => {
    const top = summary?.top_gap_ticker ? modeled.find((r) => r.ticker === summary.top_gap_ticker) : null;
    return top || modeled.find((r) => r.ledger_eligible && !r.match_ambiguous) || null;
  }, [summary, modeled]);

  // One list per seg option; the '(N)' beside each label is that list's
  // length, so the count is always exactly what the board can show.
  const segLists = useMemo(() => {
    const all = [...modeled, ...unpriced];
    const lists = { modeled, all };
    SEG.forEach((s) => { if (!(s.key in lists)) lists[s.key] = all.filter((r) => r.q.kind === s.key); });
    return lists;
  }, [modeled, unpriced]);

  const boardRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    let list = segLists[seg] || [];
    if (bigOnly) list = list.filter((r) => r.big);
    if (q) list = list.filter((r) => (r.matched_player || '').toLowerCase().includes(q) || (r.title || '').toLowerCase().includes(q));
    return list;
  }, [seg, bigOnly, query, segLists]);

  // The interval's nominal coverage and a real event scale, read off the payload.
  const coverage = useMemo(() => {
    const hit = modeled.find((r) => r.interval && r.interval.coverage != null);
    return hit ? Number(hit.interval.coverage) : null;
  }, [modeled]);
  const scaleExample = useMemo(() => {
    const hit = modeled.find((r) => scaleBit(r.scale));
    return hit ? hit.scale : null;
  }, [modeled]);

  const pickOneBoard = useMemo(() => unpriced.filter((r) => r.q.kind === 'pick_eq' && (r.q.n == null || r.q.n === 1)), [unpriced]);
  const heismanBoard = useMemo(() => unpriced.filter((r) => r.q.kind === 'heisman'), [unpriced]);

  const lsum = ledger?.summary || null;
  const scoring = ledger?.scoring || null;
  const entryCount = ledger?.entries ? ledger.entries.length : null;
  const calls = lsum?.calls ?? entryCount;
  const resolved = lsum?.resolved ?? (ledger?.entries ? ledger.entries.filter((e) => e.resolution && e.resolution.outcome).length : null);
  const open = lsum?.open ?? (calls != null ? calls - resolved : null);
  const modelCloser = lsum?.model_closer ?? null;
  const settlesOn = scoring?.settles_on || null;
  const settleMonth = fmtMonthYear(settlesOn);
  const settleDays = daysUntil(settlesOn);
  const brierValue = (
    <span className="fut-brier">
      {fmtBrier(lsum?.model_brier)} <span className="fut-brier-sep">/</span> {fmtBrier(lsum?.market_brier)}
    </span>
  );
  const brierSub = settlesOn
    ? `scored draft night · ${settleMonth}${settleDays != null ? ` · ${settleDays} days` : ''}`
    : 'scored on draft night';

  const haveBoard = Boolean(summary) && !loading && !errored;
  const topCells = [
    { value: haveBoard ? summary.markets : '—', label: 'Markets' },
    { value: haveBoard ? summary.modeled : '—', label: 'Priced by the model' },
    { value: haveBoard ? eligibleBig : '—', label: 'Disagreements ≥ 10 pts' },
    { value: haveBoard ? onLedgerCount : '—', label: 'On the paper ledger', tip: LEDGER_TIP },
    { value: brierValue, label: 'Model Brier / Market Brier', sub: brierSub },
  ];
  const recordCells = [
    { value: calls ?? '—', label: 'Calls' },
    { value: open ?? '—', label: 'Open' },
    { value: resolved ?? '—', label: 'Resolved' },
    {
      value: resolved > 0 && modelCloser != null ? `${modelCloser} of ${resolved}` : '— of —',
      label: 'Model closer',
    },
    { value: brierValue, label: 'Model Brier / Market Brier', sub: settlesOn ? `settles ${fmtDay(settlesOn)}${settleDays != null ? ` · ${settleDays} days` : ''}` : 'scored on draft night' },
  ];

  const fetchedAt = fmtTime(discovery?.fetched_at_utc);
  const boardBig = modeled.filter((r) => r.big).length;
  const boardRecorded = modeled.filter((r) => r.big && (ledgerTickers.has(r.ticker) || r.call)).length;

  const unpricedSection = !seasonal && !errored && !loading ? (
    <section className="fut-section" aria-labelledby="fut-unpriced-title">
      <h2 className="fut-h2" id="fut-unpriced-title">What the market thinks, unpriced</h2>
      <p className="fut-sub">Books the model does not answer. The price is the market’s alone.</p>
      <div className="fut-unpriced-grids">
        <UnpricedGrid title="The market’s #1-overall board" rows={pickOneBoard} reason={UNPRICED_REASON.pick_eq} />
        <UnpricedGrid title="The market’s Heisman board" rows={heismanBoard} reason={UNPRICED_REASON.heisman} />
      </div>
    </section>
  ) : null;

  const lastCycle = calls != null
    ? `${calls} calls${resolved > 0 && modelCloser != null ? ` · model closer ${modelCloser} of ${resolved}` : ' · model closer — of —'}`
    : '—';

  return (
    <div className="fut-page">
      {/* ── (1) Hero ── */}
      <header className="fut-hero">
        <div className="fut-hero-media" aria-hidden="true">
          <img src={HERO_IMG} alt="" />
        </div>
        <div className="fut-hero-tint" aria-hidden="true" />
        <div className="fut-hero-scrim-x" aria-hidden="true" />
        <div className="fut-hero-scrim-y" aria-hidden="true" />
        <div className="fut-hero-inner">
          <div className="fut-hero-text">
            <p className="eyebrow">Model vs. Market · {draftYear} NFL Draft</p>
            <h1 className="fut-title">Where the model and the market disagree</h1>
            <p className="fut-lede">
              Kalshi prices the {draftYear} draft in public. The model prices it too. This page lines the two up — read-only, no positions, every 10-point disagreement frozen on paper and scored on draft night.
            </p>
            <p className="fut-legal">
              Not investment advice. Prediction markets are restricted to adults 18+. Source links open kalshi.com.
            </p>
          </div>
          <HeadlineCard
            row={headline}
            seasonal={seasonal}
            warming={loading}
            errored={errored}
            listedOnly={listedOnly}
            marketCount={rows.length}
            recordCells={recordCells}
          />
        </div>
      </header>

      <div className="fut-inner">
        <Scoreboard
          className="fut-scoreboard-top"
          cells={topCells}
          foot={(
            <>
              <span className="fut-stats-meta">
                {fetchedAt ? `prices as of ${fetchedAt}` : loading ? 'reading the market…' : 'no prices'}
                <DiscoveryChip discovery={discovery} />
              </span>
              <DiscoveryBanner discovery={discovery} />
            </>
          )}
        />

        {seasonal && (
          <div className="fut-ink-strip">
            <span className="fut-ink-eyebrow">Between cycles</span>
            <span className="fut-ink-text">Draft futures reopen after the draft — last cycle’s record: {lastCycle}.</span>
          </div>
        )}

        {listedOnly && (
          <p className="fut-banner">
            <strong>Listed, not priced.</strong> These markets are on the book, but no row maps onto a model output right now — hover a dash for why. Nothing here is a gap.
          </p>
        )}

        {/* listed-only: the market's boards move up under the hero */}
        {listedOnly && unpricedSection}

        {/* ── (2) How the number is made — always visible, before the board ── */}
        <MethodSection coverage={coverage} scaleExample={scaleExample} />

        {/* ── (3) Where they disagree ── */}
        {!seasonal && !errored && !listedOnly && (
          <section className="fut-section" aria-labelledby="fut-board-title">
            <div className="fut-section-head">
              <div>
                <h2 className="fut-h2" id="fut-board-title">Where they disagree</h2>
                <p className="fut-sub">One row per market, largest gap first. Rows under 10 points sit back; they roughly agree.</p>
              </div>
            </div>
            <div className="fut-controls">
              <div className="seg fut-seg" role="radiogroup" aria-label="Which markets">
                {SEG.map((s) => {
                  const count = (segLists[s.key] || []).length;
                  return (
                    <label className="seg-opt" key={s.key}>
                      <input type="radio" name="fut-seg" value={s.key} checked={seg === s.key} onChange={() => setSeg(s.key)} />
                      {s.label} {!loading && <span className="fut-seg-count">({count})</span>}
                    </label>
                  );
                })}
              </div>
              <label className="radio fut-toggle">
                <input type="checkbox" checked={bigOnly} onChange={(e) => setBigOnly(e.target.checked)} />
                <span className="dot" aria-hidden="true" />
                Disagreements only (≥ 10 pts)
              </label>
              <input
                className="input fut-search"
                type="search"
                placeholder="Search player or market"
                aria-label="Search player or market"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>

            {loading ? (
              <RailSkeleton rows={6} />
            ) : (
              <>
                <p className="fut-sr-only" id="fut-board-cols">
                  Each row: rank, player and market, the gap rail from 0 to 100, market price, model-implied probability, gap (model minus market), ledger status, source link.
                </p>
                <div className="fut-board-head" aria-hidden="true">
                  <span className="fut-rank" />
                  <span className="fut-cell-shot" />
                  <span className="fut-cell-who">market · player</span>
                  <span className="fut-axis">
                    <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
                  </span>
                  <span className="fut-num">market</span>
                  <span className="fut-num">model</span>
                  <span className="fut-num">gap (model − market)</span>
                  <span className="fut-cell-tip" />
                  <span />
                  <span />
                </div>
                <div className="fut-board" role="list" aria-describedby="fut-board-cols">
                  {boardRows.length === 0 ? (
                    <p className="fut-muted fut-empty">No rows match.</p>
                  ) : (
                    boardRows.map((r, i) => (
                      <BoardRow
                        key={r.ticker}
                        row={r}
                        rank={i + 1}
                        onLedger={ledgerTickers.has(r.ticker) || Boolean(r.call)}
                        delay={Math.min(i, 12) * 35}
                      />
                    ))
                  )}
                </div>
              </>
            )}
          </section>
        )}

        {/* ── (4) What the market thinks, unpriced ── */}
        {!listedOnly && unpricedSection}

        {/* ── (5) The record ── */}
        <RecordSection
          ledger={ledger}
          ledgerErrored={ledgerErrored}
          marketMap={marketMap}
          boardBig={boardBig}
          boardRecorded={boardRecorded}
          recordCells={recordCells}
          settlesOn={settlesOn}
          scoring={scoring}
        />

        {/* ── (6) Dense table for people who want to copy numbers ── */}
        {!loading && rows.length > 0 && <TableFallback rows={[...modeled, ...unpriced]} />}
      </div>
    </div>
  );
}

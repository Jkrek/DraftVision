import React, { useEffect, useMemo, useState } from 'react';
import InfoTip from '../InfoTip';
import './Backtest.css';

/* Backtest — the receipts page. Renders public/data/backtest.json:
   what the production ensemble would have said about the 2019-2020 draft
   classes, which were held out of training entirely, measured against the
   comparator that actually matters (the consensus big board), with the full
   per-player ledger downloadable so every number can be re-derived. Misses
   are shown on purpose — that is the point of the page. */

const PUBLIC = process.env.PUBLIC_URL || '';
const PAGE_SIZE = 100;

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
  {
    key: 'all',
    label: 'All players',
    blurb: 'Every scored holdout player, nothing curated. Click a column to sort; the CSV is the same rows.',
  },
];

/* Sort keys for the full-ledger tab. `dir` is the natural direction for a
   first click; nulls always sink to the bottom whichever way you sort. */
const LEDGER_SORTS = {
  prob:      { get: (r) => r.pred_success_prob, dir: 'desc' },
  pred_pick: { get: (r) => r.pred_pick,         dir: 'asc'  },
  consensus: { get: (r) => r.consensus_rank,    dir: 'asc'  },
  actual:    { get: (r) => r.actual_pick,       dir: 'asc'  },
  av:        { get: (r) => r.career_av,         dir: 'desc' },
  name:      { get: (r) => r.name,              dir: 'asc'  },
  year:      { get: (r) => r.draft_year,        dir: 'asc'  },
};

function fmtPct(p) {
  return `${(p * 100).toFixed(1)}%`;
}

function fmt(v, digits = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

function pm(v, digits = 3) {
  return v ? `± ${Number(v).toFixed(digits)}` : '';
}

function MetricRow({ label, tip, model, other, better, digits = 4, note }) {
  return (
    <tr>
      <td className="bt-metric-label">
        <span className="bt-metric-name">
          {label}
          {tip && <InfoTip text={tip} place="bottom" />}
        </span>
        <span className="bt-metric-note">{better}{note ? ` · ${note}` : ''}</span>
      </td>
      <td className="bt-num bt-num-model">{fmt(model, digits)}</td>
      <td className="bt-num">{fmt(other, digits)}</td>
    </tr>
  );
}

/* Honest verdict on the AUC gap, driven by the numbers so it stays true
   after a regeneration. Two comparisons: all covered players (the boards'
   "passed" players ranked last) and only the players the board ranked. The
   rolling-CV fold std is the noise yardstick. */
function gapWord(d, noise) {
  if (d < -0.01) return 'behind';
  if (Math.abs(d) <= 0.01) return 'tied with';
  if (d < noise) return 'narrowly ahead of';
  return 'ahead of';
}

function aucVerdict(c, modelAll, modelRanked, foldStd) {
  const noise = foldStd || 0.02;
  const dAll = modelAll - c.auc;
  const dRk = modelRanked - c.auc_ranked_only;
  const pts = (d) => `${d >= 0 ? '+' : ''}${(d * 100).toFixed(1)}`;
  const all = `Across all ${c.n_auc} players the model is ${gapWord(dAll, noise)} the board on success ranking (${modelAll.toFixed(3)} vs ${c.auc.toFixed(3)}, ${pts(dAll)} AUC points).`;
  let ranked;
  if (dRk < -0.01) {
    ranked = `On the ${c.n_ranked} players the board actually ranked, the board wins (${c.auc_ranked_only.toFixed(3)} vs ${modelRanked.toFixed(3)}): the model does not out-scout the consensus where the consensus has an opinion.`;
  } else if (Math.abs(dRk) <= 0.01) {
    ranked = `On the ${c.n_ranked} players the board actually ranked, it is a statistical tie (${modelRanked.toFixed(3)} vs ${c.auc_ranked_only.toFixed(3)}): the model does not out-scout the consensus where the consensus has an opinion.`;
  } else if (dRk < noise) {
    ranked = `On the ${c.n_ranked} players the board actually ranked, the gap shrinks to ${pts(dRk)} points (${modelRanked.toFixed(3)} vs ${c.auc_ranked_only.toFixed(3)}) — smaller than the fold-to-fold spread in the rolling CV below (± ${noise.toFixed(3)}). Where the scouts have an opinion, the model is roughly even with them, not better.`;
  } else {
    ranked = `On the ${c.n_ranked} players the board actually ranked, the model stays ahead by ${pts(dRk)} points (${modelRanked.toFixed(3)} vs ${c.auc_ranked_only.toFixed(3)}), more than the fold-to-fold spread in the rolling CV below (± ${noise.toFixed(3)}).`;
  }
  return `${all} ${ranked}`;
}

/* Who wins each pick-projection metric — counted, not asserted. */
function pickScoreboard(pk, c) {
  if (!pk || !c || !c.pick) return null;
  const cp = c.pick;
  const items = [
    { label: 'overall rank correlation', m: pk.spearman_all, o: cp.spearman_all, higher: true },
    { label: 'top-64 ordering', m: pk.spearman_top64, o: cp.spearman_top64, higher: true },
    { label: 'first-rounders flagged', m: pk.r1_recall_within_45, o: cp.r1_recall_within_45, higher: true },
    { label: 'average miss on drafted players', m: cp.model_mae_same_rows, o: cp.mae_picks_drafted_ranked, higher: false, eps: 0.5 },
  ].filter((i) => i.m != null && i.o != null);
  const epsDefault = 0.005; // correlations / recall; MAE rows carry their own (picks)
  const model = []; const board = []; const tie = [];
  items.forEach((i) => {
    const d = (i.m - i.o) * (i.higher ? 1 : -1);
    const eps = i.eps ?? epsDefault;
    if (Math.abs(d) <= eps) tie.push(i.label); else if (d > 0) model.push(i.label); else board.push(i.label);
  });
  const list = (a) => a.join(', ');
  const parts = [];
  if (model.length) parts.push(`the model wins ${model.length} (${list(model)})`);
  if (board.length) parts.push(`the board wins ${board.length} (${list(board)})`);
  if (tie.length) parts.push(`${tie.length === 1 ? 'one is' : `${tie.length} are`} a tie (${list(tie)})`);
  return `On slotting players into the draft order the two trade blows across ${items.length} metrics: ${parts.join('; ')}.`;
}

/* The worst over-confident bin with enough players to mean anything, plus
   any bin that misses by more than 5 points in the OTHER direction — the
   "everything else tracks" sentence is computed on |gap|, not assumed. */
function reliabilityNote(rel) {
  if (!rel) return null;
  const solid = rel.filter((r) => r.count >= 20 && r.mean_predicted != null);
  if (!solid.length) return null;
  const gapOf = (r) => (r.fraction_positive - r.mean_predicted) * 100;
  const binLabel = (r) => r.bin.replace(/[[)\]]/g, '').replace(',', '–');
  const worst = solid.reduce((a, r) => (gapOf(r) < gapOf(a) ? r : a));
  const gap = gapOf(worst);
  if (gap > -5) return 'No bin with 20+ players is over-confident by more than 5 points.';
  const over = solid.filter((r) => gapOf(r) < -5).map(binLabel);
  const under = solid.filter((r) => gapOf(r) > 5);
  const rest = under.length
    ? `The ${under.map((r) => `${binLabel(r)} bin (+${gapOf(r).toFixed(0)} points, n=${r.count})`).join(' and ')} ${under.length === 1 ? 'misses' : 'miss'} the other way — under-confident — and every remaining bin with 20+ players tracks within 5 points.`
    : 'Every other bin with 20+ players tracks within 5 points.';
  return `The model is over-confident once it gets bullish: in the ${over.join(', ')} ranges, players hit ${Math.abs(gap).toFixed(0)} points less often than predicted at worst (${binLabel(worst)}, n=${worst.count}). ${rest} A mid-range call is closer to a coin flip than the number says.`;
}

export default function Backtest() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('top20');
  const [sortKey, setSortKey] = useState('prob');
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(0);

  useEffect(() => {
    let alive = true;
    fetch(`${PUBLIC}/data/backtest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json) => { if (alive) setData(json); })
      .catch((e) => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, []);

  const rows = useMemo(() => {
    if (!data || tab === 'all') return [];
    const list = data.players.filter((p) => p.categories.includes(tab));
    if (tab === 'fade') return [...list].sort((a, b) => a.pred_success_prob - b.pred_success_prob);
    return list; // already sorted by predicted probability, descending
  }, [data, tab]);

  const ledger = useMemo(() => {
    if (!data || !data.ledger) return [];
    const { get } = LEDGER_SORTS[sortKey];
    const sign = sortDir === 'asc' ? 1 : -1;
    return [...data.ledger].sort((a, b) => {
      const va = get(a); const vb = get(b);
      const na = va === null || va === undefined; const nb = vb === null || vb === undefined;
      if (na && nb) return 0;
      if (na) return 1;
      if (nb) return -1;
      if (typeof va === 'string') return sign * va.localeCompare(vb);
      return sign * (va - vb);
    });
  }, [data, sortKey, sortDir]);

  const onSort = (key) => {
    if (key === sortKey) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir(LEDGER_SORTS[key].dir);
    }
    setPage(0);
  };

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
  const c = m.consensus;
  const pk = m.pick ? m.pick.served_blend : null;
  const rc = data.rolling_cv;
  const fwd = data.forward_split;
  const foldStd = rc && rc.summary && rc.summary.success_auc ? rc.summary.success_auc.std : null;
  const modelAucSame = c && c.model_auc_same_rows != null ? c.model_auc_same_rows : m.auc;
  const pageCount = Math.max(1, Math.ceil(ledger.length / PAGE_SIZE));
  const pageRows = ledger.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const csvHref = `${PUBLIC}${data.ledger_csv || '/data/backtest_predictions.csv'}`;

  const ledgerHeaders = [
    { label: 'Player',        cls: 'bt-c-name',   sort: 'name' },
    { label: 'Class',         cls: 'bt-c-year',   sort: 'year' },
    { label: 'Pred. success', cls: 'bt-c-prob',   sort: 'prob' },
    { label: 'Pred. bucket',  cls: 'bt-c-bucket', sort: null },
    { label: 'Pred. pick',    cls: 'bt-c-range',  sort: 'pred_pick',
      tip: 'The served pick projection, with its 80% range underneath — the conformal interval from the same eval-phase recipe train_models.py reports. The point comes from the blended pick regressor and the range from separate q10/q90 quantile heads, so for a handful of players the range does not contain the point — that is a real disagreement between the heads, shown as-is. Undrafted is written as 300+.' },
    { label: 'Consensus',     cls: 'bt-c-pick',   sort: 'consensus',
      tip: 'Rank on the consensus big board (WideLeft/ESPN staged boards) before the draft. A dash means the boards passed on the player.' },
    { label: 'Actual',        cls: 'bt-c-pick',   sort: 'actual' },
    { label: 'Outcome',       cls: 'bt-c-outcome', sort: null },
    { label: 'Career AV',     cls: 'bt-c-av',     sort: 'av',
      tip: 'Pro-Football-Reference Approximate Value accumulated in the NFL — the continuous career-value label the career head is trained on.' },
  ];

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
              The misses are shown on purpose, and so is the comparison that matters: not
              a hand-written heuristic, but the <strong>consensus big board</strong> the
              scouting industry published before those drafts. Every row scored is in
              the downloadable ledger, so nothing here has to be taken on trust.
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
        {/* ── Headline: model vs the consensus big board ── */}
        <section aria-label="Model vs the consensus big board">
          <div className="bt-divider">Model vs the consensus big board &mdash; same {c ? c.n_auc : m.holdout_rows} holdout players</div>
          <div className="bt-metrics-wrap bt-metrics-wide">
            <table className="bt-metrics">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="bt-num-h">Model</th>
                  <th className="bt-num-h">Consensus board</th>
                </tr>
              </thead>
              <tbody>
                <MetricRow
                  label="AUC (success ranking)"
                  tip="Probability that a randomly chosen eventual NFL success is ranked above a randomly chosen bust. The board is scored by its rank (lower is better); players the boards passed on count as ranked below everyone."
                  better="higher is better"
                  model={modelAucSame}
                  other={c ? c.auc : null}
                />
                {c && c.auc_ranked_only != null && (
                  <MetricRow
                    label="AUC, board-ranked players only"
                    tip={`The same comparison restricted to the ${c.n_ranked} players the consensus board ranked, so neither side gets credit or blame for the players the boards passed on.`}
                    better="higher is better"
                    note={`same ${c.n_ranked} players`}
                    model={c.model_auc_ranked_only}
                    other={c.auc_ranked_only}
                  />
                )}
                <MetricRow
                  label="Brier score (success)"
                  tip="Mean squared error of the probability against the 0/1 outcome — measures calibration as well as ranking. A rank is not a probability, so the board has no Brier score."
                  better="lower is better"
                  model={m.brier}
                  other={null}
                  note="board has no probabilities"
                />
                {pk && c && (
                  <>
                    <MetricRow
                      label="Rank correlation vs actual pick"
                      tip="Spearman correlation between the projected order and the real draft order across every scored player, undrafted counted as pick 300 (board: passed = rank 400)."
                      better="higher is better"
                      model={pk.spearman_all}
                      other={c.pick.spearman_all}
                    />
                    <MetricRow
                      label="Rank correlation, top 64"
                      tip="The same correlation restricted to players who actually went in the first two rounds — does it order the top of the draft."
                      better="orders the top two rounds, higher is better"
                      model={pk.spearman_top64}
                      other={c.pick.spearman_top64}
                    />
                    <MetricRow
                      label="1st-rounders flagged by pick 45"
                      tip="Share of actual first-round picks the projection placed at pick 45 or earlier."
                      better="share of actual R1s, higher is better"
                      model={pk.r1_recall_within_45}
                      other={c.pick.r1_recall_within_45}
                    />
                    <MetricRow
                      label="Avg. miss on drafted players"
                      tip={`Mean absolute error in picks, on the ${c.pick.n_mae} drafted players the board ranked 1-262 — the board only names a pick for players it ranks, so both sides are scored on exactly those rows.`}
                      better="picks off, lower is better"
                      note={`same ${c.pick.n_mae} players`}
                      digits={1}
                      model={c.pick.model_mae_same_rows}
                      other={c.pick.mae_picks_drafted_ranked}
                    />
                  </>
                )}
              </tbody>
            </table>

            {c && (
              <div className="bt-verdict">
                <p>{aucVerdict(c, modelAucSame, c.model_auc_ranked_only != null ? c.model_auc_ranked_only : modelAucSame, foldStd)}</p>
                <p>
                  Two things to keep in mind before reading that as a win or a loss. First, the
                  board is an <strong>input</strong> to the model &mdash; consensus rank is one of
                  its features &mdash; so this is not the model against an independent source; it is
                  &ldquo;consensus plus everything else&rdquo; against consensus alone. Second, most of
                  the all-player gap comes from the {c.n_auc - c.n_ranked} players the boards passed on,
                  who are scored as ranked below everyone; the model still has to put a number on them,
                  and it does so better than &ldquo;last&rdquo;.
                </p>
                {pickScoreboard(pk, c) && <p>{pickScoreboard(pk, c)}</p>}
                <p>
                  What the model adds is not a better eye. It scores the roughly 12&ndash;15 thousand
                  players in the app the same way, when a big board names a few hundred; it puts a
                  calibrated probability on each one instead of an ordinal; it attaches an 80% pick
                  range ({m.pick_interval ? `${(m.pick_interval.coverage * 100).toFixed(0)}% actual coverage on this holdout, median width ${Math.round(m.pick_interval.median_width_picks)} picks` : 'see the ledger'}) and a
                  career-value estimate; and it keeps doing so for players the boards passed on.
                </p>
              </div>
            )}

            <p className="bt-metrics-foot">
              For the record, the app&apos;s hand-written heuristic fallback scores AUC {fmt(b.auc, 3)},
              Brier {fmt(b.brier, 3)} and draft-grade accuracy {fmt(b.accuracy, 3)} on the same
              holdout (model: {fmt(m.accuracy, 3)}); the 4-bucket grade head alone projects picks
              with an average miss of {m.pick ? fmt(m.pick.classifier_baseline.mae_picks_drafted, 1) : '—'} vs
              the served blend&apos;s {pk ? fmt(pk.mae_picks_drafted, 1) : '—'} over all {m.pick ? m.pick.n_scored : '—'} scored
              players. Those are floors, not comparators. Base rate: only {fmtPct(m.holdout_success_rate)} of
              these {m.holdout_rows} players became NFL successes at all.
            </p>
          </div>
        </section>

        {/* ── Reliability ── */}
        {m.reliability && (
          <section aria-label="Reliability">
            <div className="bt-divider">Does 30% mean 30%? &mdash; reliability on the holdout</div>
            <div className="bt-metrics-wrap bt-metrics-wide">
              <table className="bt-metrics bt-compact">
                <thead>
                  <tr>
                    <th>Predicted range</th>
                    <th className="bt-num-h">Players</th>
                    <th className="bt-num-h">Mean predicted</th>
                    <th className="bt-num-h">Actually hit</th>
                    <th className="bt-num-h">Gap</th>
                  </tr>
                </thead>
                <tbody>
                  {m.reliability.map((r) => {
                    const gap = r.count ? r.fraction_positive - r.mean_predicted : null;
                    return (
                      <tr key={r.bin} className={r.count < 20 ? 'bt-thin' : ''}>
                        <td className="bt-metric-label">{r.bin.replace(/[[)\]]/g, '').replace(',', ' – ')}</td>
                        <td className="bt-num">{r.count}</td>
                        <td className="bt-num bt-num-model">{r.count ? fmtPct(r.mean_predicted) : '—'}</td>
                        <td className="bt-num">{r.count ? fmtPct(r.fraction_positive) : '—'}</td>
                        <td className={`bt-num ${gap == null ? '' : gap < 0 ? 'bt-bust' : 'bt-hit'}`}>
                          {gap == null ? '—' : `${gap >= 0 ? '+' : ''}${(gap * 100).toFixed(1)} pts`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="bt-metrics-foot">
                Players are binned by the probability the model gave them; a well-calibrated model
                sees &ldquo;actually hit&rdquo; track &ldquo;mean predicted&rdquo;. A negative gap means the
                model was over-confident in that range. Bins with fewer than 20 players are dimmed &mdash;
                their observed rates are mostly noise. {reliabilityNote(m.reliability)}
              </p>
            </div>
          </section>
        )}

        {/* ── Rolling-origin CV ── */}
        {rc && rc.folds && rc.folds.length > 0 && (
          <section aria-label="Rolling-origin cross-validation">
            <div className="bt-divider">Six drafts, not one</div>
            <p className="bt-tab-blurb">
              One holdout is one roll of the dice, so the same training recipe was re-run as a
              rolling-origin CV with a fresh set of models per fold: for each test year Y, retrain on
              2000 through Y&minus;2, calibrate on Y&minus;1, score Y cold &mdash; six separate drafts, each
              seen only after everything before it. Only the 2019 fold uses the same fit as the holdout
              above; the 2020 fold trains through 2018 and calibrates on 2019, so its row is a different
              set of models scoring that class, not the holdout models re-scored.
            </p>
            <div className="bt-metrics-wrap bt-metrics-wide">
              <table className="bt-metrics bt-compact">
                <thead>
                  <tr>
                    <th>Test year</th>
                    <th className="bt-num-h">Players</th>
                    <th className="bt-num-h">AUC</th>
                    <th className="bt-num-h">Brier</th>
                    <th className="bt-num-h">Grade acc.</th>
                    <th className="bt-num-h">Pick MAE</th>
                    <th className="bt-num-h">Spearman</th>
                  </tr>
                </thead>
                <tbody>
                  {rc.folds.map((f) => (
                    <tr key={f.test_year}>
                      <td className="bt-metric-label">
                        {f.test_year}
                        <span className="bt-metric-note">train {f.train_years ? `${f.train_years[0]}–${f.train_years[1]}` : '—'}, cal {f.cal_year}</span>
                      </td>
                      <td className="bt-num">{f.n_test}</td>
                      <td className="bt-num bt-num-model">{fmt(f.success_auc, 3)}</td>
                      <td className="bt-num">{fmt(f.success_brier, 3)}</td>
                      <td className="bt-num">{fmt(f.grade_acc, 3)}</td>
                      <td className="bt-num">{fmt(f.pick_mae_drafted, 1)}</td>
                      <td className="bt-num">{fmt(f.pick_spearman_all, 3)}</td>
                    </tr>
                  ))}
                  {rc.summary && (
                    <tr className="bt-summary-row">
                      <td className="bt-metric-label">Mean &plusmn; std</td>
                      <td className="bt-num">{rc.folds.reduce((s, f) => s + (f.n_test || 0), 0)}</td>
                      <td className="bt-num bt-num-model">{fmt(rc.summary.success_auc.mean, 3)} <span className="bt-pm">{pm(rc.summary.success_auc.std)}</span></td>
                      <td className="bt-num">{fmt(rc.summary.success_brier.mean, 3)} <span className="bt-pm">{pm(rc.summary.success_brier.std)}</span></td>
                      <td className="bt-num">{fmt(rc.summary.grade_acc.mean, 3)} <span className="bt-pm">{pm(rc.summary.grade_acc.std)}</span></td>
                      <td className="bt-num">{fmt(rc.summary.pick_mae_drafted.mean, 1)} <span className="bt-pm">{pm(rc.summary.pick_mae_drafted.std, 1)}</span></td>
                      <td className="bt-num">{fmt(rc.summary.pick_spearman_all.mean, 3)} <span className="bt-pm">{pm(rc.summary.pick_spearman_all.std)}</span></td>
                    </tr>
                  )}
                </tbody>
              </table>
              <p className="bt-metrics-foot">
                Source: {rc.source}. Feature set {rc.feature_set}; z-score references are recomputed per
                fold and never include the test year. The 2019 and 2020 folds cover the same players as
                the holdout above, each scored by its own fold&apos;s models.
              </p>
            </div>
          </section>
        )}

        {/* ── Forward-looking disclosure ── */}
        <section aria-label="What to expect going forward">
          <div className="bt-divider">What to expect going forward</div>
          <div className="bt-disclosure">
            {fwd ? (
              <>
                <p>
                  The 2019&ndash;20 holdout sits in the densest, best-covered part of the data. The one
                  deployment-realistic test on record &mdash; <strong>train through 2023, calibrate on 2024,
                  score the 2025&ndash;26 classes cold</strong> ({fwd.n_test} players) &mdash; put draft-grade
                  accuracy at <strong>{fmt(fwd.grade_accuracy_raw, 4)}</strong> raw
                  ({fmt(fwd.grade_accuracy_calibrated, 4)} calibrated, macro-F1 {fmt(fwd.macro_f1_raw, 4)}),
                  against {fmt(fwd.frozen_grade_accuracy_same_features, 4)} for the same feature stack on the
                  frozen 2019&ndash;20 split &mdash; a drop of roughly {Math.round((fwd.frozen_grade_accuracy_same_features - fwd.grade_accuracy_raw) * 100)} points.
                  That forward run used the v3 feature stack &mdash; {fwd.feature_stack.split(' — ')[0].replace(/^v3 winner \(/, '').replace(/\)$/, '')} &mdash; and the
                  consensus-board and all-star features that lift the frozen accuracy to {fmt(m.accuracy, 4)} were
                  added afterwards and have <strong>not</strong> been forward-tested the same way. So the honest
                  expectation for a brand-new class is closer to the mid-0.40s on grade accuracy than to the
                  frozen number, until a forward split of the current stack says otherwise.
                </p>
                <p>
                  {data.selection_note} Nothing here was trained on these players, but the numbers on
                  this page are the ones that survived that selection, which biases them upward. The rolling
                  CV above is the better estimate of typical performance; the forward split is the better
                  estimate of the next draft.
                </p>
                <p className="bt-metrics-foot">Source: {fwd.source} (labels &ldquo;+A+B+C+D|flat|fwd&rdquo; and &ldquo;+A+B+C+D|flat|hist&rdquo;), summarised in models/experiments/RESULTS.md and models/metadata.json evaluation.note.</p>
              </>
            ) : (
              <p>{data.selection_note || 'The 2019–20 holdout was also the model-selection split for every experiment, so its numbers carry selection optimism.'}</p>
            )}
          </div>
        </section>

        {/* ── Receipts ── */}
        <section aria-label="Notable calls">
          <div className="bt-divider">The receipts</div>

          <div className="bt-tabbar">
            <div className="bt-tabs" role="tablist" aria-label="Call categories">
              {TABS.map((t) => (
                <button
                  key={t.key}
                  role="tab"
                  aria-selected={tab === t.key}
                  className={`bt-tab${tab === t.key ? ' bt-tab-active' : ''}`}
                  onClick={() => { setTab(t.key); setPage(0); }}
                >
                  {t.key === 'all' && data.ledger ? `All ${data.ledger.length}` : t.label}
                </button>
              ))}
            </div>
            <a className="bt-btn" href={csvHref} download="backtest_predictions.csv">
              Download the ledger (CSV)
            </a>
          </div>
          <p className="bt-tab-blurb">{active.blurb}</p>

          {tab !== 'all' ? (
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
                  <span className="bt-c-pick">{p.pred_pick ? `~#${Math.round(p.pred_pick)}` : '—'}</span>
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
          ) : (
            <>
              <div className="bt-table" role="table">
                <div className="bt-row bt-row-head" role="row">
                  {ledgerHeaders.map((h) => (
                    <span
                      key={h.label}
                      className={`${h.cls} bt-thc`}
                      role="columnheader"
                      aria-sort={h.sort ? (sortKey === h.sort ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none') : undefined}
                    >
                      {h.sort ? (
                        <button
                          type="button"
                          className={`bt-th${sortKey === h.sort ? ' active' : ''}`}
                          onClick={() => onSort(h.sort)}
                        >
                          {h.label}{sortKey === h.sort ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''}
                        </button>
                      ) : (
                        <span>{h.label}</span>
                      )}
                      {h.tip && <InfoTip text={h.tip} place="bottom-left" />}
                    </span>
                  ))}
                </div>
                {pageRows.map((p) => (
                  <div className="bt-row" role="row" key={`${p.name}-${p.draft_year}-${p.position}`}>
                    <span className="bt-c-name">
                      <span className="bt-name">{p.name}</span>
                      <span className="bt-sub">{p.position}{p.college ? ` · ${p.college}` : ''}</span>
                    </span>
                    <span className="bt-c-year">{p.draft_year}</span>
                    <span className="bt-c-prob">{fmtPct(p.pred_success_prob)}</span>
                    <span className="bt-c-bucket">{p.pred_grade_bucket}</span>
                    <span className="bt-c-range">
                      {p.pred_pick >= 299.5 ? 'UDFA' : `~#${Math.round(p.pred_pick)}`}
                      <span className="bt-sub">
                        {Math.round(p.pick_lo)}–{p.pick_hi >= 299.5 ? '300+' : Math.round(p.pick_hi)}
                      </span>
                    </span>
                    <span className="bt-c-pick">{p.consensus_rank ? `#${p.consensus_rank}` : '—'}</span>
                    <span className="bt-c-pick">
                      {p.actual_pick ? `#${p.actual_pick}` : 'UDFA'}
                      {p.actual_round ? <span className="bt-sub"> Rd {p.actual_round}</span> : null}
                    </span>
                    <span className={`bt-c-outcome ${p.actual_success ? 'bt-hit' : 'bt-bust'}`}>
                      {p.actual_success ? 'Hit' : 'Bust'}
                    </span>
                    <span className="bt-c-av">{p.career_av == null ? '—' : p.career_av}</span>
                  </div>
                ))}
              </div>
              <div className="bt-pager">
                <span className="bt-pager-status">
                  Showing {ledger.length ? page * PAGE_SIZE + 1 : 0}&ndash;{Math.min((page + 1) * PAGE_SIZE, ledger.length)} of {ledger.length}
                </span>
                <div className="bt-pager-btns">
                  <button type="button" className="bt-btn bt-btn-quiet" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
                  <span className="bt-pager-status">Page {page + 1} / {pageCount}</span>
                  <button type="button" className="bt-btn bt-btn-quiet" disabled={page >= pageCount - 1} onClick={() => setPage(page + 1)}>Next</button>
                </div>
              </div>
            </>
          )}
        </section>

        <p className="bt-foot">
          Generated {new Date(data.generated_at).toLocaleDateString()}. {data.holdout_note}{' '}
          &ldquo;Hit&rdquo; = a Pro Bowl, 3+ seasons as a primary starter, or equivalent
          career value. Regenerate with scripts/generate_backtest.py; every number on this page
          can be recomputed from the CSV plus training_data/combine_outcomes.csv.
        </p>
      </main>
    </div>
  );
}

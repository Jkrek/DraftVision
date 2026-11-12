import React, { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { anonFetch } from '../../lib/api';
import { nflLogoUrl } from '../nflTeams';
import { playerSlug } from '../../lib/edgeData';
import './TeamNeeds.css';

/*
 * Team Needs — one NFL depth chart at a time, drawn as a formation on a
 * field, every player coloured by the year his contract ends. The header
 * carries the team's projected first-round slot from the season simulation.
 *
 * Colour is never the only signal: every name carries its end year ('26).
 */

const STATUS_LABEL = {
  expiring: (season) => `Ends after ${season}`,
  next: (season) => `Ends after ${season + 1}`,
  signed: (season) => `Signed through ${season + 2} or later`,
  unknown: () => 'No contract on file',
};
const STATUS_ORDER = ['expiring', 'next', 'signed', 'unknown'];

// Rows are listed the way they sit on the field. Defence faces the offence,
// so its rows read right-to-left (RCB lines up across from the left WR).
// `edge` pushes a spot out to the sideline. The wide offence is a grid of
// [position, column, row] so the slot receiver can stand off the line, in the
// gap between the tight end and the outside receiver.
const LAYOUT = {
  wide: {
    defense: {
      '4-3': [
        ['FS', 'SS'],
        ['SLB', 'MLB', 'WLB', 'NB'],
        [{ k: 'RCB', edge: 'left' }, 'RDE', 'RDT', 'LDT', 'LDE', { k: 'LCB', edge: 'right' }],
      ],
      '3-4': [
        ['FS', 'SS'],
        ['SLB', 'RILB', 'LILB', 'WLB', 'NB'],
        [{ k: 'RCB', edge: 'left' }, 'RDE', 'NT', 'LDE', { k: 'LCB', edge: 'right' }],
      ],
    },
    offense: {
      cols: 9,
      cells: [
        ['WR-L', 1, 1], ['LT', 2, 1], ['LG', 3, 1], ['C', 4, 1], ['RG', 5, 1], ['RT', 6, 1], ['TE', 7, 1], ['WR-R', 9, 1],
        ['QB', 4, 2], ['WR-S', 8, 2],
        ['FB', 3, 3], ['RB', 4, 3],
      ],
    },
  },
  narrow: {
    defense: {
      '4-3': [
        ['RCB', 'FS', 'SS', 'LCB'],
        ['SLB', 'MLB', 'WLB', 'NB'],
        ['RDE', 'RDT', 'LDT', 'LDE'],
      ],
      '3-4': [
        ['RCB', 'FS', 'SS', 'LCB'],
        ['SLB', 'RILB', 'LILB', 'WLB', 'NB'],
        ['RDE', 'NT', 'LDE'],
      ],
    },
    offense: [
      ['LT', 'LG', 'C', 'RG', 'RT'],
      ['WR-L', 'TE', 'WR-S', 'WR-R'],
      ['QB'],
      ['FB', 'RB'],
    ],
  },
};
const SPECIALISTS = ['PK', 'P', 'LS'];
const POS_LABEL = { 'WR-L': 'WR', 'WR-R': 'WR', 'WR-S': 'SLOT', PK: 'K' };

const SUFFIX = /^(jr\.?|sr\.?|ii|iii|iv|v)$/i;
function lastName(name) {
  const parts = (name || '').trim().split(/\s+/);
  if (parts.length < 2) return name || '';
  const tail = SUFFIX.test(parts[parts.length - 1]) && parts.length > 2 ? parts.splice(-2) : parts.splice(-1);
  return tail.join(' ');
}

const yr = (ends) => (ends ? `’${String(ends).slice(-2)}` : '—');
const money = (m) => (m == null ? '—' : m >= 1 ? `$${m.toFixed(1)}M` : `$${Math.round(m * 1000)}K`);
const pct = (p) => (p >= 0.995 ? '99%+' : p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`);

function useNarrow(query = '(max-width: 760px)') {
  const get = () => typeof window !== 'undefined' && window.matchMedia && window.matchMedia(query).matches;
  const [narrow, setNarrow] = useState(get);
  useEffect(() => {
    if (!window.matchMedia) return undefined;
    const mq = window.matchMedia(query);
    const on = () => setNarrow(mq.matches);
    on();
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

// Index the team's spots by the key the layout uses. The three receivers
// share one position code and differ only by formation slot: lowest slot is
// the left receiver, next the right, the last one the slot receiver.
function indexSpots(spots) {
  const by = {};
  const wrs = [];
  (spots || []).forEach((s) => {
    if (s.unit === 'offense' && s.pos === 'WR') wrs.push(s);
    else by[s.pos] = s;
  });
  wrs.sort((a, b) => a.slot - b.slot);
  ['WR-L', 'WR-R', 'WR-S'].forEach((k, i) => { if (wrs[i]) by[k] = wrs[i]; });
  return by;
}

function Spot({ k, spot, edge, style, narrow, season, selected, onSelect }) {
  if (!spot) return null;
  const [starter, ...rest] = spot.players;
  const backups = rest.slice(0, narrow ? 1 : 2);
  const label = POS_LABEL[k] || k;
  const describe = (p) => `${p.name}, ${label}. ${STATUS_LABEL[p.contract.status](season)}.`;
  const isSel = (p) => selected && selected.name === p.name && selected.pos === label;
  return (
    <div className={`tn-spot${edge ? ` tn-edge-${edge}` : ''}`} style={style}>
      <span className="tn-pos">{label}</span>
      <button
        type="button"
        className={`tn-starter tn-s-${starter.contract.status}${isSel(starter) ? ' tn-selected' : ''}`}
        onClick={() => onSelect({ ...starter, pos: label, depth: 1 })}
        aria-label={describe(starter)}
        aria-pressed={Boolean(isSel(starter))}
      >
        <span className="tn-marker" aria-hidden="true" />
        <span className="tn-name">{narrow ? lastName(starter.name) : starter.name}</span>
        <span className="tn-yr">{yr(starter.contract.ends)}</span>
      </button>
      {backups.map((p, i) => (
        <button
          type="button"
          key={p.name}
          className={`tn-backup tn-s-${p.contract.status}${isSel(p) ? ' tn-selected' : ''}`}
          onClick={() => onSelect({ ...p, pos: label, depth: i + 2 })}
          aria-label={describe(p)}
          aria-pressed={Boolean(isSel(p))}
        >
          <span className="tn-tick" aria-hidden="true" />
          <span className="tn-name">{lastName(p.name)}</span>
          <span className="tn-yr">{yr(p.contract.ends)}</span>
        </button>
      ))}
    </div>
  );
}

function Rows({ rows, by, ...rest }) {
  return rows.map((row, i) => (
    <div className="tn-row" key={i}>
      {row.map((cell) => {
        const k = typeof cell === 'string' ? cell : cell.k;
        return <Spot key={k} k={k} spot={by[k]} edge={cell.edge} {...rest} />;
      })}
    </div>
  ));
}

function Grid({ grid, by, ...rest }) {
  return (
    <div className="tn-grid" style={{ gridTemplateColumns: `repeat(${grid.cols}, minmax(0, 1fr))` }}>
      {grid.cells.map(([k, col, row]) => (
        <Spot key={k} k={k} spot={by[k]} style={{ gridColumn: col, gridRow: row }} {...rest} />
      ))}
    </div>
  );
}

const nick = (name) => (name || '').split(' ').pop();
const list = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);

// The first-round picks a team holds once trades are applied. Data built
// before trades were tracked has no `picks`: fall back to the team's own slot.
function picksOf(d) {
  if (!d) return [];
  return d.picks || [{ slot_p10: d.slot_p10, slot_p50: d.slot_p50, slot_p90: d.slot_p90, p_top5: d.p_top5, p_top10: d.p_top10, own: true }];
}

function PickStat({ d, names }) {
  const picks = picksOf(d);
  const name = (abbr) => nick(names[abbr]) || abbr;
  const acquired = (d.acquired_from || []).map(name);
  const sent = (d.sent_to || []).map((t) => (
    t.mode === 'own' ? name(t.team) : `${name(t.team)} (the ${t.mode} of its two picks)`
  ));
  return (
    <div className="tn-stat">
      <div className="tn-stat-label">Projected first-round pick{picks.length > 1 ? 's' : ''}</div>
      <div className="tn-stat-value">{picks.length ? `No. ${picks.map((p) => p.slot_p50).join(' · ')}` : 'None'}</div>
      <div className="tn-stat-sub">
        {picks.length === 1 && (
          <>
            likely between {picks[0].slot_p10} and {picks[0].slot_p90}
            {picks[0].p_top10 >= 0.05 && ` · top 10: ${pct(picks[0].p_top10)}`}
            {picks[0].p_top5 >= 0.05 && ` · top 5: ${pct(picks[0].p_top5)}`}
          </>
        )}
        {picks.length > 1 && `likely ${picks.map((p) => `${p.slot_p10}–${p.slot_p90}`).join(', ')}`}
        {!picks.length && `Its own pick projects at No. ${d.slot_p50}`}
        {acquired.length > 0 && <div>Acquired from the {list(acquired)}</div>}
        {sent.length > 0 && <div>Traded to the {list(sent)}</div>}
      </div>
    </div>
  );
}

function reason(r, season) {
  const n = r.expiring || r.next;
  const year = r.expiring ? season : season + 1;
  return `${n} of ${r.starters} ${r.group_label.toLowerCase()} starter${r.starters > 1 ? 's' : ''} signed only through ${year}`;
}

function Fits({ fits, season, draftYear }) {
  if (!fits || !fits.length) return null;
  return (
    <section className="tn-fits" aria-label="Prospects in range">
      <div className="tn-unit-label">Prospects in range</div>
      <p className="tn-fits-lede">
        {draftYear} prospects the model board projects inside each pick’s likely range, at positions
        where a starter’s contract is running out. An overlap of two lists, not a predicted selection.
      </p>
      <div className="tn-fits-cols">
        {fits.map((f, i) => (
          <div className="tn-fit" key={i}>
            <div className="tn-fit-head">
              Pick No. {f.slot_p50}
              <span>range {f.slot_p10}–{f.slot_p90}</span>
            </div>
            {f.prospects.length ? (
              <ol className="tn-fit-list">
                {f.prospects.map((r) => (
                  <li key={`${r.name}-${r.college}`}>
                    <span className="tn-fit-rank" title="Projected pick on the model board">{r.projected_pick}</span>
                    <span className="tn-fit-body">
                      <Link to={`/player/${playerSlug(r.name, r.college)}`}>{r.name}</Link>
                      <span className="tn-fit-meta">{r.position} · {r.college}</span>
                      <span className="tn-fit-why">{reason(r, season)}</span>
                    </span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="tn-fit-none">No prospect at a position of need projects in this range.</p>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function PlayerCard({ player, season }) {
  if (!player) {
    return <div className="tn-card tn-card-empty">Select a player to see his contract.</div>;
  }
  const c = player.contract;
  const known = c.status !== 'unknown';
  return (
    <div className={`tn-card tn-s-${c.status}`} aria-live="polite">
      <div className="tn-card-head">
        <span className="tn-card-name">{player.name}</span>
        <span className="tn-card-pos">{player.pos} · {player.depth === 1 ? 'starter' : `No. ${player.depth} on the chart`}</span>
        <span className="tn-chip"><span className="tn-tick" aria-hidden="true" />{STATUS_LABEL[c.status](season)}</span>
      </div>
      {known ? (
        <dl className="tn-card-facts">
          <div><dt>Last season under contract</dt><dd>{c.ends}</dd></div>
          <div><dt>Average per year</dt><dd>{money(c.apy)}</dd></div>
          <div><dt>Guaranteed</dt><dd>{money(c.guaranteed)}</dd></div>
          <div><dt>{season + 1} cap charge</dt><dd>{c.cap_next ? money(c.cap_next) : '—'}</dd></div>
          <div><dt>Signed</dt><dd>{c.signed || '—'}{c.rookie_deal ? ' · rookie deal' : c.type ? ` · ${c.type}` : ''}</dd></div>
        </dl>
      ) : (
        <p className="tn-card-note">No active contract was found for this player in the OverTheCap data.</p>
      )}
      {c.otc && (
        <a className="tn-card-link" href={c.otc} target="_blank" rel="noreferrer">Full contract on OverTheCap ↗</a>
      )}
    </div>
  );
}

export default function TeamNeeds() {
  const { team: teamParam } = useParams();
  const narrow = useNarrow();
  const [index, setIndex] = useState(null);
  const [detail, setDetail] = useState(null);
  const [failed, setFailed] = useState(false);
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    let alive = true;
    anonFetch('/api/team-needs')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d) => { if (alive) setIndex(d); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, []);

  // No team in the URL: open on the team projected to pick first.
  const abbr = (teamParam || (index && index.teams[0] && index.teams[0].team) || '').toUpperCase();

  useEffect(() => {
    if (!abbr) return undefined;
    let alive = true;
    setSelected(null);
    anonFetch(`/api/team-needs/${abbr}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d) => { if (alive) setDetail(d); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, [abbr]);

  const team = detail && detail.team && detail.team.team === abbr ? detail.team : null;
  const season = (detail && detail.season) || (index && index.season) || new Date().getFullYear();
  const by = useMemo(() => indexSpots(team && team.spots), [team]);
  const layout = narrow ? LAYOUT.narrow : LAYOUT.wide;
  const openings = team ? team.groups.filter((g) => g.expiring > 0) : [];
  const d = team && team.draft;
  const names = useMemo(
    () => Object.fromEntries(((index && index.teams) || []).map((t) => [t.team, t.name])),
    [index],
  );
  const firstPick = (t) => { const p = picksOf(t.draft)[0]; return p ? p.slot_p50 : null; };
  const spotProps = { narrow, season, selected, onSelect: setSelected };

  return (
    <div className="tn-page">
      <header className="tn-hero">
        <div className="tn-hero-inner">
          <div className="tn-eyebrow">Team needs · {season} season</div>
          <h1 className="tn-title">Who is under contract, and where each team will pick</h1>
          <p className="tn-lede">
            Every depth chart, coloured by the year each player’s contract ends, next to the
            draft slot the team’s record is heading for.
          </p>
        </div>
      </header>

      <main className="tn-main">
        {failed && !index && (
          <p className="tn-error">Team data could not be loaded. Try again in a minute.</p>
        )}

        {index && (
          <nav className="tn-picker" aria-label={`Teams, by first pick in the ${index.draft_year} draft`}>
            <div className="tn-picker-label">Teams by projected first pick, {index.draft_year} draft</div>
            <div className="tn-picker-strip">
              {index.teams.map((t) => (
                <Link
                  key={t.team}
                  to={`/team-needs/${t.team}`}
                  className={t.team === abbr ? 'tn-pick tn-pick-active' : 'tn-pick'}
                  aria-current={t.team === abbr ? 'page' : undefined}
                  aria-label={firstPick(t) ? `${t.name}, projected pick ${firstPick(t)}` : `${t.name}, no first-round pick`}
                >
                  <span className="tn-pick-no">{firstPick(t) || '—'}</span>
                  <img src={nflLogoUrl(t.name)} alt="" loading="lazy" width="30" height="30" />
                  <span className="tn-pick-abbr">{t.team}</span>
                </Link>
              ))}
            </div>
          </nav>
        )}

        {!team && !failed && <div className="tn-loading" aria-busy="true">Loading depth chart…</div>}

        {team && (
          <>
            <section className="tn-summary" aria-label={`${team.name} summary`}>
              <div className="tn-team">
                <img src={nflLogoUrl(team.name)} alt="" width="56" height="56" />
                <div>
                  <h2 className="tn-team-name">{team.name}</h2>
                  {d && <div className="tn-team-sub">{d.record} · projected {d.projected_wins} wins</div>}
                </div>
              </div>
              {d && <PickStat d={d} names={names} />}
              <div className="tn-stat">
                <div className="tn-stat-label">Starters with contracts ending after {season}</div>
                <div className="tn-stat-value">{team.starters_expiring} of {team.starters}</div>
                <div className="tn-stat-sub">
                  {openings.length
                    ? openings.slice(0, 4).map((g) => `${g.label} ${g.expiring} of ${g.starters}`).join(' · ')
                    : 'Every starter is signed past this season'}
                </div>
              </div>
            </section>

            <ul className="tn-legend" aria-label="Colour key">
              {STATUS_ORDER.map((s) => (
                <li key={s} className={`tn-s-${s}`}>
                  <span className="tn-marker" aria-hidden="true" />{STATUS_LABEL[s](season)}
                </li>
              ))}
            </ul>

            <section className="tn-field" aria-label={`${team.name} depth chart`}>
              <div className="tn-unit tn-unit-def">
                <div className="tn-unit-label">Defense · base {team.front}</div>
                <Rows rows={layout.defense[team.front] || layout.defense['4-3']} by={by} {...spotProps} />
              </div>
              <div className="tn-los" aria-hidden="true"><span>Line of scrimmage</span></div>
              <div className="tn-unit tn-unit-off">
                {Array.isArray(layout.offense)
                  ? <Rows rows={layout.offense} by={by} {...spotProps} />
                  : <Grid grid={layout.offense} by={by} {...spotProps} />}
                <div className="tn-unit-label">Offense</div>
              </div>
            </section>

            <div className="tn-below">
              <section className="tn-specialists" aria-label="Specialists">
                <div className="tn-unit-label">Specialists</div>
                <div className="tn-row">
                  {SPECIALISTS.map((k) => <Spot key={k} k={k} spot={by[k]} {...spotProps} />)}
                </div>
              </section>
              <PlayerCard player={selected} season={season} />
            </div>

            <Fits fits={detail.fits} season={season} draftYear={detail.draft_year} />

            <p className="tn-foot">
              “Ends after {season}” means {season} is the last season the current contract pays the player.
              It does not mean he will leave: team options, franchise tags and re-signings are not modelled.
              Draft slots come from 20,000 simulated seasons
              {detail.draft_through_week ? ` (through week ${detail.draft_through_week})` : ''}.
              {detail.trades_source ? (
                <>
                  {' '}First-round trades are applied as listed on{' '}
                  <a href={detail.trades_source} target="_blank" rel="noreferrer">Wikipedia</a>
                  {detail.trades_as_of ? ` on ${detail.trades_as_of}` : ''}; pick protections are not.
                </>
              ) : ' Traded picks are not applied.'}{' '}
              Depth chart updated {team.chart_updated}. Depth charts from ESPN and contracts from{' '}
              <a href="https://overthecap.com" target="_blank" rel="noreferrer">OverTheCap</a>, both via{' '}
              <a href="https://github.com/nflverse/nflverse-data" target="_blank" rel="noreferrer">nflverse</a> (CC-BY 4.0).
            </p>
          </>
        )}
      </main>
    </div>
  );
}

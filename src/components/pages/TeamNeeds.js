import React, { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { anonFetch } from '../../lib/api';
import { nflLogoUrl } from '../nflTeams';
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
// `edge` pushes a spot out to the sideline.
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
    offense: [
      [{ k: 'WR-L', edge: 'left' }, 'LT', 'LG', 'C', 'RG', 'RT', 'TE', { k: 'WR-R', edge: 'right' }],
      ['WR-S', 'QB'],
      ['FB', 'RB'],
    ],
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
      ['WR-L', 'WR-S', 'TE', 'WR-R'],
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

function Spot({ k, spot, edge, narrow, season, selected, onSelect }) {
  if (!spot) return null;
  const [starter, ...rest] = spot.players;
  const backups = rest.slice(0, narrow ? 1 : 2);
  const label = POS_LABEL[k] || k;
  const describe = (p) => `${p.name}, ${label}. ${STATUS_LABEL[p.contract.status](season)}.`;
  const isSel = (p) => selected && selected.name === p.name && selected.pos === label;
  return (
    <div className={`tn-spot${edge ? ` tn-edge-${edge}` : ''}`}>
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
          <nav className="tn-picker" aria-label={`Teams, in projected ${index.draft_year} draft order`}>
            <div className="tn-picker-label">Projected {index.draft_year} draft order</div>
            <div className="tn-picker-strip">
              {index.teams.map((t, i) => (
                <Link
                  key={t.team}
                  to={`/team-needs/${t.team}`}
                  className={t.team === abbr ? 'tn-pick tn-pick-active' : 'tn-pick'}
                  aria-current={t.team === abbr ? 'page' : undefined}
                  aria-label={`${t.name}, projected pick ${t.draft ? t.draft.slot_p50 : i + 1}`}
                >
                  <span className="tn-pick-no">{t.draft ? t.draft.slot_p50 : i + 1}</span>
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
              {d && (
                <div className="tn-stat">
                  <div className="tn-stat-label">Projected first-round pick</div>
                  <div className="tn-stat-value">No. {d.slot_p50}</div>
                  <div className="tn-stat-sub">
                    likely between {d.slot_p10} and {d.slot_p90}
                    {d.p_top10 >= 0.05 && ` · top 10: ${pct(d.p_top10)}`}
                    {d.p_top5 >= 0.05 && ` · top 5: ${pct(d.p_top5)}`}
                  </div>
                </div>
              )}
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
                <Rows rows={layout.offense} by={by} {...spotProps} />
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

            <p className="tn-foot">
              “Ends after {season}” means {season} is the last season the current contract pays the player.
              It does not mean he will leave: team options, franchise tags and re-signings are not modelled.
              The draft slot is where the team’s own pick lands across 20,000 simulated seasons
              {detail.draft_through_week ? ` (through week ${detail.draft_through_week})` : ''}; traded picks are not applied.
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

import React, {
  useCallback, useEffect, useRef, useState,
} from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { anonFetch } from '../lib/api';
import InfoTip from './InfoTip';
import './PredictionComponent.css';

// Metric explainers — these describe the ACTUAL calculations (see XGBOost.py)
const TIP = {
  successProb:
    'The ML ensemble’s calibrated estimate that this player becomes an NFL '
    + '“success” — defined in training as making a Pro Bowl, starting 3+ '
    + 'seasons, or reaching 30+ career Approximate Value. Built from 29 features: '
    + 'production, athleticism, recruiting pedigree and competition level.',
  projection:
    'Projected draft slot from two model heads blended: a pick-number '
    + 'regressor and the draft-round classifier, both trained on the '
    + '2000–2023 classes. The pick shown is the player’s rank within his '
    + 'own draft class on the board — where he’d go in that class’s draft.',
  grade:
    'Letter grade from percentile cutoffs of success probability across the '
    + 'full FBS board — A+ is the top 2%, A- the top 10%, C+ sits near the '
    + 'median. Graded on the curve of this class, not an absolute scale.',
  factors:
    'Per-player SHAP attribution — how much each feature pushed THIS '
    + 'prediction up or down, as a share of the total. Longer bar = bigger '
    + 'influence on the number above.',
  comps:
    'Nearest real prospects from the 2000–2023 classes at the same position '
    + 'group — statistical distance over height, weight, speed, 40/vertical '
    + '(when measured), production, recruiting stars and competition level. '
    + 'Capped at 99%: no comp is a clone.',
};

// ── Helpers ────────────────────────────────────────────────────────────────
const img = (file) => `${process.env.PUBLIC_URL}/images/CFB Content/${file}`;

// "Name-Team" → player-page / compare slug (same encoding the boards use).
const slugify = (name, team) =>
  `${name || ''}-${team || ''}`
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9-]/g, '');

// Grade letter → tier class for color-coding (house pattern: gold A,
// accent B, muted C/D — same tiers the Big Board pills use).
const gradeTier = (grade) => {
  const c = (grade || '').charAt(0).toUpperCase();
  return c === 'A' ? 'a' : c === 'B' ? 'b' : c === 'C' ? 'c' : c === 'D' ? 'd' : '';
};

// One-tap starters for the empty state — recognizable names beat a blank box.
const SUGGESTED_PLAYERS = ['Jeremiah Smith', 'Arch Manning', 'Julian Sayin'];

// ── Main component ─────────────────────────────────────────────────────────
export default function PredictionComponent() {
  const apiUrl = (path) => path; // same-origin (Flask serves everything)
  const navigate = useNavigate();

  // autocomplete state
  const [acQuery, setAcQuery]       = useState('');
  const [acResults, setAcResults]   = useState([]);
  const [acOpen, setAcOpen]         = useState(false);
  const acRef                       = useRef(null);
  const acTimer                     = useRef(null);
  const inputRef                    = useRef(null);

  // data
  const [allPlayers, setAllPlayers]         = useState([]);
  const [teams, setTeams]                   = useState([]);
  const [loadingPlayers, setLoadingPlayers] = useState(true);

  // prediction
  const [selected, setSelected]     = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predError, setPredError]   = useState(null);

  // sync
  const [syncing, setSyncing]       = useState(false);
  const [syncMsg, setSyncMsg]       = useState('');

  // ── load all data in one call ────────────────────────────────
  const loadInit = useCallback(async () => {
    setLoadingPlayers(true);
    try {
      const res  = await anonFetch(apiUrl('/init'));
      const data = await res.json();
      setAllPlayers(Array.isArray(data.players) ? data.players : []);
      setTeams(Array.isArray(data.teams) ? data.teams : []);
    } catch { setAllPlayers([]); }
    finally { setLoadingPlayers(false); }
  }, []);

  useEffect(() => { loadInit(); }, [loadInit]);

  // ── autocomplete search (all sources) ───────────────────────
  const handleAcChange = (e) => {
    const val = e.target.value;
    setAcQuery(val);
    clearTimeout(acTimer.current);
    if (val.length < 2) { setAcResults([]); setAcOpen(false); return; }
    acTimer.current = setTimeout(async () => {
      try {
        const res  = await anonFetch(apiUrl(`/search?q=${encodeURIComponent(val)}`));
        const data = await res.json();
        setAcResults(Array.isArray(data.players) ? data.players : []);
        setAcOpen(true);
      } catch { setAcResults([]); }
    }, 180);
  };

  // close autocomplete on outside click
  useEffect(() => {
    const handler = (e) => { if (acRef.current && !acRef.current.contains(e.target)) setAcOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // ── run prediction ────────────────────────────────────────────
  const runPrediction = useCallback(async (player) => {
    setSelected(player);
    setPrediction(null);
    setPredError(null);
    setPredicting(true);
    setAcOpen(false);
    setAcQuery('');
    try {
      const res  = await anonFetch(apiUrl('/predict'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: player.name,
          ...(player.position && player.position !== 'UNK' ? { position: player.position } : {}),
          ...(player.team ? { team: player.team } : {}),
          // espn_id (when the search row carries it) unlocks verified ESPN stats
          ...(player.espn_id ? { espn_id: player.espn_id } : {}),
        }),
        signal: AbortSignal.timeout(14000),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed.');
      setPrediction(data);
    } catch (err) {
      setPredError(err.message || 'Could not reach backend.');
    } finally {
      setPredicting(false);
    }
  }, []);

  // Resolve a bare name through /search before predicting — the matched row
  // carries espn_id/team, which unlocks the verified-stats path and the
  // physically-grounded comps. Bare-name predictions get a baseline profile.
  const resolveAndPredict = useCallback(async (name) => {
    try {
      const res  = await anonFetch(apiUrl(`/search?q=${encodeURIComponent(name)}`));
      const data = await res.json();
      const rows = (Array.isArray(data.players) ? data.players : []).filter((p) => p.kind !== 'hs');
      const match =
        rows.find((p) => p.name.toLowerCase() === name.toLowerCase() && p.espn_id) ||
        rows.find((p) => p.name.toLowerCase() === name.toLowerCase()) ||
        null;
      runPrediction(match || { name, position: 'UNK', team: '' });
    } catch {
      runPrediction({ name, position: 'UNK', team: '' });
    }
  }, [runPrediction]);

  // "Run model" — predict whatever is typed (prefer a non-HS autocomplete match;
  // the college model does not apply to high-school prospects)
  const handleRun = useCallback(() => {
    const name = acQuery.trim();
    if (name.length < 2 || predicting) return;
    const match = acResults.find(
      (p) => p.kind !== 'hs' && p.name.toLowerCase() === name.toLowerCase(),
    );
    if (match) runPrediction(match);
    else resolveAndPredict(name);
  }, [acQuery, acResults, predicting, runPrediction, resolveAndPredict]);

  // ── sync ──────────────────────────────────────────────────────
  const handleSync = useCallback(async () => {
    setSyncing(true); setSyncMsg('');
    try {
      const res  = await anonFetch(apiUrl('/sync/college-prospects'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_teams: 250, max_players: 5000 }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Sync failed.');
      setSyncMsg(`Synced ${data?.synced?.inserted || 0} prospects from ${data?.synced?.teams || 0} teams.`);
      await loadInit();
    } catch (err) { setSyncMsg(err.message || 'Sync failed.'); }
    finally { setSyncing(false); }
  }, [loadInit]);

  // ── auto-predict from ?name= query param (e.g. navigated from Leaderboard) ─
  const location       = useLocation();
  const didAutoPredict = useRef(false);
  useEffect(() => {
    if (didAutoPredict.current) return;
    const name = new URLSearchParams(location.search).get('name');
    if (name && name.length >= 2) {
      didAutoPredict.current = true;
      // Clean the URL so a back-navigation doesn't re-trigger
      window.history.replaceState(null, '', '/predict');
      resolveAndPredict(name);
    }
  }, [location.search, resolveAndPredict]);

  // ── auto-focus the search box — this page's one job is search. Skipped
  // when arriving via ?name= (auto-predict is already running) and on
  // narrow viewports, where focus would pop the keyboard over the page. ──
  useEffect(() => {
    const name = new URLSearchParams(window.location.search).get('name');
    if (!name && window.innerWidth >= 768 && inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // ── render ────────────────────────────────────────────────────
  return (
    <main className="predict-page">
      <div className="predict-shell">
        <div className="eyebrow">Scouting report</div>

        {/* ── Search row ── */}
        <div className="predict-search-row" ref={acRef}>
          <div className="predict-search-box">
            <input
              type="text"
              className="predict-input"
              ref={inputRef}
              value={acQuery}
              onChange={handleAcChange}
              onKeyDown={(e) => { if (e.key === 'Enter') handleRun(); }}
              placeholder="Enter a player name"
              aria-label="Player name"
            />
            {acOpen && acResults.length > 0 && (() => {
              const collegeRows = acResults.filter((p) => p.kind !== 'hs');
              const hsRows      = acResults.filter((p) => p.kind === 'hs');
              return (
                <div className="predict-ac" role="listbox">
                  {collegeRows.map((p) => {
                    const isNFL = ['legacy', 'nfl_seed'].includes(p.source);
                    return (
                      <button
                        type="button"
                        key={`c-${p.name}-${p.team || ''}`}
                        className="predict-ac-row"
                        onMouseDown={() => runPrediction(p)}
                      >
                        <span className="predict-ac-main">
                          <span className="predict-ac-name">{p.name}</span>
                          <span className="predict-ac-sub">
                            {[p.position, p.team].filter(Boolean).join(' · ')}
                            {isNFL ? ' · NFL Pro' : ''}
                          </span>
                        </span>
                        {p.grade && <span className="predict-ac-grade">{p.grade}</span>}
                      </button>
                    );
                  })}
                  {hsRows.length > 0 && (
                    <>
                      <div className="predict-ac-group" aria-hidden="true">High school</div>
                      {hsRows.map((p) => (
                        <button
                          type="button"
                          key={`hs-${p.name}-${p.school || ''}`}
                          className="predict-ac-row predict-ac-row-hs"
                          title="High school prospect — the college model doesn't apply; opens the HS board instead."
                          onMouseDown={() => navigate(`/hs-prospects?q=${encodeURIComponent(p.name)}`)}
                        >
                          <span className="predict-ac-main">
                            <span className="predict-ac-name">{p.name}</span>
                            <span className="predict-ac-sub">
                              {[p.position, p.school].filter(Boolean).join(' · ')}
                            </span>
                          </span>
                          <span className="predict-ac-sub predict-ac-hsmeta">
                            {p.stars ? `${p.stars}★` : ''}
                            {p.stars && p.year ? ' · ' : ''}
                            {p.year ? `'${String(p.year).slice(-2)}` : ''}
                          </span>
                        </button>
                      ))}
                    </>
                  )}
                </div>
              );
            })()}
          </div>
          <button
            type="button"
            className="predict-run-btn"
            onClick={handleRun}
            disabled={predicting || acQuery.trim().length < 2}
          >
            {predicting ? 'Running…' : 'Run model'}
          </button>
        </div>

        {/* ── Empty / initial state ── */}
        {!prediction && !predicting && !predError && (
          <div className="predict-empty">
            <p className="predict-empty-lede">
              Type any FBS player and run the ensemble — engineered production, athleticism
              and competition features scored by the ML models behind the board.
            </p>
            <div className="predict-try" role="group" aria-label="Suggested players">
              <span className="predict-try-label">Try:</span>
              {SUGGESTED_PLAYERS.map((name) => (
                <button
                  key={name}
                  type="button"
                  className="predict-try-chip"
                  onClick={() => resolveAndPredict(name)}
                >
                  {name}
                </button>
              ))}
            </div>
            <p className="predict-empty-meta">
              {loadingPlayers
                ? 'Loading prospect pool…'
                : `${allPlayers.length.toLocaleString()} searchable prospects across ${teams.length} programs — college, 2025 draft class and 2026 freshmen.`}
              <button
                type="button"
                className="predict-sync-btn"
                onClick={handleSync}
                disabled={syncing}
              >
                {syncing ? 'Syncing…' : 'Re-sync prospects'}
              </button>
            </p>
            {syncMsg && <p className="predict-sync-msg">{syncMsg}</p>}
          </div>
        )}

        {/* ── Loading state ── */}
        {predicting && (
          <div className="predict-status" role="status">
            <span className="predict-status-dot" aria-hidden="true" />
            Running the model ensemble for <strong>{selected?.name}</strong>…
          </div>
        )}

        {/* ── Error state ── */}
        {predError && !predicting && (
          <div className="predict-error" role="alert">
            <p className="predict-error-text">{predError}</p>
            <button
              type="button"
              className="predict-clear-btn"
              onClick={() => { setPredError(null); setSelected(null); }}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* ── Report ── */}
        {prediction && !predicting && (() => {
          const name    = prediction?.resolved_name || selected?.name;
          const pos     = prediction?.predicted_position || selected?.position || '—';
          const team    = prediction?.stats?.team || selected?.team || '';
          const espnId  = prediction?.espn_id || null;
          const teamId  = prediction?.espn_team_id || null;
          const prob    = typeof prediction.success_probability === 'number' ? prediction.success_probability : null;
          const factors = Array.isArray(prediction?.top_factors) ? prediction.top_factors : [];
          const comps   = Array.isArray(prediction?.historical_comps) ? prediction.historical_comps : [];

          return (
            <div className="predict-report">
              {/* Hero card */}
              <section className="report-hero">
                <div className="report-hero-media" aria-hidden="true">
                  <img className="report-hero-img" src={img('img-2.jpeg')} alt="" />
                  <div className="report-hero-tint" />
                  <div className="report-hero-scrim" />
                  {teamId && (
                    <img
                      className="report-hero-teammark"
                      src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${teamId}.png`}
                      alt=""
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  )}
                </div>
                <div className="report-hero-content">
                  {espnId && (
                    <img
                      className="report-headshot"
                      src={`https://a.espncdn.com/i/headshots/college-football/players/full/${espnId}.png`}
                      alt=""
                      loading="lazy"
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  )}
                  <div className="report-identity">
                    <h1 className="report-name">{name}</h1>
                    <div className="report-sub">
                      {teamId && (
                        <img
                          className="report-sub-logo"
                          src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${teamId}.png`}
                          alt=""
                          onError={(e) => { e.target.style.display = 'none'; }}
                        />
                      )}
                      <span>{pos}{team ? ` · ${team}` : ''}</span>
                    </div>
                  </div>
                  <div className="report-metrics">
                    <div className="report-metric">
                      <div className="report-metric-label">
                        Success probability
                        <InfoTip text={TIP.successProb} place="bottom" />
                      </div>
                      <div className="report-prob">
                        <span className="report-prob-value">{prob !== null ? prob : '—'}</span>
                        {prob !== null && <span className="report-prob-unit">%</span>}
                      </div>
                    </div>
                    <div className="report-metric">
                      <div className="report-metric-label">
                        Projection
                        <InfoTip text={TIP.projection} place="bottom" />
                      </div>
                      <div className="report-metric-value">{prediction?.draft_grade || '—'}</div>
                      {prediction?.projected_pick != null && prediction.projected_pick <= 262 && (
                        <div className="report-metric-sub">Pick ~{prediction.projected_pick}</div>
                      )}
                    </div>
                    <div className="report-metric">
                      <div className="report-metric-label">
                        Grade
                        <InfoTip text={TIP.grade} place="bottom-left" />
                      </div>
                      <div
                        className={`report-grade${
                          gradeTier(prediction?.prospect_grade)
                            ? ` report-grade-${gradeTier(prediction?.prospect_grade)}`
                            : ''}`}
                      >
                        {prediction?.prospect_grade || '—'}
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              {/* 1px-gapped panels */}
              <section className="report-panels">
                <div className="report-panel report-factors">
                  <div className="report-panel-label">
                    Top prediction factors
                    <InfoTip text={TIP.factors} place="bottom" />
                  </div>
                  {factors.length > 0 ? factors.map((f, i) => (
                    <div className="factor" key={f.feature}>
                      <div className="factor-head">
                        <span className="factor-label">{f.feature}</span>
                        <span className="factor-value">{f.importance}%</span>
                      </div>
                      <div className="factor-track">
                        <div
                          className="factor-bar"
                          style={{ width: `${Math.min(f.importance, 100)}%`, animationDelay: `${i * 0.06}s` }}
                        />
                      </div>
                    </div>
                  )) : (
                    <p className="report-note">No factor breakdown was returned for this player.</p>
                  )}
                </div>

                {comps.length > 0 && (
                  <div className="report-panel report-comps">
                    <div className="report-panel-label">
                      Closest historical comps
                      <InfoTip text={TIP.comps} place="bottom" />
                    </div>
                    {comps.map((c, i) => (
                      <div className="comp-row" key={`${c.name}-${i}`}>
                        <div className="comp-id">
                          <div className="comp-name">{c.name}</div>
                          <div className="comp-note">{[c.position, c.outcome].filter(Boolean).join(' · ')}</div>
                        </div>
                        <span className="comp-sim">{c.similarity}%</span>
                      </div>
                    ))}
                    <p className="report-note">
                      Nearest profiles across the model&rsquo;s engineered feature vector,
                      restricted to players at the same position group.
                    </p>
                    <div className="report-photo-strip">
                      <img src={img('img-0.jpeg')} alt="" />
                      <div className="report-photo-scrim" aria-hidden="true" />
                    </div>
                  </div>
                )}
              </section>

              {/* Next actions — the report should end somewhere, not just stop */}
              <div className="report-next">
                <span className="report-next-label">Next</span>
                {team && (
                  <Link className="report-next-btn" to={`/player/${slugify(name, team)}`}>
                    View full profile
                  </Link>
                )}
                {team && (
                  <Link className="report-next-btn" to={`/compare?a=${slugify(name, team)}`}>
                    Compare him
                  </Link>
                )}
                <Link className="report-next-btn report-next-quiet" to="/leaderboard">
                  Back to the board
                </Link>
              </div>

              {/* Meta footer */}
              <div className="report-meta">
                <span>{prediction?.model_used ? 'XGBoost · two-model ML ensemble' : 'Rule-based fallback'}</span>
                {prediction?.data_source && <span>Source: {prediction.data_source}</span>}
                <button
                  type="button"
                  className="predict-clear-btn"
                  onClick={() => { setPrediction(null); setSelected(null); }}
                >
                  Clear report
                </button>
              </div>
            </div>
          );
        })()}
      </div>
    </main>
  );
}

import React, {
  useCallback, useEffect, useRef, useState,
} from 'react';
import { useLocation } from 'react-router-dom';
import { anonFetch } from '../lib/api';
import './PredictionComponent.css';

// ── Helpers ────────────────────────────────────────────────────────────────
const img = (file) => `${process.env.PUBLIC_URL}/images/CFB Content/${file}`;

// ── Main component ─────────────────────────────────────────────────────────
export default function PredictionComponent() {
  const apiUrl = (path) => path; // same-origin (Flask serves everything)

  // autocomplete state
  const [acQuery, setAcQuery]       = useState('');
  const [acResults, setAcResults]   = useState([]);
  const [acOpen, setAcOpen]         = useState(false);
  const acRef                       = useRef(null);
  const acTimer                     = useRef(null);

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
        body: JSON.stringify({ name: player.name }),
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

  // "Run model" — predict whatever is typed (prefer an autocomplete match)
  const handleRun = useCallback(() => {
    const name = acQuery.trim();
    if (name.length < 2 || predicting) return;
    const match = acResults.find((p) => p.name.toLowerCase() === name.toLowerCase());
    runPrediction(match || { name, position: 'UNK', team: '' });
  }, [acQuery, acResults, predicting, runPrediction]);

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
      runPrediction({ name, position: 'UNK', team: '' });
    }
  }, [location.search, runPrediction]);

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
              value={acQuery}
              onChange={handleAcChange}
              onKeyDown={(e) => { if (e.key === 'Enter') handleRun(); }}
              placeholder="Enter a player name"
              aria-label="Player name"
            />
            {acOpen && acResults.length > 0 && (
              <div className="predict-ac" role="listbox">
                {acResults.map((p) => {
                  const isNFL = ['legacy', 'nfl_seed'].includes(p.source);
                  return (
                    <button
                      type="button"
                      key={p.name}
                      className="predict-ac-row"
                      onMouseDown={() => runPrediction(p)}
                    >
                      <span className="predict-ac-name">{p.name}</span>
                      <span className="predict-ac-sub">
                        {[p.position, p.team].filter(Boolean).join(' · ')}
                        {isNFL ? ' · NFL Pro' : ''}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
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
                </div>
                <div className="report-hero-content">
                  <div className="report-identity">
                    <h1 className="report-name">{name}</h1>
                    <div className="report-sub">
                      {pos}{team ? ` · ${team}` : ''}
                    </div>
                  </div>
                  <div className="report-metrics">
                    <div className="report-metric">
                      <div className="report-metric-label">Success probability</div>
                      <div className="report-prob">
                        <span className="report-prob-value">{prob !== null ? prob : '—'}</span>
                        {prob !== null && <span className="report-prob-unit">%</span>}
                      </div>
                    </div>
                    <div className="report-metric">
                      <div className="report-metric-label">Projection</div>
                      <div className="report-metric-value">{prediction?.draft_grade || '—'}</div>
                    </div>
                    <div className="report-metric">
                      <div className="report-metric-label">Grade</div>
                      <div className="report-metric-value report-metric-accent">{prediction?.prospect_grade || '—'}</div>
                    </div>
                  </div>
                </div>
              </section>

              {/* 1px-gapped panels */}
              <section className="report-panels">
                <div className="report-panel report-factors">
                  <div className="report-panel-label">Top prediction factors</div>
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
                    <div className="report-panel-label">Closest historical comps</div>
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

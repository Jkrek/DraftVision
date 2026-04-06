import React, {
  useCallback, useEffect, useMemo, useRef, useState, memo,
} from 'react';
import '../App.css';

// ── Constants ──────────────────────────────────────────────────────────────
const PAGE_SIZE = 60;
const SKILL_POSITIONS = ['QB', 'RB', 'WR', 'TE'];
const POSITION_COLORS = {
  QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b',
  TE: '#a78bfa', default: '#94a3b8',
};

// ── Helpers ────────────────────────────────────────────────────────────────
function posColor(pos) { return POSITION_COLORS[pos] || POSITION_COLORS.default; }

// ── Shared styles (defined once, not recreated per render) ─────────────────
const glassCard = {
  background: 'rgba(30,41,59,0.85)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: '12px',
};
const inputStyle = {
  padding: '10px 14px', borderRadius: '8px',
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(15,23,42,0.7)', color: '#e2e8f0',
  fontSize: '14px', outline: 'none', width: '100%', boxSizing: 'border-box',
};

// ── Source badge label ─────────────────────────────────────────────────────
function sourceBadge(source) {
  if (source === 'nfl_draft_2025') return { label: '2025 Draft', color: '#f59e0b' };
  if (source === 'freshman_2026')  return { label: 'Fr. 2026',   color: '#a78bfa' };
  return null;
}

// ── Memoized prospect card ─────────────────────────────────────────────────
const ProspectCard = memo(function ProspectCard({ player, isActive, isPredicting, onSelect }) {
  const pc = posColor(player.position);
  const badge = sourceBadge(player.source);
  return (
    <div
      onClick={onSelect}
      style={{
        padding: '0.75rem 0.9rem', cursor: 'pointer', display: 'flex',
        alignItems: 'center', gap: '0.65rem', borderRadius: '8px',
        borderLeft: `3px solid ${isActive ? pc : 'transparent'}`,
        background: isActive ? `${pc}15` : 'rgba(255,255,255,0.03)',
        transition: 'background 0.12s, border-color 0.12s',
      }}
    >
      <div style={{
        minWidth: '34px', height: '34px', borderRadius: '50%',
        background: `${pc}22`, border: `2px solid ${pc}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '10px', fontWeight: 700, color: pc, flexShrink: 0,
      }}>
        {player.position || '?'}
      </div>
      <div style={{ overflow: 'hidden', flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <p style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.85rem', margin: 0, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {player.name}
            {isPredicting && isActive && <span style={{ color: '#3b82f6', marginLeft: 6 }}>…</span>}
          </p>
          {badge && (
            <span style={{
              fontSize: '9px', fontWeight: 700, color: badge.color,
              border: `1px solid ${badge.color}`, borderRadius: '4px',
              padding: '1px 4px', whiteSpace: 'nowrap', flexShrink: 0,
            }}>{badge.label}</span>
          )}
        </div>
        <p style={{ color: '#64748b', fontSize: '0.72rem', margin: '2px 0 0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {player.team || 'Unknown'}
        </p>
      </div>
    </div>
  );
});

// ── Main component ─────────────────────────────────────────────────────────
export default function PredictionComponent() {
  const apiUrl = (path) => path; // same-origin (Flask serves everything)

  // filter state
  const [posFilter, setPosFilter]   = useState('ALL');
  const [teamFilter, setTeamFilter] = useState('ALL');
  const [nameSearch, setNameSearch] = useState('');
  const [nameDebounced, setNameDebounced] = useState('');
  const [page, setPage]             = useState(1);

  // autocomplete state
  const [acQuery, setAcQuery]       = useState('');
  const [acResults, setAcResults]   = useState([]);
  const [acOpen, setAcOpen]         = useState(false);
  const acRef                       = useRef(null);
  const acTimer                     = useRef(null);

  // data
  const [allPlayers, setAllPlayers]   = useState([]);
  const [teams, setTeams]             = useState([]);
  const [loadingPlayers, setLoadingPlayers] = useState(true);

  // prediction
  const [selected, setSelected]     = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predError, setPredError]   = useState(null);

  // sync
  const [syncing, setSyncing]       = useState(false);
  const [syncMsg, setSyncMsg]       = useState('');

  const debounceRef = useRef(null);

  // ── load all data in one call ────────────────────────────────
  const loadInit = useCallback(async () => {
    setLoadingPlayers(true);
    try {
      const res  = await fetch(apiUrl('/init'));
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
        const res  = await fetch(apiUrl(`/search?q=${encodeURIComponent(val)}`));
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

  // ── debounced grid name filter ───────────────────────────────
  const handleNameChange = useCallback((e) => {
    const val = e.target.value;
    setNameSearch(val);
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => { setNameDebounced(val); setPage(1); }, 250);
  }, []);

  const handlePosFilter  = useCallback((pos) => { setPosFilter(pos); setPage(1); }, []);
  const handleTeamFilter = useCallback((e)   => { setTeamFilter(e.target.value); setPage(1); }, []);

  // ── run prediction ────────────────────────────────────────────
  const runPrediction = useCallback(async (player) => {
    setSelected(player);
    setPrediction(null);
    setPredError(null);
    setPredicting(true);
    setAcOpen(false);
    setAcQuery('');
    try {
      const res  = await fetch(apiUrl('/predict'), {
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

  // ── sync ──────────────────────────────────────────────────────
  const handleSync = useCallback(async () => {
    setSyncing(true); setSyncMsg('');
    try {
      const res  = await fetch(apiUrl('/sync/college-prospects'), {
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

  // ── filtered + paginated grid ─────────────────────────────────
  const filtered = useMemo(() => allPlayers.filter((p) => {
    const pos = (p.position || '').toUpperCase();
    if (posFilter !== 'ALL' && pos !== posFilter) return false;
    if (teamFilter !== 'ALL' && p.team !== teamFilter) return false;
    if (nameDebounced && !p.name.toLowerCase().includes(nameDebounced.toLowerCase())) return false;
    return true;
  }), [allPlayers, posFilter, teamFilter, nameDebounced]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated  = useMemo(() => filtered.slice(0, page * PAGE_SIZE), [filtered, page]);

  const isSuccess = prediction?.success === 'Success';

  // ── tab style helper ──────────────────────────────────────────
  const tabStyle = (active, pos = posFilter) => ({
    padding: '5px 12px', borderRadius: '999px', cursor: 'pointer',
    border: `1px solid ${active ? posColor(pos) : 'rgba(255,255,255,0.1)'}`,
    background: active ? `${posColor(pos)}22` : 'transparent',
    color: active ? posColor(pos) : '#64748b',
    fontWeight: active ? 700 : 400, fontSize: '12px', transition: 'all 0.12s',
    whiteSpace: 'nowrap',
  });

  // ── render ────────────────────────────────────────────────────
  return (
    <div style={{
      backgroundImage: "linear-gradient(180deg,rgba(15,23,42,0.85) 0%,rgba(15,23,42,0.98) 100%),url('/images/Top-NFL-Players.jpeg')",
      backgroundSize: 'cover', backgroundPosition: 'center', backgroundAttachment: 'fixed',
      minHeight: '100vh', paddingTop: '80px', paddingBottom: '60px',
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 16px' }}>

        {/* ── Page title ── */}
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <h1 style={{ color: '#f1f5f9', fontSize: '2rem', fontWeight: 800, margin: 0 }}>Prospect Predictor</h1>
          <p style={{ color: '#64748b', marginTop: '0.4rem', fontSize: '0.95rem' }}>
            {allPlayers.length > 0
              ? `${allPlayers.length.toLocaleString()} prospects — college, 2025 draft class & 2026 freshmen`
              : 'Loading…'}
          </p>
        </div>

        {/* ── Global search autocomplete ── */}
        <div ref={acRef} style={{ maxWidth: '560px', margin: '0 auto 1.5rem', position: 'relative' }}>
          <input
            type="text"
            value={acQuery}
            onChange={handleAcChange}
            placeholder="🔍  Search — Arch Manning, Cam Ward, Bryce Underwood…"
            style={{
              ...inputStyle,
              padding: '13px 18px', fontSize: '15px', borderRadius: '10px',
              border: '1px solid rgba(59,130,246,0.35)',
              boxShadow: '0 0 0 3px rgba(59,130,246,0.08)',
            }}
          />
          {acOpen && acResults.length > 0 && (
            <div style={{
              position: 'absolute', top: 'calc(100% + 6px)', left: 0, right: 0,
              ...glassCard, zIndex: 100, overflow: 'hidden',
              boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
            }}>
              {acResults.map((p) => {
                const pc = posColor(p.position);
                const isNFL = ['legacy', 'nfl_seed'].includes(p.source);
                return (
                  <div
                    key={p.name}
                    onMouseDown={() => runPrediction(p)}
                    style={{
                      padding: '10px 14px', cursor: 'pointer', display: 'flex',
                      alignItems: 'center', gap: '10px',
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,0.1)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    <div style={{
                      minWidth: '32px', height: '32px', borderRadius: '50%',
                      background: `${pc}22`, border: `2px solid ${pc}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '9px', fontWeight: 700, color: pc,
                    }}>{p.position || '?'}</div>
                    <div>
                      <p style={{ color: '#f1f5f9', fontWeight: 600, fontSize: '14px', margin: 0 }}>{p.name}</p>
                      <p style={{ color: '#64748b', fontSize: '12px', margin: 0 }}>
                        {p.team}{isNFL ? ' · NFL Pro' : ''}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ── Two-panel layout ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: '1.25rem', alignItems: 'start' }}>

          {/* ── LEFT: Filter + Prospect List ── */}
          <div style={{ ...glassCard, overflow: 'hidden' }}>
            {/* Controls header */}
            <div style={{ padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ display: 'flex', gap: '0.6rem', marginBottom: '0.75rem' }}>
                <input
                  type="text"
                  placeholder="Filter list…"
                  value={nameSearch}
                  onChange={handleNameChange}
                  style={{ ...inputStyle, fontSize: '13px', padding: '8px 12px' }}
                />
                <button
                  onClick={handleSync}
                  disabled={syncing}
                  style={{
                    padding: '8px 12px', borderRadius: '8px', cursor: 'pointer', whiteSpace: 'nowrap',
                    background: 'transparent', border: '1px solid rgba(59,130,246,0.4)',
                    color: '#3b82f6', fontSize: '12px', fontWeight: 600,
                  }}
                >{syncing ? '…' : '↻ Sync'}</button>
              </div>

              {/* Team select */}
              <select
                value={teamFilter}
                onChange={handleTeamFilter}
                style={{ ...inputStyle, fontSize: '13px', padding: '8px 12px', marginBottom: '0.6rem', cursor: 'pointer' }}
              >
                <option value="ALL">All Teams</option>
                {teams.map(t => <option key={t} value={t}>{t}</option>)}
              </select>

              {/* Position tabs — skill positions only */}
              <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                {['ALL', ...SKILL_POSITIONS].map(pos => (
                  <button key={pos} onClick={() => handlePosFilter(pos)} style={tabStyle(posFilter === pos, pos)}>
                    {pos}
                  </button>
                ))}
              </div>

              {syncMsg && <p style={{ color: '#22c55e', fontSize: '12px', margin: '0.5rem 0 0' }}>{syncMsg}</p>}
            </div>

            {/* Prospect list */}
            <div style={{ padding: '0.5rem 0.5rem', maxHeight: '600px', overflowY: 'auto' }}>
              <p style={{ color: '#475569', fontSize: '11px', padding: '0 0.5rem 0.5rem', margin: 0 }}>
                {filtered.length} prospect{filtered.length !== 1 ? 's' : ''}
                {posFilter !== 'ALL' ? ` · ${posFilter}` : ''}
              </p>

              {loadingPlayers ? (
                <div style={{ textAlign: 'center', color: '#475569', padding: '2rem', fontSize: '14px' }}>Loading…</div>
              ) : filtered.length === 0 ? (
                <div style={{ textAlign: 'center', color: '#475569', padding: '2rem', fontSize: '13px' }}>No prospects match. Try syncing.</div>
              ) : (
                <>
                  {paginated.map(player => (
                    <ProspectCard
                      key={player.name}
                      player={player}
                      isActive={selected?.name === player.name}
                      isPredicting={predicting}
                      onSelect={() => runPrediction(player)}
                    />
                  ))}
                  {page < totalPages && (
                    <button
                      onClick={() => setPage(p => p + 1)}
                      style={{
                        width: '100%', marginTop: '0.5rem', padding: '8px',
                        background: 'none', border: '1px solid rgba(255,255,255,0.08)',
                        borderRadius: '6px', color: '#64748b', cursor: 'pointer', fontSize: '12px',
                      }}
                    >
                      Load {Math.min(PAGE_SIZE, filtered.length - page * PAGE_SIZE)} more
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* ── RIGHT: Prediction Result ── */}
          <div>
            {!selected && !predicting && (
              <div style={{ ...glassCard, padding: '3rem 2rem', textAlign: 'center', color: '#475569' }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🏈</div>
                <p style={{ fontSize: '1rem', margin: 0 }}>Search any player above or click a prospect from the list</p>
              </div>
            )}

            {predicting && (
              <div style={{ ...glassCard, padding: '3rem 2rem', textAlign: 'center' }}>
                <div style={{ color: '#3b82f6', fontSize: '1.5rem', marginBottom: '0.5rem' }}>⏳</div>
                <p style={{ color: '#94a3b8', margin: 0 }}>
                  Running prediction for <strong style={{ color: '#e2e8f0' }}>{selected?.name}</strong>…
                </p>
              </div>
            )}

            {predError && !predicting && (
              <div style={{ ...glassCard, padding: '1.5rem', color: '#ef4444', textAlign: 'center' }}>{predError}</div>
            )}

            {prediction && !predicting && (
              <div style={{ ...glassCard, padding: '1.75rem' }}>

                {/* Header */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.3rem' }}>
                      <span style={{
                        padding: '3px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: 700,
                        background: `${posColor(prediction?.predicted_position)}22`,
                        border: `1px solid ${posColor(prediction?.predicted_position)}`,
                        color: posColor(prediction?.predicted_position),
                      }}>{prediction?.predicted_position || '?'}</span>
                      <span style={{ color: '#64748b', fontSize: '13px' }}>{prediction?.stats?.team || ''}</span>
                    </div>
                    <h2 style={{ color: '#f1f5f9', margin: 0, fontSize: '1.6rem', fontWeight: 800 }}>
                      {prediction?.resolved_name || selected?.name}
                    </h2>
                  </div>
                  <div style={{
                    padding: '0.65rem 1.25rem', borderRadius: '10px', textAlign: 'center',
                    background: isSuccess ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                    border: `2px solid ${isSuccess ? '#22c55e' : '#ef4444'}`,
                  }}>
                    <div style={{ fontSize: '1.6rem' }}>{isSuccess ? '✅' : '❌'}</div>
                    <div style={{ fontWeight: 800, color: isSuccess ? '#22c55e' : '#ef4444', fontSize: '0.95rem', marginTop: '2px' }}>
                      {isSuccess ? 'NFL Ready' : 'Unlikely'}
                    </div>
                  </div>
                </div>

                {/* Probability bar */}
                {typeof prediction.success_probability === 'number' && (
                  <div style={{ marginBottom: '1.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                      <span style={{ color: '#94a3b8', fontSize: '13px' }}>NFL Success Probability</span>
                      <span style={{ color: '#f1f5f9', fontWeight: 700 }}>{prediction.success_probability}%</span>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.07)', borderRadius: '999px', height: '10px', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', width: `${prediction.success_probability}%`, borderRadius: '999px',
                        background: isSuccess ? 'linear-gradient(90deg,#16a34a,#22c55e)' : 'linear-gradient(90deg,#b91c1c,#ef4444)',
                        transition: 'width 0.7s ease',
                      }} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569', fontSize: '11px', marginTop: '3px' }}>
                      <span>0% — Bust</span><span>50%</span><span>100% — Elite</span>
                    </div>
                  </div>
                )}

                {/* Scout profile + stats */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', marginBottom: '1.5rem' }}>
                  {prediction?.summary && (
                    <div>
                      <p style={{ color: '#475569', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>Scout Profile</p>
                      {Object.entries(prediction.summary).map(([k, v]) => (
                        <div key={k} style={{
                          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                          padding: '0.45rem 0.7rem', marginBottom: '0.35rem',
                          background: 'rgba(255,255,255,0.04)', borderRadius: '6px',
                        }}>
                          <span style={{ color: '#64748b', fontSize: '12px', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</span>
                          <span style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '13px' }}>{v}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {prediction?.stats && (
                    <div>
                      <p style={{ color: '#475569', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem' }}>
                        College Stats {prediction.data_source === 'espn_live' && <span style={{ color: '#22c55e', fontWeight: 700 }}>· Live ESPN</span>}
                      </p>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: '0.35rem' }}>
                        {[
                          ['Games', prediction.stats.games_played],
                          ['Pass TD', prediction.stats.passing_touchdowns],
                          ['Pass Yds', (prediction.stats.passing_yards || 0).toLocaleString()],
                          ['Rush TD', prediction.stats.rushing_touchdowns],
                          ['Rush Yds', (prediction.stats.rushing_yards || 0).toLocaleString()],
                          ...(prediction.summary?.completion_pct ? [
                            ['CMP%', prediction.summary.completion_pct],
                            ['INT', prediction.summary.interceptions ?? '—'],
                            ['QBR', prediction.summary.passer_rating ?? '—'],
                          ] : [
                            ['Prod.', prediction.stats.production_score],
                          ]),
                        ].map(([label, val]) => (
                          <div key={label} style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '6px', padding: '0.4rem', textAlign: 'center' }}>
                            <p style={{ color: '#475569', fontSize: '9px', margin: '0 0 2px', textTransform: 'uppercase' }}>{label}</p>
                            <p style={{ color: '#e2e8f0', fontWeight: 700, fontSize: '0.9rem', margin: 0 }}>{val ?? '—'}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Feature importance */}
                {Array.isArray(prediction?.top_factors) && prediction.top_factors.length > 0 && (
                  <div style={{ marginBottom: '1.25rem' }}>
                    <p style={{ color: '#475569', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.6rem' }}>Top Prediction Factors</p>
                    {prediction.top_factors.map(f => (
                      <div key={f.feature} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.45rem' }}>
                        <span style={{ color: '#94a3b8', fontSize: '12px', minWidth: '160px' }}>{f.feature}</span>
                        <div style={{ flex: 1, background: 'rgba(255,255,255,0.06)', borderRadius: '999px', height: '6px', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${Math.min(f.importance, 100)}%`, borderRadius: '999px', background: 'linear-gradient(90deg,#3b82f6,#60a5fa)' }} />
                        </div>
                        <span style={{ color: '#e2e8f0', fontSize: '11px', minWidth: '36px', textAlign: 'right', fontWeight: 600 }}>{f.importance}%</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Footer */}
                <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.75rem', color: '#475569', fontSize: '11px' }}>
                  <span>{prediction?.model_used ? '🤖 XGBoost Classifier' : '📐 Rule fallback'}</span>
                  <span>Source: {prediction?.data_source}</span>
                  <button onClick={() => { setPrediction(null); setSelected(null); }} style={{ background: 'none', border: 'none', color: '#475569', cursor: 'pointer', fontSize: '11px' }}>Clear ✕</button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

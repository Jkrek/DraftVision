import React, {
  useCallback, useEffect, useMemo, useRef, useState, memo,
} from 'react';
import '../App.css';

// ── Constants ──────────────────────────────────────────────────────────────
const PAGE_SIZE = 60;
const ALL_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'S', 'CB', 'LB', 'DL', 'OL', 'K', 'P'];
const POSITION_COLORS = {
  QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b',
  TE: '#a78bfa', S: '#f43f5e', CB: '#f43f5e',
  LB: '#fb923c', DL: '#ef4444', OL: '#64748b',
  K: '#94a3b8', P: '#94a3b8', default: '#94a3b8',
};

// ── Helpers ────────────────────────────────────────────────────────────────
function posColor(pos) { return POSITION_COLORS[pos] || POSITION_COLORS.default; }

// ── Grade color helper ─────────────────────────────────────────────────────
function gradeColor(grade) {
  if (!grade) return '#64748b';
  if (grade.startsWith('A')) return '#f59e0b';
  if (grade.startsWith('B')) return '#818cf8';
  if (grade.startsWith('C')) return '#64748b';
  return '#ef4444';
}

// ── SVG Radar Chart ────────────────────────────────────────────────────────
function RadarChart({ axes, size = 180 }) {
  const cx = size / 2, cy = size / 2, r = size * 0.33;
  const n   = axes.length;
  const ang = (i) => (Math.PI * 2 * i / n) - Math.PI / 2;
  const pt  = (i, v) => [cx + Math.cos(ang(i)) * r * v, cy + Math.sin(ang(i)) * r * v];
  const outerPts = axes.map((_, i) => pt(i, 1));
  const dataPts  = axes.map((a, i) => pt(i, Math.max(0.04, Math.min(1, a.value))));
  const poly = pts => pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ') + 'Z';
  return (
    <svg width={size} height={size} style={{ overflow: 'visible' }}>
      {[0.25, 0.5, 0.75, 1.0].map(ring => (
        <polygon key={ring}
          points={outerPts.map(([x,y]) => `${(cx+(x-cx)*ring).toFixed(1)},${(cy+(y-cy)*ring).toFixed(1)}`).join(' ')}
          fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="1" />
      ))}
      {outerPts.map(([x,y], i) => (
        <line key={i} x1={cx.toFixed(1)} y1={cy.toFixed(1)} x2={x.toFixed(1)} y2={y.toFixed(1)}
          stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
      ))}
      <path d={poly(dataPts)} fill="rgba(99,102,241,0.18)" stroke="#6366f1" strokeWidth="2" strokeLinejoin="round" />
      {dataPts.map(([x,y], i) => <circle key={i} cx={x.toFixed(1)} cy={y.toFixed(1)} r="3.5" fill="#818cf8" />)}
      {axes.map((a, i) => {
        const a2 = ang(i);
        const lx = cx + Math.cos(a2) * (r + 22), ly = cy + Math.sin(a2) * (r + 22);
        return (
          <text key={i} x={lx.toFixed(1)} y={ly.toFixed(1)} textAnchor="middle"
            dominantBaseline="middle" fontSize="9" fill="#64748b" fontFamily="system-ui,sans-serif">
            {a.label}
          </text>
        );
      })}
    </svg>
  );
}

// ── useWindowWidth ─────────────────────────────────────────────────────────
function useWindowWidth() {
  const [w, setW] = React.useState(typeof window !== 'undefined' ? window.innerWidth : 1200);
  useEffect(() => {
    const h = () => setW(window.innerWidth);
    window.addEventListener('resize', h);
    return () => window.removeEventListener('resize', h);
  }, []);
  return w;
}

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

  const isSuccess  = prediction?.success === 'Success';
  const windowWidth = useWindowWidth();
  const isMobile   = windowWidth < 900;

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
        <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '340px 1fr', gap: '1.25rem', alignItems: 'start' }}>

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

              {/* Position tabs */}
              <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
                {['ALL', ...ALL_POSITIONS].map(pos => (
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

            {prediction && !predicting && (() => {
              const pos    = prediction?.predicted_position || '?';
              const pc     = posColor(pos);
              const grade  = prediction?.prospect_grade;
              const gc     = gradeColor(grade);
              const dgc    = prediction?.draft_grade_class;
              const dgColor = dgc === 0 ? '#f59e0b' : dgc === 1 ? '#818cf8' : '#64748b';
              const phys   = prediction?.physical || {};
              const comps  = Array.isArray(prediction?.historical_comps) ? prediction.historical_comps : [];

              // Radar axes (0–1 scale)
              const radarAxes = [
                { label: 'Production',   value: (prediction.stats?.production_score  || 0) / 100 },
                { label: 'Athleticism',  value: (prediction.stats?.combine_speed_score || 0) / 100 },
                { label: 'Competition',  value: Math.max(0, (11 - (prediction.stats?.conference_tier || 5)) / 10) },
                { label: 'Physical',     value: (((phys.height_score || 50) + (phys.vert_score || 50)) / 200) },
                { label: 'Accolades',    value: Math.min(1, (prediction.stats?.is_award_winner || 0) * 0.6 + (prediction.stats?.is_all_american || 0) * 0.4) },
              ];

              // Determine which college stats to show (offensive vs defensive)
              const isDefensive = ['CB','S','DB','LB','DL','DE','DT'].includes(pos);
              const isOL = ['OL','OT','OG','C'].includes(pos);
              const statPairs = isDefensive
                ? [['Games', prediction.stats?.games_played], ['Tackles', prediction.stats?.tackles], ['Sacks', prediction.stats?.sacks], ['INTs', prediction.stats?.interceptions], ['PDs', prediction.stats?.pass_deflections], ['Prod.', prediction.stats?.production_score]]
                : isOL
                ? [['Games', prediction.stats?.games_played], ['Prod.', prediction.stats?.production_score]]
                : [['Games', prediction.stats?.games_played], ['Pass TD', prediction.stats?.passing_touchdowns], ['Pass Yds', (prediction.stats?.passing_yards||0).toLocaleString()], ['Rush TD', prediction.stats?.rushing_touchdowns], ['Rush Yds', (prediction.stats?.rushing_yards||0).toLocaleString()], ...(prediction.summary?.completion_pct ? [['CMP%', prediction.summary.completion_pct],['INT',prediction.summary.interceptions??'—'],['QBR',prediction.summary.passer_rating??'—']] : [['Prod.', prediction.stats?.production_score]])];

              return (
                <div style={{ ...glassCard, padding: '1.75rem' }}>

                  {/* ── Header ── */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '1.25rem' }}>
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
                        <span style={{ padding: '3px 10px', borderRadius: '999px', fontSize: '11px', fontWeight: 700, background: `${pc}22`, border: `1px solid ${pc}`, color: pc }}>{pos}</span>
                        <span style={{ color: '#64748b', fontSize: '12px' }}>{prediction?.stats?.team || ''}</span>
                      </div>
                      <h2 style={{ color: '#f1f5f9', margin: 0, fontSize: '1.5rem', fontWeight: 800, lineHeight: 1.2 }}>
                        {prediction?.resolved_name || selected?.name}
                      </h2>
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
                      {/* Prospect Grade — most prominent */}
                      {grade && (
                        <div style={{ padding: '0.5rem 1rem', borderRadius: '10px', textAlign: 'center', background: `${gc}18`, border: `2px solid ${gc}`, minWidth: '64px' }}>
                          <div style={{ fontSize: '0.65rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Grade</div>
                          <div style={{ fontWeight: 900, fontSize: '1.5rem', color: gc, lineHeight: 1.1, marginTop: '1px' }}>{grade}</div>
                        </div>
                      )}
                      {/* Draft Projection */}
                      {prediction?.draft_grade && (
                        <div style={{ padding: '0.5rem 0.8rem', borderRadius: '10px', textAlign: 'center', background: `${dgColor}15`, border: `2px solid ${dgColor}`, minWidth: '100px' }}>
                          <div style={{ fontSize: '0.65rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Draft</div>
                          <div style={{ fontWeight: 800, fontSize: '0.78rem', color: dgColor, marginTop: '2px' }}>{prediction.draft_grade}</div>
                        </div>
                      )}
                      {/* NFL Success */}
                      <div style={{ padding: '0.5rem 0.8rem', borderRadius: '10px', textAlign: 'center', background: isSuccess ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.1)', border: `2px solid ${isSuccess ? '#22c55e' : '#ef4444'}`, minWidth: '80px' }}>
                        <div style={{ fontSize: '0.65rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>NFL</div>
                        <div style={{ fontWeight: 800, fontSize: '0.78rem', color: isSuccess ? '#22c55e' : '#ef4444', marginTop: '2px' }}>{isSuccess ? 'Likely' : 'Unlikely'}</div>
                      </div>
                    </div>
                  </div>

                  {/* ── Probability bar ── */}
                  {typeof prediction.success_probability === 'number' && (
                    <div style={{ marginBottom: '1.5rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ color: '#94a3b8', fontSize: '12px' }}>NFL Success Probability</span>
                        <span style={{ color: '#f1f5f9', fontWeight: 700, fontSize: '13px' }}>{prediction.success_probability}%</span>
                      </div>
                      <div style={{ background: 'rgba(255,255,255,0.07)', borderRadius: '999px', height: '8px', overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${prediction.success_probability}%`, borderRadius: '999px', background: isSuccess ? 'linear-gradient(90deg,#16a34a,#22c55e)' : 'linear-gradient(90deg,#b91c1c,#ef4444)', transition: 'width 0.8s ease' }} />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569', fontSize: '10px', marginTop: '2px' }}>
                        <span>0% — Bust</span><span>50%</span><span>100% — Elite</span>
                      </div>
                    </div>
                  )}

                  {/* ── Radar Chart + Scout Profile ── */}
                  <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '200px 1fr', gap: '1.25rem', marginBottom: '1.5rem', alignItems: 'start' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <p style={{ color: '#475569', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', margin: '0 0 0.5rem' }}>Prospect Profile</p>
                      <RadarChart axes={radarAxes} size={isMobile ? 150 : 180} />
                    </div>
                    <div>
                      <p style={{ color: '#475569', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', margin: '0 0 0.5rem' }}>Scout Summary</p>
                      {prediction?.summary && Object.entries(prediction.summary).map(([k, v]) => (
                        <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.4rem 0.65rem', marginBottom: '0.3rem', background: 'rgba(255,255,255,0.04)', borderRadius: '6px' }}>
                          <span style={{ color: '#64748b', fontSize: '11px', textTransform: 'capitalize' }}>{k.replace(/_/g, ' ')}</span>
                          <span style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '12px' }}>{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* ── Physical Profile ── */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <p style={{ color: '#475569', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', margin: '0 0 0.6rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                      Physical Profile
                      {phys.is_real && <span style={{ color: '#22c55e', fontWeight: 700, fontSize: '8px' }}>· COMBINE</span>}
                      {!phys.is_real && <span style={{ color: '#475569', fontWeight: 600, fontSize: '8px' }}>· EST.</span>}
                    </p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))', gap: '0.4rem' }}>
                      {[
                        ['Height',   phys.display_height || '—'],
                        ['Weight',   phys.display_weight || '—'],
                        ['40-Yard',  phys.combine_forty ? `${phys.combine_forty}s` : '—'],
                        ['Vertical', phys.vertical_inches ? `${phys.vertical_inches}"` : '—'],
                        ['Bench',    phys.combine_bench  ? `${phys.combine_bench} reps` : '—'],
                        ['Broad',    phys.combine_broad  ? `${phys.combine_broad}"` : '—'],
                      ].map(([label, val]) => (
                        <div key={label} style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', padding: '0.5rem 0.4rem', textAlign: 'center' }}>
                          <p style={{ color: '#475569', fontSize: '8px', margin: '0 0 3px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</p>
                          <p style={{ color: val === '—' ? '#334155' : '#e2e8f0', fontWeight: 700, fontSize: '0.8rem', margin: 0 }}>{val}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* ── College Stats ── */}
                  <div style={{ marginBottom: '1.5rem' }}>
                    <p style={{ color: '#475569', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', margin: '0 0 0.5rem' }}>
                      College Stats {prediction.data_source === 'espn_live' && <span style={{ color: '#22c55e', fontWeight: 700 }}>· Live ESPN</span>}
                    </p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(72px, 1fr))', gap: '0.35rem' }}>
                      {statPairs.map(([label, val]) => (
                        <div key={label} style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '6px', padding: '0.4rem', textAlign: 'center' }}>
                          <p style={{ color: '#475569', fontSize: '8px', margin: '0 0 2px', textTransform: 'uppercase' }}>{label}</p>
                          <p style={{ color: '#e2e8f0', fontWeight: 700, fontSize: '0.85rem', margin: 0 }}>{val ?? '—'}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* ── Historical Comps ── */}
                  {comps.length > 0 && (
                    <div style={{ marginBottom: '1.5rem' }}>
                      <p style={{ color: '#475569', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', margin: '0 0 0.6rem' }}>Similar Profiles</p>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                        {comps.map((comp, i) => {
                          const cpc = posColor(comp.position);
                          const simColor = comp.similarity >= 80 ? '#22c55e' : comp.similarity >= 60 ? '#f59e0b' : '#64748b';
                          return (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', padding: '0.6rem 0.8rem' }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                <span style={{ fontSize: '9px', padding: '2px 6px', borderRadius: '4px', background: `${cpc}22`, color: cpc, border: `1px solid ${cpc}`, fontWeight: 700 }}>{comp.position}</span>
                                <div>
                                  <p style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.82rem', margin: 0 }}>{comp.name}</p>
                                  <p style={{ color: '#475569', fontSize: '11px', margin: '1px 0 0' }}>{comp.outcome}</p>
                                </div>
                              </div>
                              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                                <p style={{ color: simColor, fontWeight: 700, fontSize: '0.85rem', margin: 0 }}>{comp.similarity}%</p>
                                <p style={{ color: '#334155', fontSize: '9px', margin: 0 }}>match</p>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* ── Feature Importance ── */}
                  {Array.isArray(prediction?.top_factors) && prediction.top_factors.length > 0 && (
                    <div style={{ marginBottom: '1.25rem' }}>
                      <p style={{ color: '#475569', fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.6rem' }}>Top Prediction Factors</p>
                      {prediction.top_factors.map(f => (
                        <div key={f.feature} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
                          <span style={{ color: '#94a3b8', fontSize: '11px', minWidth: '150px' }}>{f.feature}</span>
                          <div style={{ flex: 1, background: 'rgba(255,255,255,0.06)', borderRadius: '999px', height: '5px', overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${Math.min(f.importance, 100)}%`, borderRadius: '999px', background: 'linear-gradient(90deg,#3b82f6,#818cf8)' }} />
                          </div>
                          <span style={{ color: '#e2e8f0', fontSize: '11px', minWidth: '32px', textAlign: 'right', fontWeight: 600 }}>{f.importance}%</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* ── Footer ── */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.75rem', color: '#475569', fontSize: '11px', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <span>{prediction?.model_used ? 'XGBoost · Two-model ML' : 'Rule fallback'}</span>
                    <span>Source: {prediction?.data_source}</span>
                    <button onClick={() => { setPrediction(null); setSelected(null); }} style={{ background: 'none', border: 'none', color: '#475569', cursor: 'pointer', fontSize: '11px' }}>Clear ✕</button>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import '../../App.css';

const ROUND_PICKS = 32; // picks per round

function getRound(pickNum) {
  return Math.ceil(pickNum / ROUND_PICKS);
}

const POSITION_COLORS = {
  QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#a78bfa',
  CB: '#f43f5e', S: '#f43f5e', DB: '#f43f5e', FS: '#f43f5e', SS: '#f43f5e',
  LB: '#fb923c', ILB: '#fb923c', OLB: '#fb923c',
  DL: '#ef4444', DE: '#ef4444', DT: '#ef4444', EDGE: '#ef4444', NT: '#ef4444',
  OL: '#64748b', OT: '#64748b', OG: '#64748b', C: '#64748b',
  default: '#94a3b8',
};

function posColor(pos) {
  return POSITION_COLORS[(pos || '').toUpperCase()] || POSITION_COLORS.default;
}

function pffGradeColor(grade) {
  const g = parseFloat(grade);
  if (isNaN(g)) return '#475569';
  if (g >= 85) return '#22c55e';
  if (g >= 70) return '#f59e0b';
  if (g >= 60) return '#64748b';
  return '#ef4444';
}

export default function MockDraft() {
  const navigate = useNavigate();
  const fileRef  = useRef(null);

  const [picks, setPicks]       = useState([]);
  const [title, setTitle]       = useState('');
  const [meta, setMeta]         = useState(null);
  const [loading, setLoading]   = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError]       = useState(null);
  const [dragOver, setDragOver] = useState(false);

  // Load existing draft on mount
  useEffect(() => {
    fetch('/api/mock-draft')
      .then(r => r.json())
      .then(data => {
        setPicks(Array.isArray(data.picks) ? data.picks : []);
        setTitle(data.title || '');
        setMeta(data.generated_at ? { generated_at: data.generated_at, total: data.total } : null);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const uploadCSV = useCallback(async (text) => {
    setUploading(true);
    setError(null);
    try {
      const res  = await fetch('/api/mock-draft/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ csv_content: text, title: 'JKrek\'s Mock Draft' }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');

      // Reload the draft
      const fresh = await fetch('/api/mock-draft').then(r => r.json());
      setPicks(Array.isArray(fresh.picks) ? fresh.picks : []);
      setTitle(fresh.title || '');
      setMeta({ generated_at: fresh.generated_at, total: fresh.total });
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }, []);

  const handleFile = useCallback((file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => uploadCSV(e.target.result);
    reader.readAsText(file);
  }, [uploadCSV]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  // Group picks by round
  const rounds = picks.reduce((acc, pick) => {
    const rnd = pick.round ? String(pick.round) : String(getRound(pick.pick || 1));
    if (!acc[rnd]) acc[rnd] = [];
    acc[rnd].push(pick);
    return acc;
  }, {});

  const roundKeys = Object.keys(rounds).sort((a, b) => {
    const na = parseInt(a) || 99, nb = parseInt(b) || 99;
    return na - nb;
  });

  const isEmpty = picks.length === 0 && !loading;

  return (
    <div style={{ minHeight: '100vh', background: 'var(--background-dark)', padding: '2rem 1rem 5rem' }}>

      {/* Header */}
      <div style={{ maxWidth: 900, margin: '0 auto 2rem' }}>
        <h1 style={{
          textAlign: 'center', margin: '0 0 0.4rem', fontSize: 'clamp(1.6rem,4vw,2.2rem)',
          fontWeight: 800, letterSpacing: '-0.5px',
          background: 'linear-gradient(135deg,#f59e0b,#ef4444)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
        }}>
          {title || 'JKrek\'s Mock Draft'}
        </h1>
        <p style={{ textAlign: 'center', color: '#64748b', margin: '0 0 0.4rem', fontSize: '0.95rem' }}>
          Official pick-by-pick NFL Draft board
        </p>
        {meta && (
          <p style={{ textAlign: 'center', color: '#475569', fontSize: '0.78rem', margin: 0 }}>
            {meta.total} picks
            {meta.generated_at && ` · updated ${new Date(meta.generated_at).toLocaleDateString()}`}
          </p>
        )}
        {error && (
          <p style={{ textAlign: 'center', color: '#ef4444', fontSize: '0.85rem', marginTop: '0.5rem' }}>{error}</p>
        )}
      </div>

      <div style={{ maxWidth: 900, margin: '0 auto' }}>

        {/* Upload zone — always visible at top */}
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          style={{
            border: `2px dashed ${dragOver ? '#f59e0b' : 'rgba(245,158,11,0.3)'}`,
            borderRadius: 12, padding: '1.2rem 2rem', textAlign: 'center',
            cursor: 'pointer', marginBottom: '1.5rem',
            background: dragOver ? 'rgba(245,158,11,0.06)' : 'rgba(245,158,11,0.02)',
            transition: 'all 0.15s',
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv,text/csv"
            style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])}
          />
          <p style={{ color: '#f59e0b', fontWeight: 700, margin: '0 0 0.3rem', fontSize: '0.95rem' }}>
            {uploading ? 'Uploading…' : picks.length > 0 ? 'Update Mock Draft' : 'Upload PFF Mock Draft CSV'}
          </p>
          <p style={{ color: '#475569', fontSize: '0.78rem', margin: 0 }}>
            {uploading ? 'Parsing picks…' : 'Drag & drop or click · CSV export from PFF Mock Draft Simulator'}
          </p>
        </div>

        {/* Empty state */}
        {isEmpty && (
          <div style={{
            background: 'rgba(15,23,42,0.7)', border: '1px solid rgba(255,255,255,0.07)',
            borderRadius: 12, padding: '4rem 2rem', textAlign: 'center',
          }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🏈</div>
            <p style={{ color: '#64748b', fontSize: '1rem', margin: 0 }}>
              No mock draft loaded yet.<br />
              Export your draft from PFF and upload the CSV above.
            </p>
          </div>
        )}

        {loading && (
          <div style={{ textAlign: 'center', color: '#475569', padding: '3rem' }}>Loading…</div>
        )}

        {/* Draft board — grouped by round */}
        {roundKeys.map(rndKey => (
          <div key={rndKey} style={{ marginBottom: '1.5rem' }}>
            {/* Round divider */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '0.8rem', marginBottom: '0.6rem',
            }}>
              <div style={{
                background: 'linear-gradient(90deg,#f59e0b22,transparent)',
                border: '1px solid rgba(245,158,11,0.2)',
                borderRadius: 6, padding: '3px 12px',
                color: '#f59e0b', fontSize: '0.72rem', fontWeight: 800, letterSpacing: '0.1em',
                textTransform: 'uppercase',
              }}>
                {isNaN(parseInt(rndKey)) ? rndKey : `Round ${rndKey}`}
              </div>
              <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.06)' }} />
              <span style={{ color: '#334155', fontSize: '0.7rem' }}>{rounds[rndKey].length} picks</span>
            </div>

            {/* Pick rows */}
            <div style={{
              background: 'rgba(15,23,42,0.7)', border: '1px solid rgba(255,255,255,0.07)',
              borderRadius: 12, overflow: 'hidden',
            }}>
              {rounds[rndKey].map((pick, i) => {
                const pc  = posColor(pick.position);
                const gc  = pffGradeColor(pick.pff_grade);
                const teamColor = pick.color || '#334155';

                return (
                  <div
                    key={`${pick.pick}-${i}`}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '3rem 4px 2.2fr 1.8fr 2.8rem 3.5rem 4rem',
                      gap: '0 0.75rem', padding: '0.7rem 1rem',
                      alignItems: 'center',
                      borderBottom: i < rounds[rndKey].length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(245,158,11,0.05)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    {/* Pick number */}
                    <span style={{ color: '#334155', fontSize: '0.78rem', fontWeight: 700 }}>
                      #{pick.pick}
                    </span>

                    {/* Team color bar */}
                    <div style={{ width: 4, height: 36, borderRadius: 2, background: teamColor, flexShrink: 0 }} />

                    {/* Player name + school */}
                    <div style={{ minWidth: 0 }}>
                      <div
                        onClick={() => pick.player && navigate(`/predict?name=${encodeURIComponent(pick.player)}`)}
                        style={{
                          color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 700,
                          cursor: pick.player ? 'pointer' : 'default',
                          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        }}
                        onMouseEnter={e => { if (pick.player) e.currentTarget.style.color = '#f59e0b'; }}
                        onMouseLeave={e => e.currentTarget.style.color = '#e2e8f0'}
                      >
                        {pick.player || '—'}
                      </div>
                      {pick.school && (
                        <div style={{
                          color: '#475569', fontSize: '0.72rem',
                          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        }}>
                          {pick.school}
                        </div>
                      )}
                    </div>

                    {/* NFL team */}
                    <div style={{
                      color: '#94a3b8', fontSize: '0.78rem',
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {pick.nfl_team || '—'}
                    </div>

                    {/* Position badge */}
                    {pick.position ? (
                      <span style={{
                        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                        width: 30, height: 20, borderRadius: 4,
                        background: `${pc}18`, border: `1px solid ${pc}40`,
                        color: pc, fontSize: '0.65rem', fontWeight: 700,
                      }}>{pick.position}</span>
                    ) : <span />}

                    {/* PFF grade */}
                    {pick.pff_grade ? (
                      <span style={{
                        color: gc, fontSize: '0.82rem', fontWeight: 800, textAlign: 'center',
                      }}>{pick.pff_grade}</span>
                    ) : <span />}

                    {/* Predict button */}
                    {pick.player && (
                      <button
                        onClick={() => navigate(`/predict?name=${encodeURIComponent(pick.player)}`)}
                        style={{
                          padding: '4px 8px', borderRadius: 6, fontSize: '0.68rem', fontWeight: 600,
                          border: '1px solid rgba(99,102,241,0.35)',
                          background: 'rgba(99,102,241,0.1)', color: '#818cf8',
                          cursor: 'pointer', whiteSpace: 'nowrap',
                        }}
                      >
                        Predict
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import '../../App.css';

const POSITION_TABS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'DB', 'LB', 'DL', 'OL'];
const GRADE_FILTERS = ['ALL', 'A', 'B', 'C', 'D'];
const PAGE_SIZE = 100;

const POSITION_COLORS = {
  QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#a78bfa',
  DB: '#f43f5e', LB: '#fb923c', DL: '#ef4444', OL: '#64748b',
  default: '#94a3b8',
};

const POS_GROUP_MAP = {
  DB: new Set(['CB', 'S', 'DB', 'FS', 'SS']),
  LB: new Set(['LB', 'ILB', 'OLB', 'MLB']),
  DL: new Set(['DL', 'DE', 'DT', 'EDGE', 'NT']),
  OL: new Set(['OL', 'OT', 'OG', 'C', 'LS']),
};

const DRAFT_SHORT = {
  'Top 50 Pick':        '🟢 Top 50',
  'Day 2 Pick':         '🔵 Day 2',
  'Late Round Pick':    '🟡 Late Rd',
  'Undrafted Prospect': '⬜ UDFA',
};

const GRADE_ORDER = { 'A+': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8, 'D': 9 };

function posColor(pos) {
  const p = (pos || '').toUpperCase();
  for (const [group, set] of Object.entries(POS_GROUP_MAP)) {
    if (set.has(p)) return POSITION_COLORS[group];
  }
  return POSITION_COLORS[p] || POSITION_COLORS.default;
}

function gradeColor(grade) {
  if (!grade) return '#64748b';
  if (grade.startsWith('A')) return '#f59e0b';
  if (grade.startsWith('B')) return '#818cf8';
  if (grade.startsWith('C')) return '#64748b';
  return '#ef4444';
}

function posMatchesTab(pos, tab) {
  if (tab === 'ALL') return true;
  const p = (pos || '').toUpperCase();
  const group = POS_GROUP_MAP[tab];
  return group ? group.has(p) : p === tab;
}

export default function Leaderboard() {
  const navigate = useNavigate();

  const [prospects, setProspects] = useState([]);
  const [meta, setMeta]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  const [posTab, setPosTab]           = useState('ALL');
  const [gradeFilter, setGradeFilter] = useState('ALL');
  const [search, setSearch]           = useState('');
  const [sortBy, setSortBy]           = useState('grade');
  const [page, setPage]               = useState(0);

  useEffect(() => {
    setLoading(true);
    fetch('/api/prospects?limit=2000')
      .then(r => r.json())
      .then(data => {
        setProspects(Array.isArray(data.prospects) ? data.prospects : []);
        setMeta(data.meta || null);
        setError(null);
      })
      .catch(() => setError('Failed to load leaderboard.'))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    let list = prospects;
    if (posTab !== 'ALL')
      list = list.filter(p => posMatchesTab(p.position, posTab));
    if (gradeFilter !== 'ALL')
      list = list.filter(p => (p.grade || '').startsWith(gradeFilter));
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(p =>
        (p.name || '').toLowerCase().includes(q) ||
        (p.team || '').toLowerCase().includes(q),
      );
    }
    if (sortBy === 'name')
      return [...list].sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    if (sortBy === 'success')
      return [...list].sort((a, b) => (b.success_probability || 0) - (a.success_probability || 0));
    if (sortBy === 'team')
      return [...list].sort((a, b) => (a.team || '').localeCompare(b.team || ''));
    // default: grade
    return [...list].sort((a, b) =>
      (GRADE_ORDER[a.grade] ?? 9) - (GRADE_ORDER[b.grade] ?? 9) ||
      (b.success_probability || 0) - (a.success_probability || 0),
    );
  }, [prospects, posTab, gradeFilter, search, sortBy]);

  const total   = filtered.length;
  const shown   = useMemo(() => filtered.slice(0, (page + 1) * PAGE_SIZE), [filtered, page]);
  const hasMore = shown.length < total;

  const handlePlayerClick = useCallback((p) => {
    navigate(`/predict?name=${encodeURIComponent(p.name)}`);
  }, [navigate]);

  const resetFilters = useCallback(() => {
    setPosTab('ALL'); setGradeFilter('ALL'); setSearch(''); setSortBy('grade'); setPage(0);
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div style={{ minHeight: '100vh', background: 'var(--background-dark)', padding: '2rem 1rem 5rem' }}>

      {/* ── Header ── */}
      <div style={{ maxWidth: 1100, margin: '0 auto 2rem' }}>
        <h1 style={{
          textAlign: 'center', margin: '0 0 0.5rem', fontSize: 'clamp(1.6rem, 4vw, 2.2rem)',
          fontWeight: 800, letterSpacing: '-0.5px',
          background: 'linear-gradient(135deg,#818cf8,#c084fc)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
        }}>
          Prospect Leaderboard
        </h1>
        <p style={{ textAlign: 'center', color: '#64748b', margin: '0 0 0.4rem', fontSize: '0.95rem' }}>
          ML-graded predictions for every Power 5 college prospect
        </p>
        {meta ? (
          <p style={{ textAlign: 'center', color: '#475569', fontSize: '0.78rem', margin: 0 }}>
            {(meta.total || prospects.length).toLocaleString()} prospects
            {meta.generated_at && ` · updated ${new Date(meta.generated_at).toLocaleDateString()}`}
          </p>
        ) : !loading && (
          <p style={{ textAlign: 'center', color: '#f59e0b', fontSize: '0.82rem', margin: 0 }}>
            Cache empty — run{' '}
            <code style={{ background: 'rgba(245,158,11,0.12)', padding: '2px 6px', borderRadius: 4, fontSize: '0.8rem' }}>
              python build_prospect_cache.py
            </code>{' '}
            to populate
          </p>
        )}
        {error && <p style={{ textAlign: 'center', color: '#ef4444', fontSize: '0.85rem', margin: 0 }}>{error}</p>}
      </div>

      {/* ── Main ── */}
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>

        {/* ── Filter bar ── */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem', marginBottom: '1.1rem', alignItems: 'center' }}>

          {/* Position tabs */}
          <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
            {POSITION_TABS.map(pos => {
              const active = posTab === pos;
              const color  = pos === 'ALL' ? '#818cf8' : posColor(pos);
              return (
                <button key={pos} onClick={() => { setPosTab(pos); setPage(0); }} style={{
                  padding: '5px 13px', borderRadius: 20, fontSize: '0.76rem', fontWeight: 700,
                  border: `1px solid ${active ? color : 'rgba(255,255,255,0.08)'}`,
                  background: active ? color : 'rgba(255,255,255,0.04)',
                  color: active ? '#fff' : '#64748b', cursor: 'pointer', transition: 'all 0.12s',
                }}>{pos}</button>
              );
            })}
          </div>

          {/* Grade filter */}
          <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
            {GRADE_FILTERS.map(g => {
              const active = gradeFilter === g;
              const color  = g === 'ALL' ? '#64748b' : gradeColor(g);
              return (
                <button key={g} onClick={() => { setGradeFilter(g); setPage(0); }} style={{
                  padding: '5px 13px', borderRadius: 20, fontSize: '0.76rem', fontWeight: 700,
                  border: `1px solid ${active ? color : 'rgba(255,255,255,0.08)'}`,
                  background: active ? `${color}25` : 'rgba(255,255,255,0.04)',
                  color: active ? color : '#64748b', cursor: 'pointer', transition: 'all 0.12s',
                }}>{g === 'ALL' ? 'All Grades' : `${g} Grade`}</button>
              );
            })}
          </div>

          {/* Search */}
          <input
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            placeholder="Search name or school…"
            style={{
              padding: '7px 12px', borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(15,23,42,0.7)', color: '#e2e8f0',
              fontSize: '0.85rem', outline: 'none', minWidth: 180, flex: 1,
            }}
          />

          {/* Sort */}
          <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={{
            padding: '7px 12px', borderRadius: 8,
            border: '1px solid rgba(255,255,255,0.1)',
            background: 'rgba(15,23,42,0.9)', color: '#e2e8f0',
            fontSize: '0.85rem', cursor: 'pointer',
          }}>
            <option value="grade">Sort: Grade</option>
            <option value="success">Sort: Success %</option>
            <option value="name">Sort: Name</option>
            <option value="team">Sort: School</option>
          </select>
        </div>

        {/* ── Result count ── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
          <p style={{ color: '#475569', fontSize: '0.8rem', margin: 0 }}>
            {loading ? 'Loading…' : `${total.toLocaleString()} prospect${total !== 1 ? 's' : ''}`}
          </p>
          {(posTab !== 'ALL' || gradeFilter !== 'ALL' || search) && (
            <button onClick={resetFilters} style={{
              background: 'none', border: 'none', color: '#475569',
              fontSize: '0.78rem', cursor: 'pointer', textDecoration: 'underline',
            }}>Clear filters</button>
          )}
        </div>

        {/* ── Table ── */}
        <div style={{
          background: 'rgba(15,23,42,0.7)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: 12, overflow: 'hidden',
        }}>
          {/* Header row */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '2.4rem 1fr 1.5fr 3.2rem 3.8rem 5rem 7.5rem',
            gap: '0 0.5rem', padding: '0.65rem 1rem',
            background: 'rgba(255,255,255,0.025)',
            borderBottom: '1px solid rgba(255,255,255,0.07)',
          }}>
            {['#', 'Name', 'School', 'Pos', 'Grade', 'Success', 'Draft Proj'].map(col => (
              <span key={col} style={{
                color: '#334155', fontSize: '0.68rem', fontWeight: 700,
                textTransform: 'uppercase', letterSpacing: '0.06em',
              }}>{col}</span>
            ))}
          </div>

          {/* Rows */}
          {loading ? (
            <div style={{ padding: '3rem', textAlign: 'center', color: '#475569' }}>Loading prospects…</div>
          ) : shown.length === 0 ? (
            <div style={{ padding: '3rem', textAlign: 'center', color: '#475569' }}>
              {prospects.length === 0
                ? 'No prospects cached — run build_prospect_cache.py to populate'
                : 'No prospects match your filters'}
            </div>
          ) : shown.map((p, i) => {
            const gc = gradeColor(p.grade);
            const pc = posColor(p.position);
            const sp = p.success_probability || 0;
            const spColor = sp >= 65 ? '#22c55e' : sp >= 45 ? '#f59e0b' : '#64748b';

            return (
              <div
                key={`${p.name}-${i}`}
                onClick={() => handlePlayerClick(p)}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '2.4rem 1fr 1.5fr 3.2rem 3.8rem 5rem 7.5rem',
                  gap: '0 0.5rem', padding: '0.65rem 1rem',
                  cursor: 'pointer', alignItems: 'center',
                  borderBottom: '1px solid rgba(255,255,255,0.035)',
                  transition: 'background 0.1s',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(99,102,241,0.08)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                {/* Rank */}
                <span style={{ color: '#334155', fontSize: '0.72rem', fontWeight: 600 }}>
                  {i + 1}
                </span>

                {/* Name */}
                <span style={{
                  color: '#e2e8f0', fontSize: '0.87rem', fontWeight: 600,
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>{p.name}</span>

                {/* School */}
                <span style={{
                  color: '#64748b', fontSize: '0.78rem',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>{p.team || '—'}</span>

                {/* Position badge */}
                <span style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: 30, height: 20, borderRadius: 4,
                  background: `${pc}18`, border: `1px solid ${pc}40`,
                  color: pc, fontSize: '0.68rem', fontWeight: 700,
                }}>{(p.position || '?').toUpperCase()}</span>

                {/* Grade badge */}
                <span style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: 34, height: 24, borderRadius: 6,
                  background: `${gc}18`, border: `1.5px solid ${gc}`,
                  color: gc, fontSize: '0.8rem', fontWeight: 800,
                }}>{p.grade || '—'}</span>

                {/* Success % */}
                <span style={{ color: spColor, fontSize: '0.85rem', fontWeight: 700 }}>
                  {p.success_probability != null ? `${Math.round(sp)}%` : '—'}
                </span>

                {/* Draft projection */}
                <span style={{ color: '#475569', fontSize: '0.74rem' }}>
                  {DRAFT_SHORT[p.draft_grade] || p.draft_grade || '—'}
                </span>
              </div>
            );
          })}
        </div>

        {/* ── Load more ── */}
        {hasMore && (
          <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
            <button onClick={() => setPage(pg => pg + 1)} style={{
              padding: '10px 28px', borderRadius: 8,
              border: '1px solid rgba(99,102,241,0.4)',
              background: 'rgba(99,102,241,0.1)', color: '#818cf8',
              fontSize: '0.9rem', fontWeight: 600, cursor: 'pointer',
            }}>
              Load more ({(total - shown.length).toLocaleString()} remaining)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


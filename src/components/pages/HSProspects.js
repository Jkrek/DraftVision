import React, { useState, useEffect, useMemo, useCallback } from 'react';
import '../../App.css';

const POSITION_TABS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'DB', 'LB', 'DL', 'OL'];
const STARS_FILTERS = ['ALL', '5', '4', '3'];
const PAGE_SIZE     = 100;

const POSITION_COLORS = {
  QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#a78bfa',
  DB: '#f43f5e', LB: '#fb923c', DL: '#ef4444', OL: '#64748b',
  default: '#94a3b8',
};

const POS_GROUP_MAP = {
  DB: new Set(['CB', 'S', 'DB', 'FS', 'SS']),
  LB: new Set(['LB', 'ILB', 'OLB', 'MLB']),
  DL: new Set(['DL', 'DE', 'DT', 'EDGE', 'NT']),
  OL: new Set(['OL', 'OT', 'OG', 'C']),
};

function posColor(pos) {
  const p = (pos || '').toUpperCase();
  for (const [group, set] of Object.entries(POS_GROUP_MAP)) {
    if (set.has(p)) return POSITION_COLORS[group];
  }
  return POSITION_COLORS[p] || POSITION_COLORS.default;
}

function posMatchesTab(pos, tab) {
  if (tab === 'ALL') return true;
  const p = (pos || '').toUpperCase();
  const group = POS_GROUP_MAP[tab];
  return group ? group.has(p) : p === tab;
}

function StarRating({ stars }) {
  const s = parseInt(stars) || 0;
  return (
    <span style={{ color: '#f59e0b', fontSize: '0.72rem', letterSpacing: '-1px' }}>
      {'★'.repeat(s)}
      <span style={{ color: '#1e293b' }}>{'★'.repeat(Math.max(0, 5 - s))}</span>
    </span>
  );
}

function gradeColor(grade) {
  if (!grade) return '#64748b';
  if (grade.startsWith('A')) return '#f59e0b';
  if (grade.startsWith('B')) return '#818cf8';
  if (grade.startsWith('C')) return '#64748b';
  return '#ef4444';
}

export default function HSProspects() {
  const [prospects, setProspects]   = useState([]);
  const [meta, setMeta]             = useState(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState(null);
  const [apiKeyMissing, setApiKeyMissing] = useState(false);

  const [posTab, setPosTab]         = useState('ALL');
  const [starsFilter, setStarsFilter] = useState('ALL');
  const [yearFilter, setYearFilter] = useState('ALL');
  const [search, setSearch]         = useState('');
  const [sortBy, setSortBy]         = useState('rank');
  const [page, setPage]             = useState(0);

  useEffect(() => {
    setLoading(true);
    fetch('/api/hs-prospects?limit=2000')
      .then(r => r.json())
      .then(data => {
        setProspects(Array.isArray(data.prospects) ? data.prospects : []);
        setMeta(data.meta || null);
        setApiKeyMissing(!data.api_key_set);
        setError(null);
      })
      .catch(() => setError('Failed to load HS prospects.'))
      .finally(() => setLoading(false));
  }, []);

  // Collect unique years from data
  const availableYears = useMemo(() => {
    const ys = [...new Set(prospects.map(p => p.year).filter(Boolean))].sort((a, b) => b - a);
    return ys;
  }, [prospects]);

  const filtered = useMemo(() => {
    let list = prospects;
    if (posTab !== 'ALL')
      list = list.filter(p => posMatchesTab(p.position, posTab));
    if (starsFilter !== 'ALL')
      list = list.filter(p => String(p.stars) === starsFilter);
    if (yearFilter !== 'ALL')
      list = list.filter(p => String(p.year) === yearFilter);
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(p =>
        (p.name || '').toLowerCase().includes(q) ||
        (p.school || '').toLowerCase().includes(q) ||
        (p.committed_to || '').toLowerCase().includes(q) ||
        (p.state || '').toLowerCase().includes(q),
      );
    }
    if (sortBy === 'stars')
      return [...list].sort((a, b) => (b.stars || 0) - (a.stars || 0) || (a.ranking || 9999) - (b.ranking || 9999));
    if (sortBy === 'rating')
      return [...list].sort((a, b) => (b.rating || 0) - (a.rating || 0));
    if (sortBy === 'name')
      return [...list].sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    // default: rank
    return [...list].sort((a, b) => (a.ranking || 9999) - (b.ranking || 9999));
  }, [prospects, posTab, starsFilter, yearFilter, search, sortBy]);

  const total   = filtered.length;
  const shown   = useMemo(() => filtered.slice(0, (page + 1) * PAGE_SIZE), [filtered, page]);
  const hasMore = shown.length < total;

  const resetFilters = useCallback(() => {
    setPosTab('ALL'); setStarsFilter('ALL'); setYearFilter('ALL');
    setSearch(''); setSortBy('rank'); setPage(0);
  }, []);

  const hasActiveFilter = posTab !== 'ALL' || starsFilter !== 'ALL' || yearFilter !== 'ALL' || search;

  return (
    <div style={{ minHeight: '100vh', background: 'var(--background-dark)', padding: '2rem 1rem 5rem' }}>

      {/* Header */}
      <div style={{ maxWidth: 1100, margin: '0 auto 2rem' }}>
        <h1 style={{
          textAlign: 'center', margin: '0 0 0.5rem', fontSize: 'clamp(1.6rem,4vw,2.2rem)',
          fontWeight: 800, letterSpacing: '-0.5px',
          background: 'linear-gradient(135deg,#f59e0b,#22c55e)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
        }}>
          High School Prospects
        </h1>
        <p style={{ textAlign: 'center', color: '#64748b', margin: '0 0 0.4rem', fontSize: '0.95rem' }}>
          Top-ranked recruiting classes — powered by 247Sports composite ratings
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
              python build_hs_prospect_cache.py
            </code>{' '}
            to populate
          </p>
        )}
        {apiKeyMissing && (
          <p style={{ textAlign: 'center', color: '#64748b', fontSize: '0.78rem', margin: '0.3rem 0 0' }}>
            Set{' '}
            <code style={{ background: 'rgba(100,116,139,0.15)', padding: '1px 5px', borderRadius: 3 }}>
              CFBD_API_KEY
            </code>{' '}
            env var (free at collegefootballdata.com) to fetch live data
          </p>
        )}
        {error && <p style={{ textAlign: 'center', color: '#ef4444', fontSize: '0.85rem', margin: 0 }}>{error}</p>}
      </div>

      <div style={{ maxWidth: 1100, margin: '0 auto' }}>

        {/* Filter bar */}
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

          {/* Stars filter */}
          <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
            {STARS_FILTERS.map(s => {
              const active = starsFilter === s;
              return (
                <button key={s} onClick={() => { setStarsFilter(s); setPage(0); }} style={{
                  padding: '5px 13px', borderRadius: 20, fontSize: '0.76rem', fontWeight: 700,
                  border: `1px solid ${active ? '#f59e0b' : 'rgba(255,255,255,0.08)'}`,
                  background: active ? 'rgba(245,158,11,0.2)' : 'rgba(255,255,255,0.04)',
                  color: active ? '#f59e0b' : '#64748b', cursor: 'pointer', transition: 'all 0.12s',
                }}>
                  {s === 'ALL' ? 'All Stars' : `${'★'.repeat(parseInt(s))} ${s}★`}
                </button>
              );
            })}
          </div>

          {/* Year filter */}
          {availableYears.length > 1 && (
            <select value={yearFilter} onChange={e => { setYearFilter(e.target.value); setPage(0); }} style={{
              padding: '7px 12px', borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(15,23,42,0.9)', color: '#e2e8f0',
              fontSize: '0.85rem', cursor: 'pointer',
            }}>
              <option value="ALL">All Classes</option>
              {availableYears.map(y => <option key={y} value={String(y)}>Class of {y}</option>)}
            </select>
          )}

          {/* Search */}
          <input
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            placeholder="Search name, school, state, commit…"
            style={{
              padding: '7px 12px', borderRadius: 8,
              border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(15,23,42,0.7)', color: '#e2e8f0',
              fontSize: '0.85rem', outline: 'none', minWidth: 200, flex: 1,
            }}
          />

          {/* Sort */}
          <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={{
            padding: '7px 12px', borderRadius: 8,
            border: '1px solid rgba(255,255,255,0.1)',
            background: 'rgba(15,23,42,0.9)', color: '#e2e8f0',
            fontSize: '0.85rem', cursor: 'pointer',
          }}>
            <option value="rank">Sort: National Rank</option>
            <option value="stars">Sort: Stars</option>
            <option value="rating">Sort: Rating</option>
            <option value="name">Sort: Name</option>
          </select>
        </div>

        {/* Result count */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
          <p style={{ color: '#475569', fontSize: '0.8rem', margin: 0 }}>
            {loading ? 'Loading…' : `${total.toLocaleString()} prospect${total !== 1 ? 's' : ''}`}
          </p>
          {hasActiveFilter && (
            <button onClick={resetFilters} style={{
              background: 'none', border: 'none', color: '#475569',
              fontSize: '0.78rem', cursor: 'pointer', textDecoration: 'underline',
            }}>Clear filters</button>
          )}
        </div>

        {/* Table */}
        <div style={{
          background: 'rgba(15,23,42,0.7)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: 12, overflow: 'hidden',
        }}>
          {/* Header */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '3rem 1fr 1.6fr 3.2rem 4.2rem 3.2rem 4.5rem 5.5rem',
            gap: '0 0.5rem', padding: '0.65rem 1rem',
            background: 'rgba(255,255,255,0.025)',
            borderBottom: '1px solid rgba(255,255,255,0.07)',
          }}>
            {['Rank', 'Name', 'HS / City', 'Pos', 'Stars', 'Yr', 'Rating', 'Commit'].map(col => (
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
                ? 'No HS prospects cached — run build_hs_prospect_cache.py to populate'
                : 'No prospects match your filters'}
            </div>
          ) : shown.map((p, i) => {
            const pc = posColor(p.position);

            return (
              <div
                key={`${p.name}-${i}`}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '3rem 1fr 1.6fr 3.2rem 4.2rem 3.2rem 4.5rem 5.5rem',
                  gap: '0 0.5rem', padding: '0.65rem 1rem',
                  alignItems: 'center',
                  borderBottom: '1px solid rgba(255,255,255,0.035)',
                  cursor: 'default',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(245,158,11,0.05)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                {/* Rank */}
                <span style={{ color: p.ranking <= 25 ? '#f59e0b' : '#334155', fontSize: '0.78rem', fontWeight: 700 }}>
                  #{p.ranking || '—'}
                </span>

                {/* Name */}
                <span style={{
                  color: '#e2e8f0', fontSize: '0.87rem', fontWeight: 600,
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>{p.name}</span>

                {/* HS / city */}
                <div style={{ minWidth: 0 }}>
                  <div style={{
                    color: '#64748b', fontSize: '0.78rem',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>{p.school || '—'}</div>
                  {(p.city || p.state) && (
                    <div style={{ color: '#334155', fontSize: '0.68rem' }}>
                      {[p.city, p.state].filter(Boolean).join(', ')}
                    </div>
                  )}
                </div>

                {/* Position */}
                <span style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                  width: 30, height: 20, borderRadius: 4,
                  background: `${pc}18`, border: `1px solid ${pc}40`,
                  color: pc, fontSize: '0.68rem', fontWeight: 700,
                }}>{(p.position || '?').toUpperCase()}</span>

                {/* Stars */}
                <StarRating stars={p.stars} />

                {/* Class year */}
                <span style={{ color: '#475569', fontSize: '0.78rem' }}>{p.year || '—'}</span>

                {/* Rating */}
                <span style={{
                  color: p.rating >= 0.99 ? '#f59e0b' : p.rating >= 0.93 ? '#22c55e' : '#64748b',
                  fontSize: '0.82rem', fontWeight: 700,
                }}>
                  {p.rating ? p.rating.toFixed(4) : '—'}
                </span>

                {/* Committed to */}
                <span style={{
                  color: p.committed_to ? '#818cf8' : '#334155',
                  fontSize: '0.74rem', fontWeight: p.committed_to ? 600 : 400,
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>
                  {p.committed_to || 'Uncommitted'}
                </span>
              </div>
            );
          })}
        </div>

        {/* Load more */}
        {hasMore && (
          <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
            <button onClick={() => setPage(pg => pg + 1)} style={{
              padding: '10px 28px', borderRadius: 8,
              border: '1px solid rgba(245,158,11,0.4)',
              background: 'rgba(245,158,11,0.1)', color: '#f59e0b',
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

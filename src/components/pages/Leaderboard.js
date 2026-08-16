import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import './Leaderboard.css';

const PAGE_SIZE = 100;

const POS_GROUP_MAP = {
  DB: new Set(['CB', 'S', 'DB', 'FS', 'SS']),
  LB: new Set(['LB', 'ILB', 'OLB', 'MLB']),
  DL: new Set(['DL', 'DE', 'DT', 'EDGE', 'NT']),
  OL: new Set(['OL', 'OT', 'OG', 'C', 'LS']),
};

const GROUP_ORDER = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB'];

const DRAFT_SHORT = {
  'Top 50 Pick':        'Top 50',
  'Day 2 Pick':         'Day 2',
  'Late Round Pick':    'Late round',
  'Undrafted Prospect': 'UDFA',
};

const GRADE_ORDER = { 'A+': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8, 'D': 9 };

function posGroup(pos) {
  const p = (pos || '').toUpperCase();
  for (const [group, set] of Object.entries(POS_GROUP_MAP)) {
    if (set.has(p)) return group;
  }
  return p || '?';
}

function posMatchesTab(pos, tab) {
  if (tab === 'ALL') return true;
  const p = (pos || '').toUpperCase();
  const group = POS_GROUP_MAP[tab];
  return group ? group.has(p) : p === tab;
}

const clampPct = (v) => Math.max(0, Math.min(100, Number(v) || 0));

export default function Leaderboard() {
  const navigate = useNavigate();

  const [prospects, setProspects] = useState([]);
  const [meta, setMeta]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  const [posTab, setPosTab]     = useState('ALL');
  const [search, setSearch]     = useState('');
  const [sortBy, setSortBy]     = useState('grade');
  const [page, setPage]         = useState(0);
  const [expanded, setExpanded] = useState(null);
  const [animKey, setAnimKey]   = useState(0);

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
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(p =>
        (p.name || '').toLowerCase().includes(q) ||
        (p.position || '').toLowerCase().includes(q) ||
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
  }, [prospects, posTab, search, sortBy]);

  // Position distribution — distinct groups actually present in the data.
  const dist = useMemo(() => {
    const counts = new Map();
    prospects.forEach(p => {
      const g = posGroup(p.position);
      counts.set(g, (counts.get(g) || 0) + 1);
    });
    const groups = [...counts.keys()].sort((a, b) => {
      const ai = GROUP_ORDER.indexOf(a), bi = GROUP_ORDER.indexOf(b);
      if (ai !== -1 || bi !== -1) return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
      return counts.get(b) - counts.get(a);
    });
    const max = Math.max(1, ...counts.values());
    return groups.map(g => ({
      label: g,
      count: counts.get(g),
      pct: Math.max(4, (counts.get(g) / max) * 100),
    }));
  }, [prospects]);

  const posTabs = useMemo(() => ['ALL', ...dist.map(d => d.label)], [dist]);

  // Scale for the translucent fill bar behind each row (relative grade).
  const [minProb, maxProb] = useMemo(() => {
    if (!filtered.length) return [0, 100];
    let lo = Infinity, hi = -Infinity;
    filtered.forEach(p => {
      const v = p.success_probability || 0;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    });
    return [lo, hi];
  }, [filtered]);

  const total   = filtered.length;
  const shown   = useMemo(() => filtered.slice(0, (page + 1) * PAGE_SIZE), [filtered, page]);
  const hasMore = shown.length < total;

  const handlePlayerClick = useCallback((p) => {
    navigate(`/predict?name=${encodeURIComponent(p.name)}`);
  }, [navigate]);

  const selectPos = useCallback((pos) => {
    setPosTab(prev => (prev === pos && pos !== 'ALL' ? 'ALL' : pos));
    setPage(0);
    setExpanded(null);
    setAnimKey(k => k + 1);
  }, []);

  const onSearch = useCallback((e) => {
    setSearch(e.target.value);
    setPage(0);
    setExpanded(null);
    setAnimKey(k => k + 1);
  }, []);

  const onSort = useCallback((key) => {
    setSortBy(key);
    setPage(0);
    setExpanded(null);
    setAnimKey(k => k + 1);
  }, []);

  const resetFilters = useCallback(() => {
    setPosTab('ALL'); setSearch(''); setSortBy('grade'); setPage(0); setExpanded(null);
    setAnimKey(k => k + 1);
  }, []);

  const sortHeaders = [
    { label: 'Rk',         cls: 'lb-c-rank',   sort: null },
    { label: 'Player',     cls: 'lb-c-name',   sort: 'name' },
    { label: 'Pos',        cls: 'lb-c-pos',    sort: null },
    { label: 'School',     cls: 'lb-c-school', sort: 'team' },
    { label: 'Projection', cls: 'lb-c-proj',   sort: null },
    { label: 'Grade',      cls: 'lb-c-grade',  sort: 'grade' },
    { label: 'Success',    cls: 'lb-c-prob',   sort: 'success' },
  ];

  return (
    <div className="lb-page">

      {/* ── Photo header ── */}
      <header className="lb-hero">
        <div className="lb-hero-media" aria-hidden="true">
          <img src={process.env.PUBLIC_URL + '/images/CFB Content/zion.webp'} alt="" />
        </div>
        <div className="lb-hero-scrim-x" aria-hidden="true" />
        <div className="lb-hero-scrim-y" aria-hidden="true" />
        <div className="lb-hero-inner">
          <div className="lb-hero-text">
            <div className="lb-eyebrow">Prospect leaderboard</div>
            <h1 className="lb-title">Every graded prospect</h1>
          </div>
          <div className="lb-count">
            {loading
              ? 'Loading…'
              : `${total.toLocaleString()} shown` + (meta ? ' · cached, sub-100ms' : '')}
          </div>
        </div>
      </header>

      <div className="lb-main">

        {/* ── Search + position filter ── */}
        <div className="lb-controls">
          <input
            className="input lb-search"
            type="search"
            value={search}
            onChange={onSearch}
            placeholder="Search player, position or school"
            aria-label="Search player, position or school"
          />
          <div className="lb-seg" role="group" aria-label="Filter by position">
            {posTabs.map(pos => (
              <button
                key={pos}
                type="button"
                className={`lb-seg-btn${posTab === pos ? ' active' : ''}`}
                aria-pressed={posTab === pos}
                onClick={() => selectPos(pos)}
              >
                {pos}
              </button>
            ))}
          </div>
        </div>

        {/* ── Position distribution ── */}
        {dist.length > 0 && (
          <div className="lb-dist">
            {dist.map((d, i) => (
              <button
                key={d.label}
                type="button"
                className={`lb-dist-cell${posTab === d.label ? ' active' : ''}`}
                onClick={() => selectPos(d.label)}
              >
                <span className="lb-dist-head">
                  <span className="lb-dist-label">{d.label}</span>
                  <span className="lb-dist-count">{d.count.toLocaleString()}</span>
                </span>
                <span className="lb-dist-track">
                  <span
                    className="lb-dist-bar"
                    style={{ width: `${d.pct}%`, animationDelay: `${(i * 0.05).toFixed(2)}s` }}
                  />
                </span>
              </button>
            ))}
          </div>
        )}

        {/* ── Board table ── */}
        <div className="lb-table">
          <div className="lb-thead">
            {sortHeaders.map(h => h.sort ? (
              <button
                key={h.label}
                type="button"
                className={`lb-th ${h.cls}${sortBy === h.sort ? ' active' : ''}`}
                onClick={() => onSort(h.sort)}
              >
                {h.label}
              </button>
            ) : (
              <div key={h.label} className={`lb-th-static ${h.cls}`}>{h.label}</div>
            ))}
            <div className="lb-c-chev" aria-hidden="true" />
          </div>

          {loading ? (
            <div className="lb-state">Loading prospects…</div>
          ) : error ? (
            <div className="lb-state">{error}</div>
          ) : shown.length === 0 ? (
            <div className="lb-state">
              {prospects.length === 0 ? (
                <>Cache empty — run <code>python build_prospect_cache.py</code> to populate.</>
              ) : (
                <>
                  No prospects match your filters.{' '}
                  <button type="button" className="lb-clear" onClick={resetFilters}>Clear filters</button>
                </>
              )}
            </div>
          ) : shown.map((p, i) => {
            const id   = `${p.name}-${i}`;
            const open = expanded === id;
            const top  = i < 3;
            const sp   = p.success_probability || 0;
            const fillPct = maxProb > minProb
              ? 6 + 94 * ((sp - minProb) / (maxProb - minProb))
              : 50;

            const minis = [];
            if (p.production_score != null)
              minis.push({ label: 'Production score', value: `${Number(p.production_score).toFixed(1)} / 100`, pct: clampPct(p.production_score) });
            if (p.combine_speed_score != null)
              minis.push({ label: 'Combine speed score', value: `${Number(p.combine_speed_score).toFixed(1)} / 100`, pct: clampPct(p.combine_speed_score) });
            if (p.success_probability != null)
              minis.push({ label: 'Success probability', value: `${sp.toFixed(1)}%`, pct: clampPct(sp) });

            return (
              <div className="lb-rowwrap" key={id}>
                <div
                  className={`lb-row${top ? ' lb-row-top' : ''}`}
                  role="button"
                  tabIndex={0}
                  aria-expanded={open}
                  onClick={() => setExpanded(open ? null : id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      setExpanded(open ? null : id);
                    }
                  }}
                >
                  <span
                    className="lb-fill"
                    key={`fill-${animKey}`}
                    aria-hidden="true"
                    style={{ width: `${fillPct.toFixed(1)}%`, animationDelay: `${(i % PAGE_SIZE * 0.025).toFixed(3)}s` }}
                  />
                  <span className="lb-rank lb-c-rank">{i + 1}</span>
                  <span className="lb-name lb-c-name">
                    <span>{p.name}</span>
                    {p.data_source && p.data_source !== 'espn_live' && (
                      <span className="lb-est" title="No verified season stats — profile is estimated">EST</span>
                    )}
                  </span>
                  <span className="lb-pos lb-c-pos">{(p.position || '?').toUpperCase()}</span>
                  <span className="lb-school lb-c-school">{p.team || '—'}</span>
                  <span className="lb-proj lb-c-proj">{DRAFT_SHORT[p.draft_grade] || p.draft_grade || '—'}</span>
                  <span className="lb-grade lb-c-grade"><span>{p.grade || '—'}</span></span>
                  <span className="lb-prob lb-c-prob">{p.success_probability != null ? sp.toFixed(1) : '—'}</span>
                  <span className={`lb-chev lb-c-chev${open ? ' open' : ''}`} aria-hidden="true">›</span>
                </div>

                <div className={`lb-panel${open ? ' open' : ''}`} aria-hidden={!open}>
                  <div className="lb-panel-inner">
                    {minis.length > 0 && (
                      <div className="lb-minis">
                        {minis.map(m => (
                          <div className="lb-mini" key={m.label}>
                            <div className="lb-mini-head">
                              <span>{m.label}</span>
                              <span className="lb-mini-val">{m.value}</span>
                            </div>
                            <div className="lb-mini-track">
                              <span className="lb-mini-bar" style={{ width: `${m.pct}%` }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    <button
                      type="button"
                      className="lb-report"
                      tabIndex={open ? 0 : -1}
                      onClick={(e) => { e.stopPropagation(); handlePlayerClick(p); }}
                    >
                      Full scouting report →
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── Load more ── */}
        {hasMore && (
          <div className="lb-more">
            <button type="button" className="btn btn-primary" onClick={() => setPage(pg => pg + 1)}>
              Load more ({(total - shown.length).toLocaleString()} remaining)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

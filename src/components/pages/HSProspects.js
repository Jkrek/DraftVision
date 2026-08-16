import React, { useState, useEffect, useMemo, useCallback } from 'react';
import './HSProspects.css';

const POSITION_TABS = ['ALL', 'QB', 'RB', 'WR', 'TE', 'DB', 'LB', 'DL', 'OL'];
const STARS_FILTERS = ['ALL', '5', '4', '3'];
const PAGE_SIZE     = 100;

const POS_GROUP_MAP = {
  DB: new Set(['CB', 'S', 'DB', 'FS', 'SS']),
  LB: new Set(['LB', 'ILB', 'OLB', 'MLB']),
  DL: new Set(['DL', 'DE', 'DT', 'EDGE', 'NT']),
  OL: new Set(['OL', 'OT', 'OG', 'C']),
};

function posMatchesTab(pos, tab) {
  if (tab === 'ALL') return true;
  const p = (pos || '').toUpperCase();
  const group = POS_GROUP_MAP[tab];
  return group ? group.has(p) : p === tab;
}

function StarPill({ stars }) {
  const s = parseInt(stars) || 0;
  return (
    <span className="hsp-star-pill" aria-label={`${s} star rating`}>
      <span className="hsp-star-filled">{'★'.repeat(s)}</span>
      <span className="hsp-star-empty">{'★'.repeat(Math.max(0, 5 - s))}</span>
    </span>
  );
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
    <div className="hsp-page">
      <div className="hsp-inner">

        {/* Header */}
        <header className="hsp-header">
          <div className="hsp-heading">
            <div className="hsp-eyebrow">High school pipeline</div>
            <h1 className="hsp-title">HS Prospects</h1>
          </div>
          <div className="hsp-count">
            {loading
              ? 'Loading…'
              : meta
                ? `${(meta.total || prospects.length).toLocaleString()} prospects${meta.generated_at ? ` · updated ${new Date(meta.generated_at).toLocaleDateString()}` : ''}`
                : `${prospects.length.toLocaleString()} prospects`}
          </div>
        </header>

        {!meta && !loading && (
          <p className="hsp-note">
            Cache empty — run <code>python build_hs_prospect_cache.py</code> to populate
          </p>
        )}
        {apiKeyMissing && (
          <p className="hsp-note">
            Set <code>CFBD_API_KEY</code> env var (free at collegefootballdata.com) to fetch live data
          </p>
        )}
        {error && <p className="hsp-error">{error}</p>}

        {/* Filter bar */}
        <div className="hsp-filters">
          <input
            className="hsp-search"
            value={search}
            onChange={e => { setSearch(e.target.value); setPage(0); }}
            placeholder="Search name, school, state or commit"
            aria-label="Search prospects"
          />

          {/* Position tabs */}
          <div className="hsp-seg" role="group" aria-label="Filter by position">
            {POSITION_TABS.map(pos => (
              <button
                key={pos}
                onClick={() => { setPosTab(pos); setPage(0); }}
                className={`hsp-seg-btn${posTab === pos ? ' is-active' : ''}`}
              >
                {pos}
              </button>
            ))}
          </div>

          {/* Stars filter */}
          <div className="hsp-seg" role="group" aria-label="Filter by star rating">
            {STARS_FILTERS.map(s => (
              <button
                key={s}
                onClick={() => { setStarsFilter(s); setPage(0); }}
                className={`hsp-seg-btn${starsFilter === s ? ' is-active' : ''}`}
              >
                {s === 'ALL' ? 'All stars' : `${s}★`}
              </button>
            ))}
          </div>

          {/* Year filter */}
          {availableYears.length > 1 && (
            <select
              className="hsp-select"
              value={yearFilter}
              onChange={e => { setYearFilter(e.target.value); setPage(0); }}
              aria-label="Filter by class year"
            >
              <option value="ALL">All classes</option>
              {availableYears.map(y => <option key={y} value={String(y)}>Class of {y}</option>)}
            </select>
          )}

          {/* Sort */}
          <select
            className="hsp-select"
            value={sortBy}
            onChange={e => setSortBy(e.target.value)}
            aria-label="Sort prospects"
          >
            <option value="rank">Sort: National rank</option>
            <option value="stars">Sort: Stars</option>
            <option value="rating">Sort: Rating</option>
            <option value="name">Sort: Name</option>
          </select>
        </div>

        {/* Result count */}
        <div className="hsp-resultbar">
          <p className="hsp-result-count">
            {loading ? 'Loading…' : `${total.toLocaleString()} prospect${total !== 1 ? 's' : ''}`}
          </p>
          {hasActiveFilter && (
            <button className="hsp-clear" onClick={resetFilters}>Clear filters</button>
          )}
        </div>

        {/* Table */}
        <div className="hsp-table">
          <div className="hsp-thead" aria-hidden="true">
            <div className="hsp-col-rank">Rk</div>
            <div className="hsp-col-name">Player</div>
            <div className="hsp-col-school">HS / City</div>
            <div className="hsp-col-pos">Pos</div>
            <div className="hsp-col-stars">Stars</div>
            <div className="hsp-col-year">Yr</div>
            <div className="hsp-col-commit">Commit</div>
            <div className="hsp-col-rating">Rating</div>
          </div>

          {loading ? (
            <div className="hsp-empty">Loading prospects…</div>
          ) : shown.length === 0 ? (
            <div className="hsp-empty">
              {prospects.length === 0
                ? 'No HS prospects cached — run build_hs_prospect_cache.py to populate'
                : 'No prospects match your filters'}
            </div>
          ) : shown.map((p, i) => (
            <div key={`${p.name}-${i}`} className="hsp-row">
              <div className={`hsp-col-rank${p.ranking <= 25 ? ' is-top' : ''}`}>
                #{p.ranking || '—'}
              </div>

              <div className="hsp-col-name">{p.name}</div>

              <div className="hsp-col-school">
                <div className="hsp-school">{p.school || '—'}</div>
                {(p.city || p.state) && (
                  <div className="hsp-school-loc">
                    {[p.city, p.state].filter(Boolean).join(', ')}
                  </div>
                )}
              </div>

              <div className="hsp-col-pos">{(p.position || '?').toUpperCase()}</div>

              <div className="hsp-col-stars"><StarPill stars={p.stars} /></div>

              <div className="hsp-col-year">{p.year || '—'}</div>

              <div className={`hsp-col-commit${p.committed_to ? ' is-committed' : ''}`}>
                {p.committed_to || 'Uncommitted'}
              </div>

              <div className={`hsp-col-rating${p.rating >= 0.99 ? ' is-elite' : ''}`}>
                {p.rating ? p.rating.toFixed(4) : '—'}
              </div>
            </div>
          ))}
        </div>

        {/* Load more */}
        {hasMore && (
          <div className="hsp-more">
            <button className="btn btn-primary" onClick={() => setPage(pg => pg + 1)}>
              Load more ({(total - shown.length).toLocaleString()} remaining)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

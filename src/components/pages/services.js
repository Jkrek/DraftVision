import React, { useEffect, useMemo, useState } from 'react';
import './services.css';

const PAGE_SIZE = 100;

function formatProb(value) {
  const n = typeof value === 'number' ? value : parseFloat(value);
  if (isNaN(n)) return null;
  return n.toFixed(1);
}

// "2026-08-14T03:12:00Z" -> "Aug 14"; null on anything unparseable.
function formatBoardDate(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export default function Services() {
  const [prospects, setProspects] = useState([]);
  const [meta, setMeta] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  useEffect(() => {
    fetch('/api/prospects?limit=5000')
      .then(r => r.json())
      .then(d => {
        setProspects(Array.isArray(d.prospects) ? d.prospects : []);
        setMeta(d.meta || null);
      })
      .catch(() => {
        setProspects([]);
        setError('Could not load prospects. Is the prediction server running?');
      })
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return prospects;
    return prospects.filter(p =>
      (p.name || '').toLowerCase().includes(q) ||
      (p.team || '').toLowerCase().includes(q)
    );
  }, [prospects, search]);

  const visible = filtered.slice(0, visibleCount);
  const remaining = filtered.length - visible.length;
  const boardDate = formatBoardDate(meta && meta.generated_at);

  const handleSearch = (e) => {
    setSearch(e.target.value);
    setVisibleCount(PAGE_SIZE);
  };

  return (
    <div className="stars-page">
      <header className="stars-head">
        <div className="eyebrow">Prospect leaderboard</div>
        <h1 className="stars-title">College Stars</h1>
        <p className="stars-count">
          {loading
            ? 'Loading graded prospects…'
            : `${filtered.length.toLocaleString()} of ${prospects.length.toLocaleString()} graded prospects` +
              (boardDate ? ` · Board updated ${boardDate}` : '')}
        </p>
      </header>

      <div className="stars-controls">
        <input
          type="text"
          className="stars-search"
          placeholder="Search by name or school…"
          value={search}
          onChange={handleSearch}
          aria-label="Search prospects by name or school"
        />
      </div>

      <div className="stars-board">
        <div className="stars-board-head" aria-hidden="true">
          <div className="col-rank">Rk</div>
          <div className="col-player">Player</div>
          <div className="col-pos">Pos</div>
          <div className="col-school">School</div>
          <div className="col-proj">Projection</div>
          <div className="col-grade">Grade</div>
          <div className="col-prob">Success</div>
        </div>

        {loading && (
          <div className="stars-state">Loading college prospects…</div>
        )}

        {!loading && error && (
          <div className="stars-state">{error}</div>
        )}

        {!loading && !error && visible.length === 0 && (
          <div className="stars-state">No prospects match your search.</div>
        )}

        {!loading && visible.map((p, i) => {
          const rank = i + 1;
          const prob = formatProb(p.success_probability);
          return (
            <div
              key={`${p.name}-${p.team}-${i}`}
              className={`stars-row${rank <= 3 && !search ? ' stars-row--top' : ''}`}
            >
              <div className="col-rank">{rank}</div>
              <div className="col-player">
                {p.espn_team_id && (
                  <img
                    className="col-team-logo"
                    src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${p.espn_team_id}.png`}
                    alt=""
                    loading="lazy"
                    onError={e => { e.target.style.display = 'none'; }}
                  />
                )}
                {p.name}
                {p.data_source && p.data_source !== 'espn_live' && (
                  <span className="col-est" title="No verified season stats — profile is estimated">EST</span>
                )}
              </div>
              <div className="col-pos">{p.position || '—'}</div>
              <div className="col-school">{p.team || '—'}</div>
              <div className="col-proj">{p.draft_grade || '—'}</div>
              <div className="col-grade">
                <span className="grade-pill">{p.grade || '—'}</span>
              </div>
              <div className="col-prob">
                {prob === null ? '—' : (<>{prob}<span className="col-prob-unit">%</span></>)}
              </div>
            </div>
          );
        })}
      </div>

      {!loading && remaining > 0 && (
        <div className="stars-more">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => setVisibleCount(c => c + PAGE_SIZE)}
          >
            Load {Math.min(PAGE_SIZE, remaining)} more · {remaining.toLocaleString()} remaining
          </button>
        </div>
      )}
    </div>
  );
}

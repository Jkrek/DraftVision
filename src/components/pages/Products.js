import React, { useEffect, useState } from 'react';
import '../../App.css';

const POSITION_COLORS = {
  QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b',
  TE: '#a78bfa', default: '#94a3b8',
};
function posColor(p) { return POSITION_COLORS[p] || POSITION_COLORS.default; }

export default function Products() {
  const [prospects, setProspects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch('/api/prospects?limit=5000')
      .then(r => r.json())
      .then(d => setProspects(Array.isArray(d.prospects) ? d.prospects : []))
      .catch(() => setProspects([]))
      .finally(() => setLoading(false));
  }, []);

  const filtered = prospects.filter(p =>
    !search || p.name.toLowerCase().includes(search.toLowerCase()) || (p.team || '').toLowerCase().includes(search.toLowerCase())
  );

  const backgroundStyle = {
    backgroundImage: `linear-gradient(120deg, rgba(11,23,42,0.92), rgba(15,23,42,0.98)), url(${process.env.PUBLIC_URL}/images/ctjspicture.png)`,
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    minHeight: '100vh',
    paddingTop: '80px',
    paddingBottom: '60px',
  };

  const glassCard = {
    background: 'rgba(30,41,59,0.8)',
    border: '1px solid rgba(255,255,255,0.08)',
    borderRadius: '10px',
  };

  return (
    <div style={backgroundStyle}>
      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 20px' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h1 style={{ color: '#f1f5f9', fontSize: '2.2rem', fontWeight: 800, margin: 0 }}>College Stars</h1>
          <p style={{ color: '#94a3b8', marginTop: '0.5rem' }}>
            {prospects.length} synced prospects · Browse and discover rising talent
          </p>
        </div>

        <div style={{ ...glassCard, padding: '1rem 1.25rem', marginBottom: '1.5rem' }}>
          <input
            type="text"
            placeholder="Search by name or school…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{
              width: '100%', padding: '10px 14px', borderRadius: '8px',
              border: '1px solid rgba(255,255,255,0.12)', background: 'rgba(15,23,42,0.7)',
              color: '#e2e8f0', fontSize: '14px', outline: 'none', boxSizing: 'border-box',
            }}
          />
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '4rem' }}>Loading college prospects…</div>
        ) : (
          <>
            <p style={{ color: '#64748b', fontSize: '13px', marginBottom: '0.75rem' }}>
              Showing {filtered.length} of {prospects.length} prospects
            </p>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: '0.65rem',
            }}>
              {filtered.map(p => {
                const pc = posColor(p.position);
                return (
                  <div key={p.name} style={{ ...glassCard, padding: '0.85rem 1rem', display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                    <div style={{
                      minWidth: '36px', height: '36px', borderRadius: '50%',
                      background: `${pc}22`, border: `2px solid ${pc}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '10px', fontWeight: 700, color: pc,
                    }}>
                      {p.position || '?'}
                    </div>
                    <div style={{ overflow: 'hidden' }}>
                      <p style={{ color: '#e2e8f0', fontWeight: 600, fontSize: '0.85rem', margin: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.name}</p>
                      <p style={{ color: '#64748b', fontSize: '0.75rem', margin: '2px 0 0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.team}</p>
                    </div>
                  </div>
                );
              })}
            </div>
            {filtered.length === 0 && (
              <div style={{ textAlign: 'center', color: '#64748b', padding: '3rem' }}>No prospects match your search.</div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

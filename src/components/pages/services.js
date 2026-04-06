import React, { useState, useCallback, useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import './services.css';

const API_TIMEOUT = 14000;

function SuccessCellRenderer({ value }) {
  const isSuccess = value === 'Success';
  return (
    <span style={{
      padding: '2px 10px', borderRadius: '999px', fontWeight: 700, fontSize: '12px',
      background: isSuccess ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
      color: isSuccess ? '#22c55e' : '#ef4444',
      border: `1px solid ${isSuccess ? '#22c55e55' : '#ef444455'}`,
    }}>
      {isSuccess ? '✓ Success' : '✗ Unlikely'}
    </span>
  );
}

function ProbCellRenderer({ value }) {
  const pct = typeof value === 'number' ? value : parseFloat(value);
  if (isNaN(pct)) return <span style={{ color: '#64748b' }}>—</span>;
  const color = pct >= 50 ? '#22c55e' : '#ef4444';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{ flex: 1, background: 'rgba(255,255,255,0.08)', borderRadius: '999px', height: '6px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: '999px' }} />
      </div>
      <span style={{ color, fontWeight: 700, minWidth: '40px', textAlign: 'right', fontSize: '13px' }}>{pct}%</span>
    </div>
  );
}

export default function Services() {
  const [playerName, setPlayerName] = useState('');
  const [rowData, setRowData]       = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState('');

  const backgroundStyle = {
    backgroundImage: `linear-gradient(135deg, rgba(11,19,43,0.88), rgba(15,23,42,0.97)), url(${process.env.PUBLIC_URL}/images/dantemoore.png)`,
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    minHeight: '100vh',
    paddingTop: '80px',
    paddingBottom: '60px',
  };

  const glassCard = {
    background: 'rgba(30,41,59,0.85)',
    border: '1px solid rgba(255,255,255,0.08)',
    borderRadius: '12px',
    backdropFilter: 'blur(12px)',
  };

  const columnDefs = useMemo(() => [
    { headerName: 'Player', field: 'name', sortable: true, filter: true, minWidth: 160,
      cellStyle: { color: '#e2e8f0', fontWeight: 600 } },
    { headerName: 'Position', field: 'position', sortable: true, filter: true, width: 110,
      cellStyle: { color: '#94a3b8' } },
    { headerName: 'Draft Proj.', field: 'draft_projection', sortable: true, filter: true, width: 130,
      cellStyle: { color: '#f59e0b', fontWeight: 600 } },
    { headerName: 'College Tier', field: 'college_tier', sortable: true, filter: true, width: 130,
      cellStyle: { color: '#94a3b8' } },
    { headerName: 'Prod. Score', field: 'production_score', sortable: true, filter: true, width: 120,
      cellStyle: { color: '#a78bfa', fontWeight: 600 } },
    { headerName: 'Success Prob.', field: 'success_probability', sortable: true, filter: true, width: 200,
      cellRenderer: ProbCellRenderer },
    { headerName: 'Verdict', field: 'success', sortable: true, filter: true, width: 140,
      cellRenderer: SuccessCellRenderer },
  ], []);

  const defaultColDef = useMemo(() => ({
    resizable: true,
    flex: 1,
    minWidth: 100,
  }), []);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    const name = playerName.trim();
    if (!name) return;
    if (rowData.some(r => r.name.toLowerCase() === name.toLowerCase())) {
      setError(`${name} is already in the comparison table.`);
      return;
    }
    setError('');
    setLoading(true);
    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
        signal: AbortSignal.timeout(API_TIMEOUT),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed.');

      setRowData(prev => [...prev, {
        name: data.resolved_name || name,
        position: data.predicted_position || data.stats?.position || 'Unknown',
        draft_projection: data.summary?.draft_projection || '—',
        college_tier: data.summary?.college_tier || '—',
        production_score: data.summary?.production_score || '—',
        success_probability: data.success_probability ?? null,
        success: data.success,
      }]);
      setPlayerName('');
    } catch (err) {
      setError(err.message || 'Could not reach the prediction server.');
    } finally {
      setLoading(false);
    }
  }, [playerName, rowData]);

  return (
    <div style={backgroundStyle}>
      <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 20px' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h1 style={{ color: '#f1f5f9', fontSize: '2.2rem', fontWeight: 800, margin: 0 }}>Player Comparison</h1>
          <p style={{ color: '#94a3b8', marginTop: '0.5rem' }}>
            Add multiple prospects to compare their ML predictions side-by-side
          </p>
        </div>

        {/* Input form */}
        <div style={{ ...glassCard, padding: '1.25rem 1.5rem', marginBottom: '1.5rem' }}>
          <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
            <input
              type="text"
              value={playerName}
              onChange={(e) => { setPlayerName(e.target.value); setError(''); }}
              placeholder="Enter a prospect name (e.g. Arch Manning)"
              style={{
                flex: 1, minWidth: '220px', padding: '10px 14px', borderRadius: '8px',
                border: `1px solid ${error ? '#ef4444' : 'rgba(255,255,255,0.12)'}`,
                background: 'rgba(15,23,42,0.7)', color: '#e2e8f0', fontSize: '14px', outline: 'none',
              }}
            />
            <button
              type="submit"
              disabled={loading || !playerName.trim()}
              style={{
                padding: '10px 24px', borderRadius: '8px', fontWeight: 700, fontSize: '14px',
                cursor: loading ? 'not-allowed' : 'pointer',
                background: loading ? 'rgba(59,130,246,0.4)' : '#3b82f6',
                border: 'none', color: '#fff', whiteSpace: 'nowrap',
              }}
            >
              {loading ? 'Analyzing…' : '+ Add to Comparison'}
            </button>
            {rowData.length > 0 && (
              <button
                type="button"
                onClick={() => setRowData([])}
                style={{ padding: '10px 16px', borderRadius: '8px', background: 'transparent', border: '1px solid rgba(239,68,68,0.4)', color: '#ef4444', cursor: 'pointer', fontSize: '13px' }}
              >
                Clear All
              </button>
            )}
          </form>
          {error && <p style={{ color: '#ef4444', fontSize: '13px', margin: '0.5rem 0 0' }}>{error}</p>}
        </div>

        {/* Grid */}
        <div style={{ ...glassCard, overflow: 'hidden' }}>
          <div className="ag-theme-alpine-dark" style={{ height: rowData.length > 0 ? Math.min(420, 80 + rowData.length * 50) : 200, width: '100%' }}>
            <AgGridReact
              rowData={rowData}
              columnDefs={columnDefs}
              defaultColDef={defaultColDef}
              noRowsOverlayComponent={() => (
                <div style={{ color: '#64748b', textAlign: 'center', padding: '2rem' }}>
                  Add a prospect above to start comparing
                </div>
              )}
            />
          </div>
        </div>

        <p style={{ color: '#475569', fontSize: '12px', textAlign: 'center', marginTop: '1rem' }}>
          Predictions use the DraftVision XGBoost ML classifier · Click column headers to sort
        </p>
      </div>
    </div>
  );
}

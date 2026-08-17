import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { anonFetch } from '../../lib/api';
import '../../App.css';
import './MockDraft.css';
import { nflLogoUrl } from '../nflTeams';

const ROUND_PICKS = 32; // picks per round

function getRound(pickNum) {
  return Math.ceil(pickNum / ROUND_PICKS);
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
    anonFetch('/api/mock-draft')
      .then(r => r.json())
      .then(data => {
        setPicks(Array.isArray(data.picks) ? data.picks : []);
        setTitle(data.title || '');
        setMeta(data.generated_at ? { generated_at: data.generated_at, total: data.total } : null);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const uploadImage = useCallback(async (file) => {
    setUploading(true);
    setError(null);
    try {
      // Read file as base64
      const b64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => {
          // e.target.result is "data:image/png;base64,XXXX" — strip the prefix
          const [, data] = e.target.result.split(',');
          resolve(data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      const mediaType = file.type || 'image/png';
      const res  = await anonFetch('/api/mock-draft/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_b64: b64, media_type: mediaType, title: 'JKrek\'s Mock Draft' }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');

      // Reload the draft
      const fresh = await anonFetch('/api/mock-draft').then(r => r.json());
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
    uploadImage(file);
  }, [uploadImage]);

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
    <div className="mock-page">
      <div className="mock-inner">

        {/* Header — flush left */}
        <header className="mock-header">
          <div className="eyebrow">Mock draft</div>
          <h1 className="mock-title">{title || 'JKrek\'s Mock Draft'}</h1>
          <p className="mock-sub">Official pick-by-pick NFL Draft board</p>
          {meta && (
            <p className="mock-meta">
              {meta.total} picks
              {meta.generated_at && ` · updated ${new Date(meta.generated_at).toLocaleDateString()}`}
            </p>
          )}
          {error && <p className="mock-error">{error}</p>}
        </header>

        {/* Upload dropzone — always visible at top */}
        <div
          className={`mock-dropzone${dragOver ? ' is-dragover' : ''}`}
          role="button"
          tabIndex={0}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileRef.current?.click(); } }}
        >
          <input
            ref={fileRef}
            className="mock-file-input"
            type="file"
            accept="image/png,image/jpeg,image/webp,image/*"
            onChange={e => handleFile(e.target.files[0])}
          />
          {uploading ? (
            <p className="mock-status"><span className="mock-dot" />Claude vision is parsing the picks</p>
          ) : (
            <>
              <p className="mock-drop-label">
                {picks.length > 0 ? 'Update mock draft' : 'Upload mock draft'}
              </p>
              <p className="mock-drop-hint">
                Drop a mock draft PNG — Claude vision parses the picks
              </p>
            </>
          )}
        </div>

        {/* Empty state */}
        {isEmpty && (
          <div className="mock-empty">
            <p>
              No mock draft loaded yet. Export a PNG of your board from the PFF
              Mock Draft Simulator and drop it above.
            </p>
          </div>
        )}

        {loading && (
          <div className="mock-loading">
            <p className="mock-status"><span className="mock-dot" />Loading board</p>
          </div>
        )}

        {/* Draft board — grouped by round */}
        {roundKeys.map(rndKey => (
          <section key={rndKey} className="mock-round">
            {/* Round divider */}
            <div className="mock-round-head">
              <span className="mock-round-label">
                {isNaN(parseInt(rndKey)) ? rndKey : `Round ${rndKey}`}
              </span>
              <span className="mock-round-rule" />
              <span className="mock-round-count">{rounds[rndKey].length} picks</span>
            </div>

            {/* Pick rows */}
            <div className="mock-board">
              {rounds[rndKey].map((pick, i) => (
                <div key={`${pick.pick}-${i}`} className="mock-row">

                  {/* Pick number */}
                  <span className="mock-row-pick">{pick.pick}</span>

                  {/* Player name + school */}
                  <div className="mock-row-player">
                    <div
                      className={`mock-row-name${pick.player ? ' is-link' : ''}`}
                      onClick={() => pick.player && navigate(`/predict?name=${encodeURIComponent(pick.player)}`)}
                    >
                      {pick.player || '—'}
                    </div>
                    {pick.school && (
                      <div className="mock-row-school">{pick.school}</div>
                    )}
                  </div>

                  {/* NFL team */}
                  <div className="mock-row-team">
                    {nflLogoUrl(pick.nfl_team) && (
                      <img
                        className="mock-row-logo"
                        src={nflLogoUrl(pick.nfl_team, 500)}
                        alt=""
                        loading="lazy"
                        onError={e => { e.target.style.display = 'none'; }}
                      />
                    )}
                    <span>{pick.nfl_team || '—'}</span>
                  </div>

                  {/* Position */}
                  {pick.position
                    ? <span className="mock-row-pos">{pick.position}</span>
                    : <span />}

                  {/* PFF grade */}
                  {pick.pff_grade
                    ? <span className="mock-row-grade"><span>{pick.pff_grade}</span></span>
                    : <span />}

                  {/* Predict button */}
                  {pick.player ? (
                    <button
                      className="mock-row-btn"
                      onClick={() => navigate(`/predict?name=${encodeURIComponent(pick.player)}`)}
                    >
                      Predict
                    </button>
                  ) : <span />}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

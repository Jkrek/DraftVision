import React, { useState, useRef, useCallback } from 'react';
import { anonFetch } from '../../../lib/api';
import MockRow from './MockRow';

const ROUND_PICKS = 32;

function getRound(pickNum) {
  return Math.ceil(pickNum / ROUND_PICKS);
}

/* The original PNG-upload mock-draft flow, folded into a collapsed
   <details> section. Drop a mock-draft image → POST /api/mock-draft/upload
   (Claude vision parses the picks server-side) → render the stored board.
   The stored draft is fetched lazily, the first time the section opens. */
export default function ImportMock() {
  const fileRef = useRef(null);

  const [loaded, setLoaded]       = useState(false);
  const [picks, setPicks]         = useState([]);
  const [title, setTitle]         = useState('');
  const [meta, setMeta]           = useState(null);
  const [loading, setLoading]     = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError]         = useState(null);
  const [dragOver, setDragOver]   = useState(false);

  const loadDraft = useCallback(async () => {
    setLoading(true);
    try {
      const data = await anonFetch('/api/mock-draft').then((r) => r.json());
      setPicks(Array.isArray(data.picks) ? data.picks : []);
      setTitle(data.title || '');
      setMeta(data.generated_at ? { generated_at: data.generated_at, total: data.total } : null);
    } catch {
      /* stored board unavailable — the dropzone still works */
    } finally {
      setLoading(false);
    }
  }, []);

  const handleToggle = useCallback((e) => {
    if (e.target.open && !loaded) {
      setLoaded(true);
      loadDraft();
    }
  }, [loaded, loadDraft]);

  const uploadImage = useCallback(async (file) => {
    setUploading(true);
    setError(null);
    try {
      // Read file as base64
      const b64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
          // e.target.result is "data:image/png;base64,XXXX" — strip the prefix
          const [, data] = e.target.result.split(',');
          resolve(data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      const mediaType = file.type || 'image/png';
      const res = await anonFetch('/api/mock-draft/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_b64: b64, media_type: mediaType, title: "JKrek's Mock Draft" }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');
      await loadDraft();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }, [loadDraft]);

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

  return (
    <details className="mock-import" onToggle={handleToggle}>
      <summary className="mock-import-summary">
        <span className="mock-import-chev">›</span>
        Import a mock from an image
        <span className="mock-import-hint">PNG → Claude vision parses the picks</span>
      </summary>

      <div className="mock-import-body">
        <div
          className={`mock-dropzone${dragOver ? ' is-dragover' : ''}`}
          role="button"
          tabIndex={0}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current && fileRef.current.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              if (fileRef.current) fileRef.current.click();
            }
          }}
        >
          <input
            ref={fileRef}
            className="mock-file-input"
            type="file"
            accept="image/png,image/jpeg,image/webp,image/*"
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {uploading ? (
            <p className="mock-status"><span className="mock-dot" />Claude vision is parsing the picks</p>
          ) : (
            <>
              <p className="mock-drop-label">
                {picks.length > 0 ? 'Update imported mock' : 'Upload a mock draft image'}
              </p>
              <p className="mock-drop-hint">
                Drop a mock draft PNG — e.g. an exported PFF Mock Draft Simulator board
              </p>
            </>
          )}
        </div>

        {error && <p className="mock-error">{error}</p>}

        {loading && (
          <p className="mock-status"><span className="mock-dot" />Loading imported board</p>
        )}

        {picks.length > 0 && (
          <div className="mock-import-board">
            <div className="mock-import-title">
              {title || 'Imported mock'}
              {meta && meta.generated_at && (
                <span className="mock-import-meta">
                  {' '}· {meta.total} picks · updated {new Date(meta.generated_at).toLocaleDateString()}
                </span>
              )}
            </div>

            {roundKeys.map((rndKey) => (
              <section key={rndKey} className="mock-round">
                <div className="mock-round-head">
                  <span className="mock-round-label">
                    {isNaN(parseInt(rndKey)) ? rndKey : `Round ${rndKey}`}
                  </span>
                  <span className="mock-round-rule" />
                  <span className="mock-round-count">{rounds[rndKey].length} picks</span>
                </div>
                <div className="mock-board">
                  {rounds[rndKey].map((pick, i) => (
                    <MockRow
                      key={`${pick.pick}-${i}`}
                      pick={pick.pick}
                      name={pick.player}
                      school={pick.school}
                      nflTeam={pick.nfl_team}
                      position={pick.position}
                      grade={pick.pff_grade}
                    />
                  ))}
                </div>
              </section>
            ))}
          </div>
        )}
      </div>
    </details>
  );
}

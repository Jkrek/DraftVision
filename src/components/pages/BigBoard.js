import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { anonFetch } from '../../lib/api';
import './BigBoard.css';

// Owner-curated big board per draft class. Reads GET /api/big-board?class=YYYY
// ({class, board, rest, missing, updated_at}); when that endpoint is absent
// (backend not deployed yet) it degrades to the model ranking from
// /api/prospects?draft_class=YYYY. Owner editing POSTs {class, board:["Name|Team"]}
// with the analytics key (same sessionStorage slot as /insights).

const KEY_STORE = 'dv_analytics_key';
const CLASSES = ['2027', '2028', '2029', '2030'];
const PAGE_SIZE = 100;

const GRADE_ORDER = { 'A+': 0, 'A': 1, 'A-': 2, 'B+': 3, 'B': 4, 'B-': 5, 'C+': 6, 'C': 7, 'C-': 8, 'D': 9 };

const keyOf = (p) => `${p.name}|${p.team || ''}`;
const mirrorKey = (year) => `dv_big_board_${year}`;

// Same default ordering the leaderboard uses — "model ranked".
function modelSort(list) {
  return [...list].sort((a, b) =>
    (GRADE_ORDER[a.grade] ?? 9) - (GRADE_ORDER[b.grade] ?? 9) ||
    (b.success_probability || 0) - (a.success_probability || 0),
  );
}

// "2026-08-18T03:12:00Z" -> "Aug 18"; null on anything unparseable.
function formatDate(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

// Last-saved board mirrored to localStorage (covers post-redeploy resets).
function readMirror(year) {
  try {
    const raw = localStorage.getItem(mirrorKey(year));
    const arr = raw ? JSON.parse(raw) : null;
    return Array.isArray(arr) && arr.length ? arr : null;
  } catch {
    return null;
  }
}

function writeMirror(year, keys) {
  try {
    localStorage.setItem(mirrorKey(year), JSON.stringify(keys));
  } catch {
    /* private mode etc. — mirror is best-effort */
  }
}

function BoardRow({ p, rank, curated, onOpen }) {
  const sp = p.success_probability;
  return (
    <div
      className={`bb-row${curated ? ' bb-curated' : ''}`}
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen(); }
      }}
    >
      {curated && <span className="bb-curated-bar" aria-hidden="true" />}
      <span className="bb-rank bb-c-rank">{rank}</span>
      <span className="bb-name bb-c-name">
        {p.espn_team_id && (
          <img
            className="bb-team-logo"
            src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${p.espn_team_id}.png`}
            alt=""
            loading="lazy"
            onError={(e) => { e.target.style.display = 'none'; }}
          />
        )}
        <span>{p.name}</span>
        {p.data_source && p.data_source !== 'espn_live' && (
          <span className="bb-est" title="No verified season stats — profile is estimated">EST</span>
        )}
      </span>
      <span className="bb-pos bb-c-pos">{(p.position || '?').toUpperCase()}</span>
      <span className="bb-school bb-c-school">{p.team || '—'}</span>
      <span className="bb-grade bb-c-grade"><span>{p.grade || '—'}</span></span>
      <span className="bb-prob bb-c-prob">{sp != null ? Number(sp).toFixed(1) : '—'}</span>
    </div>
  );
}

export default function BigBoard() {
  const navigate = useNavigate();

  const [classTab, setClassTab] = useState(CLASSES[0]);
  const [data, setData] = useState(null); // {board, rest, missing, updatedAt, fallback}
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reload, setReload] = useState(0);
  const [page, setPage] = useState(0);

  // Editor state — draft === null means "not editing".
  const [draft, setDraft] = useState(null);
  const [search, setSearch] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [keyPrompt, setKeyPrompt] = useState({ open: false, restore: false });
  const [keyInput, setKeyInput] = useState('');

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    setPage(0);
    (async () => {
      try {
        const r = await anonFetch(`/api/big-board?class=${classTab}`);
        if (!r.ok) throw new Error('no-board-api');
        const d = await r.json();
        if (!alive) return;
        setData({
          board: Array.isArray(d.board) ? d.board : [],
          rest: Array.isArray(d.rest) ? d.rest : [],
          missing: Array.isArray(d.missing) ? d.missing : [],
          updatedAt: d.updated_at || null,
          fallback: false,
        });
      } catch {
        // Board API unavailable — degrade to the model ranking for the class.
        try {
          const r2 = await anonFetch(`/api/prospects?draft_class=${classTab}&limit=2000`);
          if (!r2.ok) throw new Error('http');
          const d2 = await r2.json();
          if (!alive) return;
          setData({
            board: [],
            rest: modelSort(Array.isArray(d2.prospects) ? d2.prospects : []),
            missing: [],
            updatedAt: null,
            fallback: true,
          });
        } catch {
          if (alive) { setData(null); setError('Failed to load the big board.'); }
        }
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [classTab, reload]);

  const board = useMemo(() => (data ? data.board : []), [data]);
  const rest = useMemo(() => (data ? data.rest : []), [data]);

  // Full class pool (board + rest, deduped) — editor search runs over this.
  const pool = useMemo(() => {
    const seen = new Set();
    const out = [];
    [...board, ...rest].forEach((p) => {
      const k = keyOf(p);
      if (!seen.has(k)) { seen.add(k); out.push(p); }
    });
    return out;
  }, [board, rest]);

  const editing = draft !== null;
  const draftKeys = useMemo(() => new Set((draft || []).map(keyOf)), [draft]);
  const dirty =
    editing &&
    JSON.stringify(draft.map(keyOf)) !== JSON.stringify(board.map(keyOf));

  const poolResults = useMemo(() => {
    if (!editing) return [];
    const q = search.trim().toLowerCase();
    let list = pool;
    if (q) {
      list = list.filter((p) =>
        (p.name || '').toLowerCase().includes(q) ||
        (p.position || '').toLowerCase().includes(q) ||
        (p.team || '').toLowerCase().includes(q),
      );
    }
    return list.slice(0, 40);
  }, [editing, pool, search]);

  /* ── Editor entry / key gate ── */

  const beginEdit = useCallback((restoreKeys) => {
    setSaveError(null);
    setSearch('');
    if (restoreKeys) {
      const byKey = new Map();
      pool.forEach((p) => byKey.set(keyOf(p), p));
      setDraft(restoreKeys.map((k) => byKey.get(k)).filter(Boolean));
    } else {
      setDraft(board.map((p) => p));
    }
  }, [pool, board]);

  const requestEdit = useCallback((restore) => {
    if (sessionStorage.getItem(KEY_STORE)) {
      beginEdit(restore ? readMirror(classTab) : null);
    } else {
      setKeyPrompt({ open: true, restore: !!restore });
    }
  }, [beginEdit, classTab]);

  const submitKey = (e) => {
    e.preventDefault();
    const k = keyInput.trim();
    if (!k) return;
    sessionStorage.setItem(KEY_STORE, k);
    const restore = keyPrompt.restore;
    setKeyPrompt({ open: false, restore: false });
    setKeyInput('');
    // If the prompt reopened mid-edit (wrong key on save), keep the draft.
    if (!editing) beginEdit(restore ? readMirror(classTab) : null);
  };

  const discard = useCallback(() => {
    setDraft(null);
    setSearch('');
    setSaveError(null);
  }, []);

  /* ── Draft mutations ── */

  const moveTo = useCallback((from, to) => {
    setDraft((d) => {
      if (!d) return d;
      const bounded = Math.max(0, Math.min(d.length - 1, to));
      if (bounded === from) return d;
      const next = [...d];
      const [row] = next.splice(from, 1);
      next.splice(bounded, 0, row);
      return next;
    });
  }, []);

  const removeAt = useCallback((idx) => {
    setDraft((d) => (d ? d.filter((_, i) => i !== idx) : d));
  }, []);

  const addToBoard = useCallback((p) => {
    setDraft((d) => (d && !d.some((x) => keyOf(x) === keyOf(p)) ? [...d, p] : d));
  }, []);

  const commitRank = useCallback((idx, raw) => {
    const n = parseInt(raw, 10);
    if (!Number.isFinite(n) || n === idx + 1) return;
    moveTo(idx, n - 1);
  }, [moveTo]);

  /* ── Save / export ── */

  const save = async () => {
    if (!draft) return;
    const key = sessionStorage.getItem(KEY_STORE);
    if (!key) { setKeyPrompt({ open: true, restore: false }); return; }
    setSaving(true);
    setSaveError(null);
    const keys = draft.map(keyOf);
    try {
      const r = await fetch('/api/big-board', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Analytics-Key': key },
        body: JSON.stringify({ class: Number(classTab), board: keys }),
      });
      if (r.status === 404) {
        // Backend 404s on a bad key so the endpoint can't be probed.
        sessionStorage.removeItem(KEY_STORE);
        setSaveError('That admin key was not accepted — the board was not saved. Enter the key again.');
        setKeyPrompt({ open: true, restore: false });
        return;
      }
      if (!r.ok) throw new Error('http');
      writeMirror(classTab, keys);
      setDraft(null);
      setSearch('');
      setReload((n) => n + 1);
    } catch {
      setSaveError('Could not save the board — the server did not respond. Your edits are still here.');
    } finally {
      setSaving(false);
    }
  };

  const exportBoard = async () => {
    const key = sessionStorage.getItem(KEY_STORE);
    if (!key) { setKeyPrompt({ open: true, restore: false }); return; }
    setSaveError(null);
    try {
      const r = await fetch(`/api/big-board/export?class=${classTab}`, {
        headers: { 'X-Analytics-Key': key },
      });
      if (r.status === 404) {
        sessionStorage.removeItem(KEY_STORE);
        setSaveError('That admin key was not accepted — export refused. Enter the key again.');
        setKeyPrompt({ open: true, restore: false });
        return;
      }
      if (!r.ok) throw new Error('http');
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `big-board-${classTab}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch {
      setSaveError('Export failed — could not reach the server.');
    }
  };

  /* ── Class tabs ── */

  const selectClass = useCallback((year) => {
    if (year === classTab) return;
    if (dirty && !window.confirm('Discard unsaved board changes?')) return;
    setDraft(null);
    setSearch('');
    setSaveError(null);
    setKeyPrompt({ open: false, restore: false });
    setClassTab(year);
  }, [classTab, dirty]);

  const openReport = useCallback((p) => {
    navigate(`/predict?name=${encodeURIComponent(p.name)}`);
  }, [navigate]);

  const shownRest = useMemo(() => rest.slice(0, (page + 1) * PAGE_SIZE), [rest, page]);
  const hasMore = shownRest.length < rest.length;
  const updatedDate = formatDate(data && data.updatedAt);
  const localMirror = readMirror(classTab);

  return (
    <div className="bb-page">

      {/* ── Photo header ── */}
      <header className="bb-hero">
        <div className="bb-hero-media" aria-hidden="true">
          <img src={process.env.PUBLIC_URL + '/images/CFB Content/malachi-toney.webp'} alt="" />
        </div>
        <div className="bb-hero-scrim-x" aria-hidden="true" />
        <div className="bb-hero-scrim-y" aria-hidden="true" />
        <div className="bb-hero-inner">
          <div className="bb-hero-text">
            <div className="bb-eyebrow">Owner&rsquo;s draft board</div>
            <h1 className="bb-title">The Big Board</h1>
          </div>
          <div className="bb-count">
            {loading
              ? 'Loading…'
              : error
                ? '—'
                : `${board.length} on the board · ${rest.length.toLocaleString()} in the pool` +
                  (updatedDate ? ` · Updated ${updatedDate}` : '')}
          </div>
        </div>
      </header>

      <div className="bb-main">

        {/* ── Class tabs ── */}
        <div className="bb-controls">
          <div className="bb-seg" role="group" aria-label="Draft class">
            {CLASSES.map((y) => (
              <button
                key={y}
                type="button"
                className={`bb-seg-btn${classTab === y ? ' active' : ''}`}
                aria-pressed={classTab === y}
                onClick={() => selectClass(y)}
              >
                {`’${y.slice(2)}`}
              </button>
            ))}
          </div>
          {!editing && !loading && !error && (
            <button type="button" className="bb-edit-toggle" onClick={() => requestEdit(false)}>
              Edit board
            </button>
          )}
        </div>

        {data && data.fallback && (
          <p className="bb-note">
            Live board service unavailable — showing the model&rsquo;s ranking for the ’{classTab.slice(2)} class.
          </p>
        )}

        {/* ── Admin key gate ── */}
        {keyPrompt.open && (
          <form className="bb-gate" onSubmit={submitKey}>
            <span className="bb-gate-label">Admin key</span>
            <input
              type="password"
              className="bb-key-input"
              placeholder="Analytics key"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              autoFocus
              autoComplete="off"
              aria-label="Admin key"
            />
            <button type="submit" className="bb-btn" disabled={!keyInput.trim()}>Unlock</button>
            <button
              type="button"
              className="bb-btn bb-btn-quiet"
              onClick={() => { setKeyPrompt({ open: false, restore: false }); setKeyInput(''); }}
            >
              Cancel
            </button>
          </form>
        )}

        {loading ? (
          <div className="bb-state">Loading the board…</div>
        ) : error ? (
          <div className="bb-state">{error}</div>
        ) : editing ? (

          /* ── ── Editor ── ── */
          <>
            <div className="bb-sec-head">
              <h2 className="bb-sec-title">The Board — editing</h2>
              <span className="bb-sec-meta">{draft.length} ranked</span>
            </div>

            <div className="bb-table">
              {draft.length === 0 ? (
                <div className="bb-state">Board is empty — search the pool below and add players.</div>
              ) : draft.map((p, i) => (
                <div className="bb-edit-row" key={keyOf(p)}>
                  <input
                    key={`${keyOf(p)}-${i}`}
                    className="bb-rank-input"
                    type="number"
                    min="1"
                    max={draft.length}
                    defaultValue={i + 1}
                    aria-label={`Rank for ${p.name}`}
                    onBlur={(e) => commitRank(i, e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
                  />
                  <span className="bb-name bb-c-name">
                    {p.espn_team_id && (
                      <img
                        className="bb-team-logo"
                        src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${p.espn_team_id}.png`}
                        alt=""
                        loading="lazy"
                        onError={(e) => { e.target.style.display = 'none'; }}
                      />
                    )}
                    <span>{p.name}</span>
                  </span>
                  <span className="bb-pos bb-c-pos">{(p.position || '?').toUpperCase()}</span>
                  <span className="bb-school bb-c-school">{p.team || '—'}</span>
                  <span className="bb-edit-ctls">
                    <button type="button" className="bb-ctl" disabled={i === 0} onClick={() => moveTo(i, i - 1)} aria-label={`Move ${p.name} up`}>↑</button>
                    <button type="button" className="bb-ctl" disabled={i === draft.length - 1} onClick={() => moveTo(i, i + 1)} aria-label={`Move ${p.name} down`}>↓</button>
                    <button type="button" className="bb-ctl bb-ctl-remove" onClick={() => removeAt(i)} aria-label={`Remove ${p.name} from board`}>✕</button>
                  </span>
                </div>
              ))}
            </div>

            <div className="bb-divider"><span>Add from the ’{classTab.slice(2)} pool</span></div>

            <input
              className="bb-search"
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search player, position or school"
              aria-label="Search the class pool"
            />

            <div className="bb-table bb-pool">
              {poolResults.length === 0 ? (
                <div className="bb-state">No players match.</div>
              ) : poolResults.map((p) => {
                const onBoard = draftKeys.has(keyOf(p));
                return (
                  <div className="bb-edit-row" key={`pool-${keyOf(p)}`}>
                    <span className="bb-name bb-c-name">
                      {p.espn_team_id && (
                        <img
                          className="bb-team-logo"
                          src={`https://a.espncdn.com/i/teamlogos/ncaa/500/${p.espn_team_id}.png`}
                          alt=""
                          loading="lazy"
                          onError={(e) => { e.target.style.display = 'none'; }}
                        />
                      )}
                      <span>{p.name}</span>
                    </span>
                    <span className="bb-pos bb-c-pos">{(p.position || '?').toUpperCase()}</span>
                    <span className="bb-school bb-c-school">{p.team || '—'}</span>
                    <span className="bb-grade bb-c-grade"><span>{p.grade || '—'}</span></span>
                    <button
                      type="button"
                      className="bb-btn bb-add"
                      disabled={onBoard}
                      onClick={() => addToBoard(p)}
                    >
                      {onBoard ? 'On board' : 'Add'}
                    </button>
                  </div>
                );
              })}
            </div>

            {/* ── Unsaved-changes bar ── */}
            <div className="bb-savebar">
              <span className="bb-savebar-status">
                {saveError
                  ? <span className="bb-error">{saveError}</span>
                  : dirty ? 'Unsaved changes' : 'No changes yet'}
              </span>
              <div className="bb-savebar-actions">
                <button type="button" className="bb-btn" onClick={save} disabled={saving || !dirty}>
                  {saving ? 'Saving…' : 'Save'}
                </button>
                <button type="button" className="bb-btn bb-btn-quiet" onClick={discard} disabled={saving}>
                  Discard
                </button>
                <button type="button" className="bb-btn bb-btn-quiet" onClick={exportBoard} disabled={saving}>
                  Export JSON
                </button>
              </div>
            </div>
          </>

        ) : (

          /* ── ── View mode ── ── */
          <>
            <div className="bb-sec-head">
              <h2 className="bb-sec-title">The Board</h2>
              {board.length > 0 && <span className="bb-sec-meta">{board.length} ranked by the owner</span>}
            </div>

            {board.length === 0 ? (
              <div className="bb-empty">
                <p className="bb-empty-title">No board yet for this class</p>
                <p className="bb-empty-sub">The list below is the model&rsquo;s ranking until an owner board is published.</p>
                {localMirror && (
                  <button type="button" className="bb-btn" onClick={() => requestEdit(true)}>
                    Restore local copy ({localMirror.length} players)
                  </button>
                )}
              </div>
            ) : (
              <div className="bb-table">
                {board.map((p, i) => (
                  <BoardRow key={`b-${keyOf(p)}`} p={p} rank={i + 1} curated onOpen={() => openReport(p)} />
                ))}
              </div>
            )}

            {data && data.missing.length > 0 && (
              <p className="bb-note">
                {data.missing.length} board {data.missing.length === 1 ? 'entry' : 'entries'} did not resolve to a
                prospect: {data.missing.join(', ')}
              </p>
            )}

            <div className="bb-divider"><span>Best of the rest — model ranked</span></div>

            {rest.length === 0 ? (
              <div className="bb-state">No remaining prospects for this class.</div>
            ) : (
              <div className="bb-table">
                {shownRest.map((p, i) => (
                  <BoardRow key={`r-${keyOf(p)}`} p={p} rank={board.length + i + 1} curated={false} onOpen={() => openReport(p)} />
                ))}
              </div>
            )}

            {hasMore && (
              <div className="bb-more">
                <button type="button" className="btn btn-primary" onClick={() => setPage((pg) => pg + 1)}>
                  Load more ({(rest.length - shownRest.length).toLocaleString()} remaining)
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

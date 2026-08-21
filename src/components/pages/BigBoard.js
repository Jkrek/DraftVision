import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
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

// Grade letter -> tier class for the pill tinting (a/b/c/d).
function gradeTier(grade) {
  const c = (grade || '').charAt(0).toUpperCase();
  return c === 'A' ? 'a' : c === 'B' ? 'b' : c === 'C' ? 'c' : c === 'D' ? 'd' : '';
}

const keyOf = (p) => `${p.name}|${p.team || ''}`;
const mirrorKey = (year) => `dv_big_board_${year}`;
const myBoardKey = (year) => `dv_my_board_${year}`;

const VIEWS = [
  { id: 'jared', label: 'Jared’s Board' },
  { id: 'my', label: 'My Board' },
  { id: 'model', label: 'Model' },
];

// "Malachi Toney|Miami" -> "malachi-toney-miami" (player-page slug).
const slugOf = (p) =>
  `${p.name || ''}-${p.team || ''}`
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9-]/g, '')
    .replace(/^-+|-+$/g, '');

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

// Visitor's personal board — "Name|Team" keys in localStorage, per class.
function readMyBoard(year) {
  try {
    const raw = localStorage.getItem(myBoardKey(year));
    const arr = raw ? JSON.parse(raw) : null;
    return Array.isArray(arr)
      ? arr.filter((k, i) => typeof k === 'string' && arr.indexOf(k) === i)
      : [];
  } catch {
    return [];
  }
}

function writeMyBoard(year, keys) {
  try {
    localStorage.setItem(myBoardKey(year), JSON.stringify(keys));
  } catch {
    /* private mode etc. — best-effort */
  }
}

function BoardRow({ p, rank, curated, onOpen }) {
  const sp = p.success_probability;
  return (
    <div
      className={`bb-row${curated ? ' bb-curated' : ''}${curated && rank === 1 ? ' bb-row-no1' : ''}`}
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
        <Link
          className="bb-name-link"
          to={`/player/${slugOf(p)}`}
          onClick={(e) => e.stopPropagation()}
        >
          {p.name}
        </Link>
        {p.data_source && p.data_source !== 'espn_live' && (
          <span className="bb-est" title="No verified season stats — profile is estimated">EST</span>
        )}
      </span>
      <span className="bb-pos bb-c-pos">{(p.position || '?').toUpperCase()}</span>
      <span className="bb-school bb-c-school">{p.team || '—'}</span>
      <span className="bb-grade bb-c-grade"><span className={gradeTier(p.grade) ? `bb-grade-${gradeTier(p.grade)}` : undefined}>{p.grade || '—'}</span></span>
      <span className="bb-prob bb-c-prob">{sp != null ? Number(sp).toFixed(1) : '—'}</span>
    </div>
  );
}

export default function BigBoard() {
  const navigate = useNavigate();

  const [classTab, setClassTab] = useState(CLASSES[0]);
  const [view, setView] = useState('jared'); // 'jared' | 'my' | 'model'
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
  const [justSaved, setJustSaved] = useState(false); // transient "Saved ✓" confirmation
  const [keyPrompt, setKeyPrompt] = useState({ open: false, restore: false });
  const [keyInput, setKeyInput] = useState('');

  // Visitor's personal board — always-editable, browser-local, no admin key.
  const [myKeys, setMyKeys] = useState(() => readMyBoard(CLASSES[0]));
  const [mySearch, setMySearch] = useState('');

  useEffect(() => {
    setMyKeys(readMyBoard(classTab));
    setMySearch('');
  }, [classTab]);

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

  const poolByKey = useMemo(() => {
    const m = new Map();
    pool.forEach((p) => m.set(keyOf(p), p));
    return m;
  }, [pool]);

  // Pure model ordering over the whole class (board + rest merged).
  const modelRanking = useMemo(() => modelSort(pool), [pool]);

  // My-board keys resolved to prospects; keys that don't resolve (e.g. a
  // player left the class pool) still render from the stored "Name|Team".
  const myBoard = useMemo(
    () => myKeys.map((k) => poolByKey.get(k) ||
      { name: k.split('|')[0], team: k.split('|')[1] || '', unresolved: true }),
    [myKeys, poolByKey],
  );
  const myKeySet = useMemo(() => new Set(myKeys), [myKeys]);

  // Top-10 agreement between the visitor's board, Jared's board and the model.
  const compare = useMemo(() => {
    if (myKeys.length === 0) return null;
    const mySet = new Set(myKeys.slice(0, 10));
    const count = (list) =>
      list.slice(0, 10).map(keyOf).filter((k) => mySet.has(k)).length;
    return {
      jared: board.length > 0
        ? { hit: count(board), n: Math.min(10, mySet.size, board.length) }
        : null,
      model: modelRanking.length > 0
        ? { hit: count(modelRanking), n: Math.min(10, mySet.size, modelRanking.length) }
        : null,
    };
  }, [myKeys, board, modelRanking]);

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

  const myPoolResults = useMemo(() => {
    if (view !== 'my' || editing) return [];
    const q = mySearch.trim().toLowerCase();
    let list = pool;
    if (q) {
      list = list.filter((p) =>
        (p.name || '').toLowerCase().includes(q) ||
        (p.position || '').toLowerCase().includes(q) ||
        (p.team || '').toLowerCase().includes(q),
      );
    }
    return list.slice(0, 40);
  }, [view, editing, pool, mySearch]);

  /* ── My-board mutations — persisted to localStorage on every change ── */

  const updateMy = useCallback((fn) => {
    setMyKeys((keys) => {
      const next = fn(keys);
      if (next !== keys) writeMyBoard(classTab, next);
      return next;
    });
  }, [classTab]);

  const myMoveTo = useCallback((from, to) => {
    updateMy((keys) => {
      const bounded = Math.max(0, Math.min(keys.length - 1, to));
      if (bounded === from) return keys;
      const next = [...keys];
      const [k] = next.splice(from, 1);
      next.splice(bounded, 0, k);
      return next;
    });
  }, [updateMy]);

  const myRemoveAt = useCallback((idx) => {
    updateMy((keys) => keys.filter((_, i) => i !== idx));
  }, [updateMy]);

  const myAdd = useCallback((p) => {
    updateMy((keys) => (keys.includes(keyOf(p)) ? keys : [...keys, keyOf(p)]));
  }, [updateMy]);

  const myCommitRank = useCallback((idx, raw) => {
    const n = parseInt(raw, 10);
    if (!Number.isFinite(n) || n === idx + 1) return;
    myMoveTo(idx, n - 1);
  }, [myMoveTo]);

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
      setJustSaved(true);
      window.setTimeout(() => setJustSaved(false), 4000);
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

  const selectView = useCallback((v) => {
    if (v === view) return;
    setView(v);
    setPage(0);
  }, [view]);

  const openReport = useCallback((p) => {
    navigate(`/predict?name=${encodeURIComponent(p.name)}`);
  }, [navigate]);

  const shownRest = useMemo(() => rest.slice(0, (page + 1) * PAGE_SIZE), [rest, page]);
  const hasMore = shownRest.length < rest.length;
  const shownModel = useMemo(
    () => modelRanking.slice(0, (page + 1) * PAGE_SIZE),
    [modelRanking, page],
  );
  const modelHasMore = shownModel.length < modelRanking.length;
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
          {!editing && (
            <div className="bb-seg" role="group" aria-label="Board view">
              {VIEWS.map((v) => (
                <button
                  key={v.id}
                  type="button"
                  className={`bb-seg-btn${view === v.id ? ' active' : ''}`}
                  aria-pressed={view === v.id}
                  onClick={() => selectView(v.id)}
                >
                  {v.label}
                </button>
              ))}
            </div>
          )}
          {!editing && !loading && !error && view === 'jared' && (
            <button type="button" className="bb-edit-toggle" onClick={() => requestEdit(false)}>
              Edit board
            </button>
          )}
          {justSaved && (
            <span className="bb-saved-note" role="status">Board saved ✓</span>
          )}
        </div>

        {data && data.fallback && (
          <p className="bb-note">
            Live board service unavailable — showing the model&rsquo;s ranking for the ’{classTab.slice(2)} class.
          </p>
        )}

        {/* ── Top-10 agreement strip — shown once the visitor has a board ── */}
        {!editing && !loading && !error && compare && (
          <div className="bb-compare" aria-label="Top-10 agreement">
            <span className="bb-compare-label">Top-10 overlap</span>
            {compare.jared && (
              <span className="bb-compare-chip">
                You and Jared agree on <strong>{compare.jared.hit} of {compare.jared.n}</strong>
              </span>
            )}
            {compare.model && (
              <span className="bb-compare-chip">
                You and the model agree on <strong>{compare.model.hit} of {compare.model.n}</strong>
              </span>
            )}
          </div>
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
                    <span className="bb-grade bb-c-grade"><span className={gradeTier(p.grade) ? `bb-grade-${gradeTier(p.grade)}` : undefined}>{p.grade || '—'}</span></span>
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
                <button type="button" className="bb-btn bb-save" onClick={save} disabled={saving || !dirty}>
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

        ) : view === 'my' ? (

          /* ── ── My Board — visitor's own, always-on editor ── ── */
          <>
            <div className="bb-sec-head">
              <h2 className="bb-sec-title">My Board</h2>
              <span className="bb-sec-meta">{myBoard.length} ranked · saved in your browser</span>
            </div>

            <p className="bb-note">
              Your board lives in this browser only (no account, no admin key) — edits save automatically.
            </p>

            <div className="bb-table">
              {myBoard.length === 0 ? (
                <div className="bb-state">Your board is empty — search the pool below and add players.</div>
              ) : myBoard.map((p, i) => (
                <div className="bb-edit-row" key={myKeys[i]}>
                  <input
                    key={`${myKeys[i]}-${i}`}
                    className="bb-rank-input"
                    type="number"
                    min="1"
                    max={myBoard.length}
                    defaultValue={i + 1}
                    aria-label={`Rank for ${p.name}`}
                    onBlur={(e) => myCommitRank(i, e.target.value)}
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
                    <Link className="bb-name-link" to={`/player/${slugOf(p)}`}>{p.name}</Link>
                  </span>
                  <span className="bb-pos bb-c-pos">{(p.position || '?').toUpperCase()}</span>
                  <span className="bb-school bb-c-school">{p.team || '—'}</span>
                  <span className="bb-grade bb-c-grade"><span className={gradeTier(p.grade) ? `bb-grade-${gradeTier(p.grade)}` : undefined}>{p.grade || '—'}</span></span>
                  <span className="bb-edit-ctls">
                    <button type="button" className="bb-ctl" disabled={i === 0} onClick={() => myMoveTo(i, i - 1)} aria-label={`Move ${p.name} up`}>↑</button>
                    <button type="button" className="bb-ctl" disabled={i === myBoard.length - 1} onClick={() => myMoveTo(i, i + 1)} aria-label={`Move ${p.name} down`}>↓</button>
                    <button type="button" className="bb-ctl bb-ctl-remove" onClick={() => myRemoveAt(i)} aria-label={`Remove ${p.name} from your board`}>✕</button>
                  </span>
                </div>
              ))}
            </div>

            <div className="bb-divider"><span>Add from the ’{classTab.slice(2)} pool</span></div>

            <input
              className="bb-search"
              type="search"
              value={mySearch}
              onChange={(e) => setMySearch(e.target.value)}
              placeholder="Search player, position or school"
              aria-label="Search the class pool"
            />

            <div className="bb-table">
              {myPoolResults.length === 0 ? (
                <div className="bb-state">No players match.</div>
              ) : myPoolResults.map((p) => {
                const onBoard = myKeySet.has(keyOf(p));
                return (
                  <div className="bb-edit-row" key={`mypool-${keyOf(p)}`}>
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
                      <Link className="bb-name-link" to={`/player/${slugOf(p)}`}>{p.name}</Link>
                    </span>
                    <span className="bb-pos bb-c-pos">{(p.position || '?').toUpperCase()}</span>
                    <span className="bb-school bb-c-school">{p.team || '—'}</span>
                    <span className="bb-grade bb-c-grade"><span className={gradeTier(p.grade) ? `bb-grade-${gradeTier(p.grade)}` : undefined}>{p.grade || '—'}</span></span>
                    <button
                      type="button"
                      className="bb-btn bb-add"
                      disabled={onBoard}
                      onClick={() => myAdd(p)}
                    >
                      {onBoard ? 'On board' : 'Add'}
                    </button>
                  </div>
                );
              })}
            </div>
          </>

        ) : view === 'model' ? (

          /* ── ── Pure model ranking — board + rest merged, model-sorted ── ── */
          <>
            <div className="bb-sec-head">
              <h2 className="bb-sec-title">Model Ranking</h2>
              <span className="bb-sec-meta">
                {modelRanking.length.toLocaleString()} prospects · grade, then success probability
              </span>
            </div>

            {modelRanking.length === 0 ? (
              <div className="bb-state">No prospects for this class.</div>
            ) : (
              <div className="bb-table">
                {shownModel.map((p, i) => (
                  <BoardRow key={`m-${keyOf(p)}`} p={p} rank={i + 1} curated={false} onOpen={() => openReport(p)} />
                ))}
              </div>
            )}

            {modelHasMore && (
              <div className="bb-more">
                <button type="button" className="btn btn-primary" onClick={() => setPage((pg) => pg + 1)}>
                  Load more ({(modelRanking.length - shownModel.length).toLocaleString()} remaining)
                </button>
              </div>
            )}
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

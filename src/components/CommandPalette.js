import React, {
  useCallback, useEffect, useMemo, useRef, useState,
} from 'react';
import { createPortal } from 'react-dom';
import { useNavigate } from 'react-router-dom';
import { anonFetch } from '../lib/api';
import { toggleTheme } from '../theme';
import './CommandPalette.css';

/*
 * CommandPalette — ⌘K / Ctrl-K.
 *
 * ARIA combobox pattern: the input owns the interaction (role="combobox",
 * aria-expanded / aria-controls / aria-activedescendant) and the rows are
 * role="option" inside a single listbox, grouped by eyebrow-labelled
 * role="group" sections. Nothing but the input takes DOM focus, so the trap
 * is simply "Tab does nothing"; focus returns to whatever opened the palette.
 *
 * Opened by the keyboard shortcut or by a `dv:palette-open` window event
 * (the navbar hint button dispatches it), so no shared state is needed.
 */

export const PALETTE_OPEN_EVENT = 'dv:palette-open';

const RECENT_KEY = 'dv_palette_recent';
const MAX_PLAYERS = 8;
const DEBOUNCE_MS = 150;
const COPIED_MS = 1200;

const PAGES = [
  { to: '/predict',      label: 'Predict',      hint: 'Run the model on any college player' },
  { to: '/leaderboard',  label: 'Model Board',  hint: 'Every prospect the model has graded' },
  { to: '/big-board',    label: 'Big Board',    hint: 'The board, ranked and filterable' },
  { to: '/futures',      label: 'Futures',      hint: 'Live disagreements with the market' },
  { to: '/hs-prospects', label: 'HS Prospects', hint: 'High-school recruits and star ratings' },
  { to: '/compare',      label: 'Compare',      hint: 'Two prospects, one screen' },
  { to: '/mock-draft',   label: 'Mock Draft',   hint: 'Build a round with the model' },
  { to: '/backtest',     label: 'Backtest',     hint: 'How the model scored past classes' },
  { to: '/',             label: 'Home',         hint: 'Back to the front page' },
];

// Shareable slug: "name-team" lowercased, spaces→'-', strip non-alnum except dash.
function slugify(name, team) {
  return `${name || ''}-${team || ''}`
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9-]/g, '');
}

// Grade letter → tier class (A gold, B accent, C/D muted) — same idea as the board.
function gradeTier(grade) {
  const c = (grade || '').charAt(0).toUpperCase();
  return c === 'A' ? 'a' : c === 'B' ? 'b' : c === 'C' ? 'c' : c === 'D' ? 'd' : '';
}

const logoUrl = (espnTeamId) => (
  espnTeamId ? `https://a.espncdn.com/i/teamlogos/ncaa/500/${espnTeamId}.png` : null
);

// Substring first, then a cheap subsequence pass ("mkdr" → "Mock Draft").
function fuzzyMatch(query, text) {
  const q = (query || '').toLowerCase().replace(/\s+/g, '');
  if (!q) return true;
  const t = (text || '').toLowerCase();
  if (t.includes((query || '').toLowerCase().trim())) return true;
  let i = 0;
  for (let k = 0; k < t.length && i < q.length; k += 1) {
    if (t[k] === q[i]) i += 1;
  }
  return i === q.length;
}

function readRecent() {
  try {
    const parsed = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
    return Array.isArray(parsed) ? parsed.filter((r) => r && r.name && r.to).slice(0, 5) : [];
  } catch {
    return [];
  }
}

function writeRecent(entry) {
  const kept = readRecent().filter((r) => r.to !== entry.to);
  const next = [entry, ...kept].slice(0, 5);
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(next));
  } catch {
    // private mode — the palette still works, it just forgets.
  }
  return next;
}

// A search row → where it goes. Only college rows have a player page; HS rows
// hand off to the HS board's own search; legacy DB rows go through /predict.
function playerTarget(p) {
  if (p.kind === 'hs') return `/hs-prospects?q=${encodeURIComponent(p.name)}`;
  if (p.kind === 'college') return `/player/${slugify(p.name, p.team)}`;
  return `/predict?name=${encodeURIComponent(p.name)}`;
}

function PlayerRow({ item }) {
  const src = logoUrl(item.espnTeamId);
  const tier = gradeTier(item.grade);
  return (
    <>
      <span className="cmdk-logo" aria-hidden="true">
        {src && (
          <img
            src={src}
            alt=""
            loading="lazy"
            onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }}
          />
        )}
      </span>
      <span className="cmdk-row-main">
        <span className="cmdk-row-name">{item.name}</span>
        <span className="cmdk-row-sub">{item.sub}</span>
      </span>
      {item.grade && (
        <span className={`cmdk-grade${tier ? ` cmdk-grade-${tier}` : ''}`}>{item.grade}</span>
      )}
    </>
  );
}

export default function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [players, setPlayers] = useState([]);
  const [recent, setRecent] = useState([]);
  const [copied, setCopied] = useState(false);
  const [active, setActive] = useState(0);

  const navigate = useNavigate();
  const inputRef = useRef(null);
  const listRef = useRef(null);
  const restoreRef = useRef(null);
  const copyTimer = useRef(null);

  const q = query.trim();

  const close = useCallback(() => setOpen(false), []);

  // ── open / close plumbing ───────────────────────────────────────────────
  useEffect(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && !e.altKey && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    const onOpenEvent = () => setOpen(true);
    window.addEventListener('keydown', onKey);
    window.addEventListener(PALETTE_OPEN_EVENT, onOpenEvent);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener(PALETTE_OPEN_EVENT, onOpenEvent);
    };
  }, []);

  // Fresh query each time it opens; remember who to hand focus back to.
  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement;
    setQuery('');
    setPlayers([]);
    setActive(0);
    setCopied(false);
    setRecent(readRecent());
    // The portal is already in the DOM by the time effects run.
    if (inputRef.current) inputRef.current.focus();
  }, [open]);

  // Focus returns to the previously focused element on close.
  useEffect(() => {
    if (open) return;
    const prev = restoreRef.current;
    restoreRef.current = null;
    if (prev && typeof prev.focus === 'function' && document.contains(prev)) prev.focus();
  }, [open]);

  // Body scroll lock while open.
  useEffect(() => {
    if (!open) return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = prev; };
  }, [open]);

  useEffect(() => () => { if (copyTimer.current) clearTimeout(copyTimer.current); }, []);

  // ── debounced player search ─────────────────────────────────────────────
  useEffect(() => {
    if (!open || q.length < 2) { setPlayers([]); return undefined; }
    const ctrl = new AbortController();
    const timer = setTimeout(() => {
      anonFetch(`/search?q=${encodeURIComponent(q)}`, { signal: ctrl.signal })
        .then((res) => res.json())
        .then((data) => {
          setPlayers(Array.isArray(data.players) ? data.players.slice(0, MAX_PLAYERS) : []);
        })
        .catch(() => { /* aborted or offline — the other sections still stand */ });
    }, DEBOUNCE_MS);
    return () => { clearTimeout(timer); ctrl.abort(); };
  }, [open, q]);

  // ── actions ─────────────────────────────────────────────────────────────
  const go = useCallback((to) => { close(); navigate(to); }, [close, navigate]);

  const copyLink = useCallback(() => {
    const done = () => {
      setCopied(true);
      if (copyTimer.current) clearTimeout(copyTimer.current);
      copyTimer.current = setTimeout(() => setCopied(false), COPIED_MS);
    };
    try {
      const write = navigator.clipboard && navigator.clipboard.writeText;
      if (write) navigator.clipboard.writeText(window.location.href).then(done, () => {});
    } catch {
      // clipboard blocked (insecure origin / permissions) — stay quiet.
    }
  }, []);

  const pickPlayer = useCallback((item) => {
    setRecent(writeRecent({
      name: item.name,
      sub: item.sub,
      grade: item.grade,
      espnTeamId: item.espnTeamId,
      to: item.to,
    }));
    go(item.to);
  }, [go]);

  // ── the grouped result set ──────────────────────────────────────────────
  const groups = useMemo(() => {
    const out = [];

    if (q.length < 2 && recent.length > 0) {
      out.push({
        key: 'recent',
        label: 'Recent',
        items: recent.map((r, i) => ({
          key: `recent-${i}-${r.to}`,
          kind: 'player',
          name: r.name,
          sub: r.sub,
          grade: r.grade,
          espnTeamId: r.espnTeamId,
          to: r.to,
        })),
      });
    }

    if (players.length > 0) {
      out.push({
        key: 'players',
        label: 'Players',
        items: players.map((p, i) => {
          const where = p.kind === 'hs' ? p.school : p.team;
          const sub = [p.position, where].filter(Boolean).join(' · ')
            || (p.kind === 'hs' ? 'High school' : 'Pro / legacy');
          return {
            key: `player-${p.kind}-${p.name}-${where || i}`,
            kind: 'player',
            name: p.name,
            sub,
            grade: p.grade,
            espnTeamId: p.espn_team_id,
            to: playerTarget(p),
          };
        }),
      });
    }

    const pages = PAGES.filter((p) => fuzzyMatch(q, `${p.label} ${p.hint}`));
    if (pages.length > 0) {
      out.push({
        key: 'pages',
        label: 'Pages',
        items: pages.map((p) => ({
          key: `page-${p.to}`,
          kind: 'page',
          name: p.label,
          sub: p.hint,
          to: p.to,
        })),
      });
    }

    const actions = [];
    if (q.length >= 2) {
      actions.push({
        key: 'action-predict',
        kind: 'action',
        name: `Predict “${q}”`,
        sub: 'Run the model on this name',
        run: () => go(`/predict?name=${encodeURIComponent(q)}`),
      });
    }
    const rest = [
      {
        key: 'action-theme',
        kind: 'action',
        name: 'Toggle theme',
        sub: 'Switch between the light and dark ground',
        run: () => { toggleTheme(); },
      },
      {
        key: 'action-copy',
        kind: 'action',
        name: 'Copy link to this page',
        sub: 'Puts the current URL on the clipboard',
        run: copyLink,
      },
      {
        key: 'action-record',
        kind: 'action',
        name: 'Go to the paper record',
        sub: 'Every logged futures call, scored',
        run: () => go('/futures#record'),
      },
    ].filter((a) => fuzzyMatch(q, `${a.name} ${a.sub}`));
    actions.push(...rest);

    if (actions.length > 0) out.push({ key: 'actions', label: 'Actions', items: actions });
    return out;
  }, [q, players, recent, go, copyLink]);

  // Flatten once for keyboard traversal; index is the source of truth.
  const flat = useMemo(() => groups.flatMap((g) => g.items), [groups]);
  useEffect(() => { setActive(0); }, [q, players, recent]);

  const clamped = flat.length === 0 ? -1 : Math.min(active, flat.length - 1);
  const activeId = clamped >= 0 ? `cmdk-opt-${clamped}` : undefined;

  // Keep the selected row in view without scrolling the page.
  useEffect(() => {
    if (!open || clamped < 0 || !listRef.current) return;
    const el = listRef.current.querySelector(`#cmdk-opt-${clamped}`);
    if (el && el.scrollIntoView) el.scrollIntoView({ block: 'nearest' });
  }, [open, clamped]);

  const run = useCallback((item) => {
    if (!item) return;
    if (item.kind === 'player') pickPlayer(item);
    else if (item.kind === 'page') go(item.to);
    else if (item.run) item.run();
  }, [pickPlayer, go]);

  const onKeyDown = (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (flat.length > 0) setActive((i) => (Math.min(i, flat.length - 1) + 1) % flat.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (flat.length > 0) {
        setActive((i) => (Math.min(i, flat.length - 1) - 1 + flat.length) % flat.length);
      }
    } else if (e.key === 'Home') {
      e.preventDefault();
      setActive(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      setActive(Math.max(0, flat.length - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      run(flat[clamped]);
    } else if (e.key === 'Tab') {
      // Only the input is focusable — the trap is "Tab goes nowhere".
      e.preventDefault();
    }
  };

  if (!open) return null;

  let idx = -1;

  return createPortal(
    <div className="cmdk-root">
      <div className="cmdk-backdrop" onClick={close} aria-hidden="true" />
      <div
        className="cmdk-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Search and commands"
        onKeyDown={onKeyDown}
      >
        <div className="cmdk-inputrow">
          <i className="fas fa-search cmdk-search-icon" aria-hidden="true" />
          <input
            ref={inputRef}
            type="text"
            className="cmdk-input"
            role="combobox"
            aria-expanded={flat.length > 0}
            aria-controls="cmdk-listbox"
            aria-activedescendant={activeId}
            aria-autocomplete="list"
            aria-label="Search players, pages and commands"
            autoComplete="off"
            spellCheck="false"
            placeholder="Type a player, page or command"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {copied && <span className="cmdk-copied" role="status">Copied</span>}
        </div>

        <div className="cmdk-list" ref={listRef}>
          <div id="cmdk-listbox" role="listbox" aria-label="Results">
            {groups.map((g) => (
              <div className="cmdk-group" role="group" aria-labelledby={`cmdk-grp-${g.key}`} key={g.key}>
                <div className="cmdk-eyebrow" id={`cmdk-grp-${g.key}`}>{g.label}</div>
                {g.items.map((item) => {
                  idx += 1;
                  const i = idx;
                  return (
                    <div
                      key={item.key}
                      id={`cmdk-opt-${i}`}
                      role="option"
                      aria-selected={i === clamped}
                      className={`cmdk-row${i === clamped ? ' cmdk-row-active' : ''}`}
                      onMouseMove={() => setActive(i)}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => run(item)}
                    >
                      {item.kind === 'player' ? <PlayerRow item={item} /> : (
                        <>
                          <span className="cmdk-glyph" aria-hidden="true">
                            <i className={item.kind === 'page' ? 'fas fa-arrow-right' : 'fas fa-bolt'} />
                          </span>
                          <span className="cmdk-row-main">
                            <span className="cmdk-row-name">{item.name}</span>
                            <span className="cmdk-row-sub">{item.sub}</span>
                          </span>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          {flat.length === 0 && (
            <p className="cmdk-empty">
              {`Nothing for “${q}” — try a player or a page.`}
            </p>
          )}
        </div>

        <div className="cmdk-footer">
          <span>↑↓ navigate · ↵ open · esc close</span>
        </div>
      </div>
    </div>,
    document.body,
  );
}

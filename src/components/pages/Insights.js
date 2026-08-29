import React, { useState, useEffect, useCallback } from 'react';
import './Insights.css';

// Private owner dashboard. Not linked from nav — reachable only at /insights,
// and the analytics key is the gate (backend returns 404 on a wrong key so the
// endpoint can't be probed; we surface that as "key not accepted").

const KEY_STORE = 'dv_analytics_key';

const fmt = (n) => (Number(n) || 0).toLocaleString('en-US');

// "2026-08-16" -> "Aug 16"; falls back to the raw string.
function shortDate(iso) {
  if (!iso) return '';
  const d = new Date(`${iso}T00:00:00`);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

// Full ISO timestamp -> "Aug 29, 2:14 PM" in the viewer's local time.
function shortStamp(iso) {
  if (!iso) return '';
  const d = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : `${iso}Z`);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  });
}

export default function Insights() {
  const [storedKey, setStoredKey] = useState(
    () => sessionStorage.getItem(KEY_STORE) || ''
  );
  const [keyInput, setKeyInput] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback((key) => {
    setLoading(true);
    setError(null);
    fetch(`/api/analytics/summary?key=${encodeURIComponent(key)}`)
      .then((r) => {
        if (r.status === 404) throw new Error('bad-key');
        if (!r.ok) throw new Error('http');
        return r.json();
      })
      .then((d) => {
        sessionStorage.setItem(KEY_STORE, key);
        setStoredKey(key);
        setData(d);
      })
      .catch((e) => {
        if (e.message === 'bad-key') {
          sessionStorage.removeItem(KEY_STORE);
          setStoredKey('');
          setData(null);
          setError('That key was not accepted. Check it and try again.');
        } else {
          setError('Could not reach the analytics service. Try again in a moment.');
        }
      })
      .finally(() => setLoading(false));
  }, []);

  // A key remembered from earlier in the session unlocks without re-prompting.
  useEffect(() => {
    if (storedKey) load(storedKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submitKey = (e) => {
    e.preventDefault();
    const k = keyInput.trim();
    if (k) load(k);
  };

  const forgetKey = () => {
    sessionStorage.removeItem(KEY_STORE);
    setStoredKey('');
    setData(null);
    setKeyInput('');
    setError(null);
  };

  /* ── Key gate ── */
  if (!data) {
    return (
      <div className="ins-page">
        <div className="ins-gate">
          <p className="eyebrow">Owner access</p>
          <h1 className="ins-gate-title">Insights</h1>
          <p className="ins-gate-sub">
            This dashboard is private. Enter the analytics key to continue.
          </p>
          <form className="ins-gate-form" onSubmit={submitKey}>
            <input
              type="password"
              className="ins-key-input"
              placeholder="Analytics key"
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              autoFocus
              autoComplete="off"
            />
            <button
              type="submit"
              className="ins-btn"
              disabled={loading || !keyInput.trim()}
            >
              {loading ? 'Checking…' : 'Unlock'}
            </button>
          </form>
          {loading && storedKey && (
            <p className="ins-gate-note">Unlocking with your saved key…</p>
          )}
          {error && <p className="ins-error">{error}</p>}
        </div>
      </div>
    );
  }

  /* ── Dashboard ── */
  const stats = [
    { label: 'DAU', value: data.dau },
    { label: 'WAU', value: data.wau },
    { label: 'MAU', value: data.mau },
    { label: 'Returning · 7d', value: data.returning_7d },
    { label: 'Signed-in WAU', value: data.signed_in_wau },
    { label: 'Signups · total', value: data.signups_total },
  ];

  const series = Array.isArray(data.daily_series_30d) ? data.daily_series_30d : [];
  const maxVisitors = Math.max(1, ...series.map((d) => Number(d.visitors) || 0));
  const routes = Array.isArray(data.top_routes_7d) ? data.top_routes_7d : [];
  const events = Array.isArray(data.recent_events) ? data.recent_events : [];
  const users = Array.isArray(data.recent_users) ? data.recent_users : [];

  return (
    <div className="ins-page">
      <header className="ins-head">
        <div>
          <p className="eyebrow">Private · usage</p>
          <h1 className="ins-title">Insights</h1>
        </div>
        <div className="ins-head-actions">
          <button className="ins-btn" onClick={() => load(storedKey)} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
          <button className="ins-btn ins-btn-quiet" onClick={forgetKey}>
            Forget key
          </button>
        </div>
      </header>

      <main className="ins-main">
        {error && <p className="ins-error">{error}</p>}

        {/* Stat cells */}
        <section className="ins-cells" aria-label="Audience stats">
          {stats.map((s) => (
            <div key={s.label} className="ins-cell">
              <span className="ins-cell-label">{s.label}</span>
              <span className="ins-cell-value">{fmt(s.value)}</span>
            </div>
          ))}
        </section>

        {/* 30-day sparkline */}
        <section className="ins-block" aria-label="Daily visitors, last 30 days">
          <div className="ins-block-head">
            <span className="ins-block-title">Daily visitors</span>
            <span className="ins-block-meta">last 30 days</span>
          </div>
          {series.length === 0 ? (
            <p className="ins-empty">No traffic recorded yet.</p>
          ) : (
            <>
              <div className="ins-spark">
                {series.map((d) => {
                  const v = Number(d.visitors) || 0;
                  const pct = Math.max(v > 0 ? 4 : 0, (v / maxVisitors) * 100);
                  return (
                    <div
                      key={d.date}
                      className="ins-spark-col"
                      title={`${shortDate(d.date)} — ${fmt(v)} visitors · ${fmt(d.hits)} hits`}
                    >
                      <span className="ins-spark-bar" style={{ height: `${pct}%` }} />
                    </div>
                  );
                })}
              </div>
              <div className="ins-spark-axis">
                <span>{shortDate(series[0].date)}</span>
                <span>{shortDate(series[series.length - 1].date)}</span>
              </div>
            </>
          )}
        </section>

        {/* Top routes */}
        <section className="ins-block ins-block-flush" aria-label="Top routes, last 7 days">
          <div className="ins-block-head ins-block-head-pad">
            <span className="ins-block-title">Top routes</span>
            <span className="ins-block-meta">last 7 days</span>
          </div>
          <div className="ins-thead">
            <span className="ins-c-route">Route</span>
            <span className="ins-c-num">Hits</span>
            <span className="ins-c-num">Uniques</span>
          </div>
          {routes.length === 0 ? (
            <p className="ins-empty ins-empty-pad">Nothing yet.</p>
          ) : (
            routes.map((r) => (
              <div key={r.route} className="ins-row">
                <span className="ins-c-route ins-route">{r.route}</span>
                <span className="ins-c-num ins-num">{fmt(r.hits)}</span>
                <span className="ins-c-num ins-num">{fmt(r.uniques)}</span>
              </div>
            ))
          )}
        </section>

        {/* Live activity feed */}
        <section className="ins-block ins-block-flush" aria-label="Recent activity">
          <div className="ins-block-head ins-block-head-pad">
            <span className="ins-block-title">Live activity</span>
            <span className="ins-block-meta">last {events.length} tracked requests</span>
          </div>
          <div className="ins-thead ins-thead-activity">
            <span className="ins-c-when">When</span>
            <span className="ins-c-route">Route</span>
            <span className="ins-c-vis">Visitor</span>
            <span className="ins-c-num">Status</span>
            <span className="ins-c-num">ms</span>
          </div>
          {events.length === 0 ? (
            <p className="ins-empty ins-empty-pad">Nothing yet.</p>
          ) : (
            events.map((e, i) => (
              <div key={`${e.ts}-${i}`} className="ins-row ins-row-activity">
                <span className="ins-c-when">{shortStamp(e.ts)}</span>
                <span className="ins-c-route ins-route">
                  {e.method !== 'GET' && <span className="ins-method">{e.method} </span>}
                  {e.route}
                </span>
                <span className={`ins-c-vis${e.signed_in ? ' ins-vis-user' : ''}`}>
                  {e.signed_in ? `user·${e.visitor}` : (e.visitor ? `anon·${e.visitor}` : '—')}
                </span>
                <span className={`ins-c-num ins-num${e.status >= 400 ? ' ins-status-err' : ''}`}>
                  {e.status}
                </span>
                <span className="ins-c-num ins-num">{e.dur_ms != null ? fmt(e.dur_ms) : '—'}</span>
              </div>
            ))
          )}
        </section>

        {/* Recent signed-in users */}
        <section className="ins-block ins-block-flush" aria-label="Recent users">
          <div className="ins-block-head ins-block-head-pad">
            <span className="ins-block-title">Recent users</span>
            <span className="ins-block-meta">by last seen</span>
          </div>
          <div className="ins-thead ins-thead-users">
            <span className="ins-c-route">User</span>
            <span className="ins-c-vis">Tier</span>
            <span className="ins-c-when">Signed up</span>
            <span className="ins-c-when">Last seen</span>
          </div>
          {users.length === 0 ? (
            <p className="ins-empty ins-empty-pad">No signups yet.</p>
          ) : (
            users.map((u) => (
              <div key={u.email || u.name} className="ins-row ins-row-users">
                <span className="ins-c-route ins-route">{u.email || u.name || '—'}</span>
                <span className="ins-c-vis">{u.tier || 'free'}</span>
                <span className="ins-c-when">{shortStamp(u.created_at)}</span>
                <span className="ins-c-when">{shortStamp(u.last_seen)}</span>
              </div>
            ))
          )}
        </section>

        <p className="ins-foot-note">
          Server logs (deploys, errors, stack traces) live in{' '}
          <code>fly logs -a draftvision</code> — this feed is the product-side
          view: who's here and what they're hitting.
        </p>
      </main>
    </div>
  );
}

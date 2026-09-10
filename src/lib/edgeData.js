import { anonFetch } from './api';

/*
 * edgeData — the one shared cache for every "model vs. market" surface
 * (nav live dot, home FuturesBand, /futures page, player-page MarketPanel,
 * predict-report metric).
 *
 * Contract (frozen — see the futures build spec):
 *   getEdge()                 → /api/edge payload
 *   getSpotlight(n = 3)       → /api/edge/spotlight?n=
 *   getPlayerMarkets(name, team) → /api/edge/player?name=&team=
 *   getLedger()               → /api/edge/ledger
 *   invalidate()              → drop every cached promise
 *
 * Each call is a module-scope promise cache keyed by URL with a 5-minute
 * TTL. A 'warming' payload is cached for only 4 s so pollers can re-ask.
 * Errors NEVER reject: they resolve to { state: 'error' } so no surface
 * can throw. Requirement, not optimisation: one home visit produces at
 * most one /api/edge/spotlight request, and the nav dot must not add a
 * second /api/edge call on top of the page's own.
 */

const TTL_MS = 5 * 60 * 1000;
const WARMING_TTL_MS = 4 * 1000;

const cache = new Map(); // url → { promise, expires }

function isWarming(payload) {
  return payload && (payload.state === 'warming' || payload.note === 'warming'
    || (payload.discovery && payload.discovery.state === 'warming'));
}

function cachedFetch(url) {
  const now = Date.now();
  const hit = cache.get(url);
  if (hit && hit.expires > now) return hit.promise;

  const promise = anonFetch(url, { signal: AbortSignal.timeout(20000) })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
    .then((json) => {
      // re-arm a short TTL for warming payloads so pollers see the real data soon
      const ttl = isWarming(json) ? WARMING_TTL_MS : TTL_MS;
      cache.set(url, { promise: Promise.resolve(json), expires: Date.now() + ttl });
      return json;
    })
    .catch((err) => {
      const fallback = { state: 'error', error: err && err.message ? err.message : 'fetch failed' };
      // brief negative cache so a flapping backend is not hammered
      cache.set(url, { promise: Promise.resolve(fallback), expires: Date.now() + WARMING_TTL_MS });
      return fallback;
    });

  cache.set(url, { promise, expires: now + TTL_MS });
  return promise;
}

export function getEdge() {
  return cachedFetch('/api/edge');
}

export function getSpotlight(n = 3) {
  return cachedFetch(`/api/edge/spotlight?n=${encodeURIComponent(n)}`);
}

export function getPlayerMarkets(name, team) {
  const q = new URLSearchParams();
  if (name) q.set('name', name);
  if (team) q.set('team', team);
  return cachedFetch(`/api/edge/player?${q.toString()}`);
}

export function getLedger() {
  return cachedFetch('/api/edge/ledger');
}

export function invalidate() {
  cache.clear();
}

/* "Name-Team" → player-page slug (same encoding the boards use). */
export function playerSlug(name, team) {
  return `${name || ''}-${team || ''}`
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9-]/g, '');
}

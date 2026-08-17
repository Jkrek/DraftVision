import { useCallback } from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { isAuth0Enabled } from '../auth/config';

// ── API contract for the analytics backend ──────────────────────────────
//   Authorization: Bearer <Auth0 access token>   (only when logged in)
//   X-DV-Anon: <stable per-browser uuid>         (always attached)
// ────────────────────────────────────────────────────────────────────────

const ANON_KEY = 'dv_anon_id';

function uuid() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older browsers.
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export function getAnonId() {
  try {
    let id = localStorage.getItem(ANON_KEY);
    if (!id) {
      id = uuid();
      localStorage.setItem(ANON_KEY, id);
    }
    return id;
  } catch {
    // localStorage unavailable (private mode, etc.) — stay anonymous-less.
    return null;
  }
}

function buildHeaders(options, token) {
  const headers = new Headers(options.headers || {});
  const anonId = getAnonId();
  if (anonId && !headers.has('X-DV-Anon')) {
    headers.set('X-DV-Anon', anonId);
  }
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  return headers;
}

/**
 * Plain fetch wrapper for non-hook contexts. Always anonymous
 * (attaches X-DV-Anon, never a bearer token).
 */
export function anonFetch(path, options = {}) {
  return fetch(path, { ...options, headers: buildHeaders(options) });
}

/**
 * Hook returning apiFetch(path, options). When Auth0 is active and the
 * user is authenticated, attaches Authorization: Bearer <access token>;
 * token errors are swallowed and the request falls back to anonymous.
 * Always attaches the X-DV-Anon device id header.
 */
export function useApi() {
  // Safe even when Auth0Provider is absent: the default context reports
  // isAuthenticated=false, and we never call its stub methods then.
  const { isAuthenticated, getAccessTokenSilently } = useAuth0();

  const apiFetch = useCallback(
    async (path, options = {}) => {
      let token = null;
      if (isAuth0Enabled && isAuthenticated) {
        try {
          token = await getAccessTokenSilently();
        } catch {
          // Consent required / refresh failed / offline — go anonymous.
          token = null;
        }
      }
      return fetch(path, { ...options, headers: buildHeaders(options, token) });
    },
    [isAuthenticated, getAccessTokenSilently]
  );

  return apiFetch;
}

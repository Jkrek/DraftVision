/*
 * Theme runtime — "Gameday Light" (default) / dark.
 *
 * public/index.html sets html[data-theme] BEFORE first paint (localStorage
 * 'dv_theme' → prefers-color-scheme → light); this module owns everything
 * after boot: reading the active theme, flipping it, persisting the choice,
 * and keeping the <meta name="theme-color"> in sync. Tokens themselves live
 * in src/nocturne.css.
 */

const STORAGE_KEY = 'dv_theme';

// Must match --color-bg in nocturne.css for each theme.
export const THEME_COLORS = { light: '#FAFAF7', dark: '#232A36' };

export function getTheme() {
  return document.documentElement.getAttribute('data-theme') === 'dark'
    ? 'dark'
    : 'light';
}

export function applyTheme(theme) {
  const t = theme === 'dark' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', t);
  try {
    localStorage.setItem(STORAGE_KEY, t);
  } catch (e) {
    /* private mode — theme still applies for this visit */
  }
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute('content', THEME_COLORS[t]);
  return t;
}

export function toggleTheme() {
  return applyTheme(getTheme() === 'dark' ? 'light' : 'dark');
}

import React, { useEffect, useState } from 'react';
import './GapBar.css';

/*
 * GapBar — the signature "model vs. market" glyph (frozen interface).
 *
 * A 0–100 rail. A hollow ring sits where the MARKET price is (yes-cents),
 * a solid accent disc where the MODEL-implied probability is, and a tinted
 * band spans the gap between them — the band's length IS the disagreement.
 *
 * Props
 *   market    0–100 cents | null
 *   model     0–100 percent | null      (null → ring only: "listed, not priced")
 *   source    'last'|'mid'|'mid_wide'|'ask'|'bid'  (dashed ring when untraded/one-sided)
 *   size      'hero'|'row'|'sm'
 *   labels    bool — '90¢' above the ring, '37%' below the disc (hero/row only)
 *   threshold default 10 — |gap| below this renders a neutral "they agree" band
 *   outcome   null|'yes'|'no'|'void'   (resolved ledger rows)
 *   ghost     cents | null — faint tick, e.g. the current price on a ledger row
 *   delay     ms before the mount animation starts
 *   ground    'paper' (page) | 'ink' (the dark home band; pinned colours)
 *
 * Direction is never encoded with green/red: ring = market, red disc = model,
 * learned once. Motion respects prefers-reduced-motion (final state at once).
 */

const clamp = (v) => Math.max(0, Math.min(100, Number(v)));
const REDUCED = typeof window !== 'undefined' && window.matchMedia
  && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const DASHED_SOURCES = new Set(['mid_wide', 'ask', 'bid']);

export default function GapBar({
  market = null,
  model = null,
  source = 'last',
  size = 'row',
  labels = false,
  threshold = 10,
  outcome = null,
  ghost = null,
  delay = 0,
  ground = 'paper',
  className = '',
  style,
}) {
  const [inView, setInView] = useState(REDUCED);
  useEffect(() => {
    if (REDUCED) return undefined;
    let raf;
    const t = setTimeout(() => { raf = requestAnimationFrame(() => setInView(true)); }, delay);
    return () => { clearTimeout(t); if (raf) cancelAnimationFrame(raf); };
  }, [delay]);

  const hasMarket = market != null && Number.isFinite(Number(market));
  const hasModel = model != null && Number.isFinite(Number(model));
  const m = hasMarket ? clamp(market) : null;
  const p = hasModel ? clamp(model) : null;
  const gap = hasMarket && hasModel ? p - m : null;

  let bandClass = 'gap-band';
  if (outcome) bandClass += ' gap-band-resolved';
  else if (gap == null) bandClass += ' gap-band-none';
  else if (Math.abs(gap) < threshold) bandClass += ' gap-band-agree';
  else if (gap > 0) bandClass += ' gap-band-model';
  else bandClass += ' gap-band-market';

  const left = gap == null ? (m ?? 50) : Math.min(m, p);
  const width = gap == null ? 0 : Math.abs(gap);
  const staged = !inView; // markers park at 50%, band collapsed, until .is-in

  const showLabels = labels && size !== 'sm';
  const lblStyle = (x) => ({ left: `clamp(0px, ${x}%, 100%)` });

  return (
    <div
      className={`gap gap-${size} gap-ground-${ground}${inView ? ' is-in' : ''}${className ? ` ${className}` : ''}`}
      style={style}
      role="img"
      aria-label={
        hasMarket && hasModel
          ? `Market ${Math.round(m)} cents, model ${Math.round(p)} percent, gap ${gap > 0 ? '+' : ''}${Math.round(gap)} points`
          : hasMarket ? `Market ${Math.round(m)} cents, model not priced` : 'No market'
      }
    >
      <div className="gap-track" aria-hidden="true">
        {size === 'hero' && (
          <>
            <span className="gap-end gap-end-0">0</span>
            <span className="gap-end gap-end-100">100</span>
          </>
        )}
      </div>

      {ghost != null && Number.isFinite(Number(ghost)) && (
        <span className="gap-ghost" style={{ left: `${clamp(ghost)}%` }} aria-hidden="true" />
      )}

      <span
        className={bandClass}
        style={{ left: `${staged ? (m ?? 50) : left}%`, width: `${staged ? 0 : width}%` }}
        aria-hidden="true"
      />

      {hasMarket && (
        <span
          className={`gap-market${DASHED_SOURCES.has(source) ? ' gap-market-dashed' : ''}`}
          style={{ left: `${staged ? 50 : m}%` }}
          aria-hidden="true"
        />
      )}
      {hasModel && (
        <span
          className="gap-model"
          style={{ left: `${staged ? 50 : p}%` }}
          aria-hidden="true"
        />
      )}

      {outcome === 'yes' && <span className="gap-outcome gap-outcome-yes" style={{ left: '100%' }} aria-hidden="true" />}
      {outcome === 'no' && <span className="gap-outcome gap-outcome-no" style={{ left: '0%' }} aria-hidden="true" />}

      {showLabels && hasMarket && (
        <span className="gap-lbl gap-lbl-market" style={lblStyle(m)} aria-hidden="true">
          {Math.round(m)}¢
        </span>
      )}
      {showLabels && hasModel && (
        <span className="gap-lbl gap-lbl-model" style={lblStyle(p)} aria-hidden="true">
          {Math.round(p)}%
        </span>
      )}
    </div>
  );
}

/* Shared caption helper — the "market higher / model higher / agree" word. */
export function gapWord(gap, threshold = 10) {
  if (gap == null || !Number.isFinite(gap)) return '';
  if (Math.abs(gap) < threshold) return 'agree';
  return gap > 0 ? 'model higher' : 'market higher';
}

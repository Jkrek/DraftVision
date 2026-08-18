import React from 'react';

/*
 * DraftVision mark — "the scouting eye".
 *
 * A single vesica (pointed ellipse) reads two ways at once: a football seen
 * side-on and the outline of an eye. Inside, where the pupil / lace line
 * would sit, four ticks rise left-to-right — football laces doubled as a
 * bar chart climbing a draft board. Bars step through the accent ramp
 * (600 -> 300) so the "rise" is tonal as well as spatial; the outline stays
 * neutral. No text, no gradients; reads at 24px.
 *
 * Variants:
 *   <LogoMark size={26} />          — full-color (navbar)
 *   <LogoMark size={22} muted />    — monochrome via currentColor (footer)
 */

const OUTLINE = 'M2.6 16 Q16 3.8 29.4 16 Q16 28.2 2.6 16 Z';

// Lace-bars: ascending ticks from a common baseline (y = 19.8).
const BARS = [
  { x: 10.8, top: 17.6, color: 'var(--color-accent-600)', mutedOpacity: 0.45 },
  { x: 14.3, top: 16.2, color: 'var(--color-accent-500)', mutedOpacity: 0.6 },
  { x: 17.8, top: 14.6, color: 'var(--color-accent-400)', mutedOpacity: 0.8 },
  { x: 21.3, top: 12.8, color: 'var(--color-accent-300)', mutedOpacity: 1 },
];

function LogoMark({ size = 26, muted = false, className }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      <path
        d={OUTLINE}
        stroke={muted ? 'currentColor' : 'var(--color-neutral-300)'}
        strokeWidth="1.7"
        strokeLinejoin="round"
        opacity={muted ? 0.9 : 1}
      />
      {BARS.map(({ x, top, color, mutedOpacity }) => (
        <line
          key={x}
          x1={x}
          y1="19.8"
          x2={x}
          y2={top}
          stroke={muted ? 'currentColor' : color}
          strokeWidth="2.2"
          strokeLinecap="round"
          opacity={muted ? mutedOpacity : 1}
        />
      ))}
    </svg>
  );
}

export default LogoMark;

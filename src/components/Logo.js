import React from 'react';

/*
 * JK Football brand mark — the owner's logo (red/black "JK" with the
 * helmeted figure), background-removed and served from public/brand/.
 *
 *   <LogoMark size={26} />          — full-color (navbar)
 *   <LogoMark size={22} muted />    — dimmed grayscale (footer)
 *
 * jk-mark-96.png is a 13KB 96px-tall render — crisp at any UI size we use
 * (retina included) without shipping the 600KB master.
 */

const SRC = process.env.PUBLIC_URL + '/brand/jk-mark-96.png';

// The art is slightly taller than wide (924x974); scale by height.
const ASPECT = 924 / 974;

/* Treatment lives in CSS (.logo-mark / .logo-mark--muted + their
   [data-theme="dark"] variants in nocturne.css) so each theme can tune it:
   the black K needs no halo on the light ground. */
function LogoMark({ size = 26, muted = false, className }) {
  return (
    <img
      className={['logo-mark', muted ? 'logo-mark--muted' : '', className].filter(Boolean).join(' ')}
      src={SRC}
      alt=""
      aria-hidden="true"
      height={size}
      width={Math.round(size * ASPECT)}
      style={{ display: 'block', objectFit: 'contain' }}
    />
  );
}

export default LogoMark;

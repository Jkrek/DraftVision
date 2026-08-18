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

function LogoMark({ size = 26, muted = false, className }) {
  return (
    <img
      className={className}
      src={SRC}
      alt=""
      aria-hidden="true"
      height={size}
      width={Math.round(size * ASPECT)}
      style={{
        display: 'block',
        objectFit: 'contain',
        filter: muted
          ? 'grayscale(1) brightness(1.6) opacity(0.55)'
          : 'drop-shadow(0 0 1px rgba(233, 233, 237, 0.2))',
      }}
    />
  );
}

export default LogoMark;

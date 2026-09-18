import React, { useEffect, useRef, useState } from 'react';

/* ──────────────────────────────────────────────────────────────────────────
   GenBurst — the Generational reveal's signature moment.

   One raw-WebGL canvas layered over the report hero: ~400 gold/red embers
   rising off the bottom edge with drift + sparkle, over a slow warm glow
   at bottom-center. Runs for ~4.5s, peaks with the grade stamp at 1.7s,
   then fades itself out and stops the rAF loop for good.

   It owns its own fallback: if the user prefers reduced motion or WebGL is
   unavailable / fails, it renders the original CSS `.gen-sparks` markup
   instead (which already respects prefers-reduced-motion in the CSS).
   ────────────────────────────────────────────────────────────────────────── */

const SPARK_COUNT = 14;          // CSS fallback spans
const PARTICLES = 400;           // desktop ember count
const PARTICLES_SMALL = 200;     // ≤640px viewports
const FLOATS_PER_PARTICLE = 6;   // seed, angle, speed, size, colorMix, lifeOffset
const STRIDE = FLOATS_PER_PARTICLE * 4; // bytes

const RUN_MS = 4500;             // total life of the effect
const PEAK_MS = 1700;            // matches --rv-stamp-delay for the gen tier
const FADE_MS = 700;             // canvas opacity fade at the tail
const GLOW_IN_MS = 1200;         // warm radial glow fade-in
const SMALL_VIEWPORT = 640;

/* ── Shaders ───────────────────────────────────────────────────────────── */

/* Embers. Every particle is one GL_POINT whose whole trajectory is derived
   from its static seed + u_time, so the attribute buffer is uploaded once
   and never touched again. */
const EMBER_VS = [
  'precision mediump float;',
  '',
  'attribute float a_seed;   // 0..1 spawn position across the card',
  'attribute float a_angle;  // per-particle phase (drift + sparkle)',
  'attribute float a_speed;  // 0..1 -> rise speed and lifetime',
  'attribute float a_size;   // base point size, CSS px (2..6)',
  'attribute float a_mix;    // 0 = gold, 1 = accent red',
  'attribute float a_life;   // 0..1 lifetime offset, so they stagger',
  '',
  'uniform float u_time;     // seconds since the burst started',
  'uniform vec2  u_res;      // drawing-buffer size, device px',
  'uniform float u_dpr;      // device pixel ratio (capped)',
  'uniform float u_master;   // 0..1 global envelope',
  '',
  'varying float v_alpha;',
  'varying float v_mix;',
  '',
  'void main() {',
  '  // lifetime 2.5 -> 3.5s; fract() gives us free respawn at the bottom',
  '  float span = 2.5 + a_speed;',
  '  float t = fract((u_time + a_life * span) / span);',
  '',
  '  // drift is authored in "screen width" terms, so correct for aspect',
  '  float aspect = u_res.x / max(u_res.y, 1.0);',
  '  float x = a_seed * 2.0 - 1.0;',
  '  x += sin(u_time * 0.8 + a_angle * 6.2831) * (0.22 / max(aspect, 0.2)) * t;',
  '  float y = -1.06 + t * (1.35 + a_speed * 0.75);',
  '  gl_Position = vec4(x, y, 0.0, 1.0);',
  '',
  '  // sparkle — fast per-particle flicker that never blinks fully out',
  '  float sparkle = 0.6 + 0.4 * sin(u_time * (7.0 + a_angle * 9.0) + a_angle * 21.0);',
  '  // snap in, then bleed away over the back half of the life',
  '  float fade = smoothstep(0.0, 0.10, t) * (1.0 - smoothstep(0.45, 1.0, t));',
  '',
  '  v_alpha = fade * sparkle * u_master;',
  '  v_mix = a_mix;',
  '  gl_PointSize = max(1.0, a_size * u_dpr * (0.75 + 0.25 * sparkle) * (1.0 - 0.25 * t));',
  '}',
].join('\n');

/* Soft disc: discard outside the unit radius, quadratic radial falloff.
   Colour is premultiplied by alpha because the canvas is premultiplied and
   the blend func is additive (ONE, ONE). */
const EMBER_FS = [
  'precision mediump float;',
  '',
  'varying float v_alpha;',
  'varying float v_mix;',
  '',
  'const vec3 GOLD = vec3(0.831, 0.647, 0.290);  // #D4A54A',
  'const vec3 RED  = vec3(0.894, 0.341, 0.310);  // #E4574F',
  '',
  'void main() {',
  '  vec2 d = gl_PointCoord - vec2(0.5);',
  '  float r2 = dot(d, d);',
  '  if (r2 > 0.25) discard;            // outside the disc',
  '  float falloff = 1.0 - r2 * 4.0;    // 1 at the core, 0 at the rim',
  '  falloff *= falloff;',
  '  float a = clamp(v_alpha, 0.0, 1.0) * falloff;',
  '  vec3 c = mix(GOLD, RED, clamp(v_mix, 0.0, 1.0));',
  '  gl_FragColor = vec4(c * a, a);     // premultiplied',
  '}',
].join('\n');

/* Glow: one full-bleed triangle strip, radial warmth anchored bottom-centre. */
const GLOW_VS = [
  'precision mediump float;',
  'attribute vec2 a_pos;',
  'varying vec2 v_uv;',
  'void main() {',
  '  v_uv = a_pos;',
  '  gl_Position = vec4(a_pos, 0.0, 1.0);',
  '}',
].join('\n');

const GLOW_FS = [
  'precision mediump float;',
  'varying vec2 v_uv;',
  'uniform vec2  u_res;',
  'uniform float u_glow;   // 0..1 fade-in x master envelope',
  '',
  'void main() {',
  '  // squash into an ellipse: wide and low, centred on the bottom edge',
  '  float aspect = u_res.x / max(u_res.y, 1.0);',
  '  vec2 q = vec2(v_uv.x * (0.55 + aspect * 0.18), (v_uv.y + 1.0) * 0.85);',
  '  float d = length(q);',
  '  float g = pow(max(0.0, 1.0 - d), 2.6) * clamp(u_glow, 0.0, 1.0) * 0.34;',
  '  vec3 warm = vec3(0.831, 0.596, 0.290);',
  '  gl_FragColor = vec4(warm * g, g);  // premultiplied, additive',
  '}',
].join('\n');

/* ── GL helpers ────────────────────────────────────────────────────────── */

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

function link(gl, vsSrc, fsSrc) {
  const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
  if (!vs || !fs) {
    if (vs) gl.deleteShader(vs);
    if (fs) gl.deleteShader(fs);
    return null;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  // shaders are owned by the program once attached+linked
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    gl.deleteProgram(prog);
    return null;
  }
  return prog;
}

function buildParticles(count) {
  const data = new Float32Array(count * FLOATS_PER_PARTICLE);
  for (let i = 0; i < count; i += 1) {
    const o = i * FLOATS_PER_PARTICLE;
    data[o] = Math.random();                       // seed — spawn x
    data[o + 1] = Math.random();                   // angle — drift/sparkle phase
    data[o + 2] = Math.random();                   // speed — rise + lifetime
    data[o + 3] = 2 + Math.random() * 4;           // size — 2..6 CSS px
    // ~1 in 4 burns red, the rest gold
    data[o + 4] = Math.random() < 0.26
      ? 0.65 + Math.random() * 0.35
      : Math.random() * 0.22;                      // colorMix
    data[o + 5] = Math.random();                   // lifeOffset
  }
  return data;
}

/* Global envelope: ramps in, peaks exactly on the grade stamp (1.7s),
   settles, then bleeds out over the last stretch of the run. */
function envelope(ms) {
  const ramp = Math.min(1, Math.max(0, (ms - 120) / (PEAK_MS - 120)));
  const rise = ramp * ramp * (3 - 2 * ramp); // smoothstep
  const tailStart = RUN_MS - FADE_MS - 500;
  const tail = ms <= tailStart
    ? 1
    : Math.max(0, 1 - (ms - tailStart) / (RUN_MS - tailStart));
  const k = (ms - PEAK_MS) / 320;
  const bump = 1 + 0.3 * Math.exp(-(k * k)); // a flare right on the stamp
  return rise * tail * bump;
}

function supportsBurst() {
  if (typeof window === 'undefined' || typeof document === 'undefined') return false;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    return false; // the CSS sparks path already honours reduced motion
  }
  if (typeof window.WebGLRenderingContext === 'undefined') return false;
  return true;
}

/* ── Component ─────────────────────────────────────────────────────────── */

export default function GenBurst() {
  const canvasRef = useRef(null);
  const [gpu, setGpu] = useState(supportsBurst);

  useEffect(() => {
    if (!gpu) return undefined;
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const attrs = {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'low-power',
      failIfMajorPerformanceCaveat: false,
    };
    let gl = null;
    try {
      gl = canvas.getContext('webgl', attrs) || canvas.getContext('experimental-webgl', attrs);
    } catch (e) {
      gl = null;
    }
    if (!gl) { setGpu(false); return undefined; }

    const small = window.innerWidth <= SMALL_VIEWPORT;
    const count = small ? PARTICLES_SMALL : PARTICLES;
    const dpr = small ? 1 : Math.min(window.devicePixelRatio || 1, 1.5);

    const emberProg = link(gl, EMBER_VS, EMBER_FS);
    const glowProg = link(gl, GLOW_VS, GLOW_FS);
    if (!emberProg || !glowProg) {
      if (emberProg) gl.deleteProgram(emberProg);
      if (glowProg) gl.deleteProgram(glowProg);
      const lost = gl.getExtension('WEBGL_lose_context');
      if (lost) lost.loseContext();
      setGpu(false);
      return undefined;
    }

    // ── buffers: one interleaved particle buffer + one full-bleed quad ──
    const emberBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, emberBuf);
    gl.bufferData(gl.ARRAY_BUFFER, buildParticles(count), gl.STATIC_DRAW);

    const quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );

    const eLoc = {
      seed: gl.getAttribLocation(emberProg, 'a_seed'),
      angle: gl.getAttribLocation(emberProg, 'a_angle'),
      speed: gl.getAttribLocation(emberProg, 'a_speed'),
      size: gl.getAttribLocation(emberProg, 'a_size'),
      mix: gl.getAttribLocation(emberProg, 'a_mix'),
      life: gl.getAttribLocation(emberProg, 'a_life'),
      uTime: gl.getUniformLocation(emberProg, 'u_time'),
      uRes: gl.getUniformLocation(emberProg, 'u_res'),
      uDpr: gl.getUniformLocation(emberProg, 'u_dpr'),
      uMaster: gl.getUniformLocation(emberProg, 'u_master'),
    };
    const gLoc = {
      pos: gl.getAttribLocation(glowProg, 'a_pos'),
      uRes: gl.getUniformLocation(glowProg, 'u_res'),
      uGlow: gl.getUniformLocation(glowProg, 'u_glow'),
    };

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);           // additive, source is premultiplied
    gl.clearColor(0, 0, 0, 0);

    let width = 0;
    let height = 0;
    const resize = () => {
      const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (w === width && h === height) return;
      width = w;
      height = h;
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
    };
    resize();

    // ── loop ──
    const t0 = performance.now();           // mount time == reveal time
    let pausedFor = 0;                      // ms spent with the tab hidden
    let hiddenAt = 0;
    let raf = 0;
    let idle = 0;
    let dead = false;
    let faded = false;

    const stop = () => {
      dead = true;
      if (raf) { cancelAnimationFrame(raf); raf = 0; }
    };

    const frame = () => {
      raf = 0;
      if (dead) return;
      if (document.hidden) { hiddenAt = hiddenAt || performance.now(); return; }

      const ms = performance.now() - t0 - pausedFor;
      if (ms >= RUN_MS) { stop(); return; }

      if (!faded && ms >= RUN_MS - FADE_MS) {
        faded = true;
        canvas.style.opacity = '0';         // CSS transitions it out
      }

      // layout is measured on mount + on window resize only; this is just a
      // guard for the case where the card had no box yet at mount
      if (width <= 1 || height <= 1) resize();

      const master = envelope(ms);
      const t = ms / 1000;

      gl.clear(gl.COLOR_BUFFER_BIT);

      // warm glow first — it sits under the embers (additive, so order is
      // cosmetic, but it keeps the intent readable)
      gl.useProgram(glowProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
      gl.enableVertexAttribArray(gLoc.pos);
      gl.vertexAttribPointer(gLoc.pos, 2, gl.FLOAT, false, 0, 0);
      gl.uniform2f(gLoc.uRes, width, height);
      gl.uniform1f(gLoc.uGlow, Math.min(1, ms / GLOW_IN_MS) * Math.min(1, master));
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.disableVertexAttribArray(gLoc.pos);

      gl.useProgram(emberProg);
      gl.bindBuffer(gl.ARRAY_BUFFER, emberBuf);
      const attrs6 = [eLoc.seed, eLoc.angle, eLoc.speed, eLoc.size, eLoc.mix, eLoc.life];
      for (let i = 0; i < attrs6.length; i += 1) {
        const loc = attrs6[i];
        if (loc >= 0) {
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, STRIDE, i * 4);
        }
      }
      gl.uniform1f(eLoc.uTime, t);
      gl.uniform2f(eLoc.uRes, width, height);
      gl.uniform1f(eLoc.uDpr, dpr);
      gl.uniform1f(eLoc.uMaster, master);
      gl.drawArrays(gl.POINTS, 0, count);
      for (let i = 0; i < attrs6.length; i += 1) {
        if (attrs6[i] >= 0) gl.disableVertexAttribArray(attrs6[i]);
      }

      raf = requestAnimationFrame(frame);
    };

    const onVisibility = () => {
      if (dead) return;
      if (document.hidden) {
        hiddenAt = hiddenAt || performance.now();
        if (raf) { cancelAnimationFrame(raf); raf = 0; }
      } else {
        if (hiddenAt) { pausedFor += performance.now() - hiddenAt; hiddenAt = 0; }
        if (!raf) raf = requestAnimationFrame(frame);
      }
    };

    const onLost = (e) => {
      e.preventDefault();
      stop();
      canvas.style.opacity = '0';
    };

    const onResize = () => { if (!dead) resize(); };

    canvas.addEventListener('webglcontextlost', onLost);
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('resize', onResize);

    // start deferred so the reveal's first paint is never blocked; the clock
    // is anchored to mount (t0), so the peak still lands on the 1.7s stamp
    const start = () => { idle = 0; if (!dead && !raf) raf = requestAnimationFrame(frame); };
    if (typeof window.requestIdleCallback === 'function') {
      idle = window.requestIdleCallback(start, { timeout: 120 });
    } else {
      idle = window.setTimeout(start, 0);
    }

    return () => {
      dead = true;
      if (raf) cancelAnimationFrame(raf);
      if (idle) {
        if (typeof window.cancelIdleCallback === 'function') window.cancelIdleCallback(idle);
        else window.clearTimeout(idle);
      }
      canvas.removeEventListener('webglcontextlost', onLost);
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('resize', onResize);
      gl.deleteBuffer(emberBuf);
      gl.deleteBuffer(quadBuf);
      gl.deleteProgram(emberProg);
      gl.deleteProgram(glowProg);
      const lose = gl.getExtension('WEBGL_lose_context');
      if (lose) lose.loseContext();
    };
  }, [gpu]);

  if (!gpu) {
    // Fallback: the original CSS sparks (already reduced-motion aware).
    return (
      <div className="gen-sparks" aria-hidden="true">
        {Array.from({ length: SPARK_COUNT }).map((_, i) => (
          <span key={i} className="gen-spark" style={{ '--i': i }} />
        ))}
      </div>
    );
  }

  return <canvas ref={canvasRef} className="gen-burst" aria-hidden="true" />;
}

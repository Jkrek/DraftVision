import React, { useEffect, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import './Ambient.css';

/*
 * Ambient — the GLSL "stadium lights" ground behind every tab.
 *
 * The research finding: a shader background earns its place only when it is
 * part of the identity, not decoration. DraftVision's identity is night
 * football, so this is not a space nebula — it is slow plumes of stadium
 * light in JK red and warm gold drifting through the dark, at an intensity
 * low enough that tables stay tables.
 *
 * Guardrails (all mandatory):
 *   - renders at 0.5× (0.35× on phones) of a DPR capped at 1.5, upscaled by CSS
 *   - frame-capped at 30 fps; pauses when the tab is hidden
 *   - prefers-reduced-motion → one static frame, no loop
 *   - no WebGL → a CSS radial-gradient fallback (Ambient.css)
 *   - deferred start (requestIdleCallback), context-loss handled, full cleanup
 *   - off on "/" (the home hero owns its ground) and when the light theme is on
 */

const REDUCED = typeof window !== 'undefined' && window.matchMedia
  && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const VS = [
  'attribute vec2 a_pos;',
  'void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }',
].join('\n');

// value-noise fbm; three drifting light fields; premultiplied additive output
const FS = [
  'precision mediump float;',
  'uniform vec2 u_res;',
  'uniform float u_time;',
  'uniform float u_gain;',
  'float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }',
  'float noise(vec2 p) {',
  '  vec2 i = floor(p), f = fract(p);',
  '  f = f * f * (3.0 - 2.0 * f);',
  '  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),',
  '             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);',
  '}',
  'float fbm(vec2 p) {',
  '  float v = 0.0, a = 0.5;',
  '  for (int i = 0; i < 4; i++) { v += a * noise(p); p = p * 2.03 + 17.0; a *= 0.5; }',
  '  return v;',
  '}',
  'void main() {',
  '  vec2 uv = gl_FragCoord.xy / u_res;',
  '  vec2 p = vec2(uv.x * u_res.x / u_res.y, uv.y);',
  '  float t = u_time * 0.035;',
  // red plume, low-left, rising
  '  float r = fbm(p * 1.6 + vec2(-t * 0.9, -t * 0.5));',
  '  r = smoothstep(0.42, 0.82, r) * smoothstep(1.35, -0.1, length(p - vec2(-0.05, 0.02)) );',
  // gold plume, upper-right, falling
  '  float g = fbm(p * 1.3 + vec2(t * 0.7, t * 0.35) + 5.0);',
  '  g = smoothstep(0.45, 0.85, g) * smoothstep(1.25, -0.1, length(p - vec2(1.85, 1.0)) );',
  // faint cool ink drift through the middle keeps the dark from going flat
  '  float b = fbm(p * 0.9 + vec2(t * 0.3, -t * 0.4) + 11.0);',
  '  b = smoothstep(0.5, 0.9, b) * 0.6;',
  '  vec3 red  = vec3(0.89, 0.34, 0.31);',
  '  vec3 gold = vec3(0.83, 0.65, 0.29);',
  '  vec3 ink  = vec3(0.36, 0.44, 0.62);',
  // content column mask: the light lives in the margins, not under the tables
  '  float cx = (u_res.x / u_res.y) * 0.5;',
  '  float edge = smoothstep(0.18, 0.62, abs(p.x - cx));',
  '  vec3 c = (red * r * 0.40 + gold * g * 0.34) * (0.35 + 0.65 * edge) + ink * b * 0.16;',
  // grain kills banding on the long gradients
  '  c += (hash(gl_FragCoord.xy + u_time) - 0.5) * 0.02;',
  '  float a = clamp(max(max(c.r, c.g), c.b), 0.0, 1.0) * u_gain;',
  '  gl_FragColor = vec4(c * u_gain, a);',
  '}',
].join('\n');

function isDark() {
  return document.documentElement.getAttribute('data-theme') === 'dark';
}

export default function Ambient() {
  const { pathname } = useLocation();
  const canvasRef = useRef(null);
  const [gpu, setGpu] = useState(true);
  const [dark, setDark] = useState(isDark);
  const active = pathname !== '/' && dark;

  // follow the theme toggle (Navbar sets data-theme on <html>)
  useEffect(() => {
    const mo = new MutationObserver(() => setDark(isDark()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => mo.disconnect();
  }, []);

  // page roots go translucent only while the layer is live (see Ambient.css)
  useEffect(() => {
    const root = document.documentElement;
    if (active) root.setAttribute('data-ambient', 'on');
    else root.removeAttribute('data-ambient');
    return () => root.removeAttribute('data-ambient');
  }, [active]);

  useEffect(() => {
    if (!active || !gpu) return undefined;
    const canvas = canvasRef.current;
    if (!canvas || typeof window.WebGLRenderingContext === 'undefined') { setGpu(false); return undefined; }
    let gl = null;
    try {
      gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false })
        || canvas.getContext('experimental-webgl', { alpha: true, premultipliedAlpha: true });
    } catch (e) { gl = null; }
    if (!gl) { setGpu(false); return undefined; }

    const compile = (type, src) => {
      const sh = gl.createShader(type);
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) { gl.deleteShader(sh); return null; }
      return sh;
    };
    const vs = compile(gl.VERTEX_SHADER, VS);
    const fs = compile(gl.FRAGMENT_SHADER, FS);
    if (!vs || !fs) { setGpu(false); return undefined; }
    const prog = gl.createProgram();
    gl.attachShader(prog, vs); gl.attachShader(prog, fs); gl.linkProgram(prog);
    gl.deleteShader(vs); gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { gl.deleteProgram(prog); setGpu(false); return undefined; }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    const uRes = gl.getUniformLocation(prog, 'u_res');
    const uTime = gl.getUniformLocation(prog, 'u_time');
    const uGain = gl.getUniformLocation(prog, 'u_gain');
    gl.disable(gl.DEPTH_TEST);
    gl.clearColor(0, 0, 0, 0);

    const phone = window.innerWidth <= 640;
    const scale = phone ? 0.35 : 0.5;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const size = () => {
      const w = Math.max(1, Math.round(window.innerWidth * dpr * scale));
      const h = Math.max(1, Math.round(window.innerHeight * dpr * scale));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      gl.viewport(0, 0, w, h);
      gl.uniform2f(uRes, w, h);
    };
    size();
    gl.uniform1f(uGain, phone ? 1.1 : 1.7);

    let raf = 0, last = 0, lost = false, idle = 0;
    const t0 = performance.now();
    const FRAME = 1000 / 30;
    const draw = (now) => {
      gl.uniform1f(uTime, (now - t0) / 1000);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };
    const loop = (now) => {
      if (lost || document.hidden) { raf = 0; return; }
      raf = requestAnimationFrame(loop);
      if (now - last < FRAME) return;
      last = now;
      draw(now);
    };
    const start = () => {
      if (lost) return;
      if (REDUCED) { draw(t0 + 40000); return; } // one still frame, well into the drift
      if (!raf) raf = requestAnimationFrame(loop);
    };
    const onVis = () => { if (!document.hidden && !REDUCED) start(); };
    const onResize = () => { size(); if (REDUCED) draw(t0 + 40000); };
    const onLost = (e) => { e.preventDefault(); lost = true; if (raf) cancelAnimationFrame(raf); raf = 0; };
    canvas.addEventListener('webglcontextlost', onLost, false);
    document.addEventListener('visibilitychange', onVis);
    window.addEventListener('resize', onResize);
    if ('requestIdleCallback' in window) idle = window.requestIdleCallback(start, { timeout: 200 });
    else idle = window.setTimeout(start, 0);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      if ('cancelIdleCallback' in window) window.cancelIdleCallback(idle); else window.clearTimeout(idle);
      canvas.removeEventListener('webglcontextlost', onLost);
      document.removeEventListener('visibilitychange', onVis);
      window.removeEventListener('resize', onResize);
      try {
        gl.deleteBuffer(buf); gl.deleteProgram(prog);
        const ext = gl.getExtension('WEBGL_lose_context');
        if (ext) ext.loseContext();
      } catch (e) { /* context already gone */ }
    };
  }, [active, gpu]);

  if (!active) return null;
  return gpu
    ? <canvas ref={canvasRef} className="ambient" aria-hidden="true" />
    : <div className="ambient ambient-fallback" aria-hidden="true" />;
}

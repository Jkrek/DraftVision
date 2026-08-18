import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { anonFetch } from '../lib/api';
import './HeroSection.css';

/* Static fallback so the "Live model output" card never looks broken. */
const FALLBACK_PLAYER = {
  name: 'Caden Story',
  pos: 'DL',
  team: 'Clemson',
  prob: 90.5,
  factors: [
    { label: 'Production score', value: '95.5' },
    { label: 'Conference tier', value: '1 of 10' },
    { label: 'Combine speed', value: '54' },
    { label: 'Projection', value: 'Top 50 Pick' },
  ],
};

const STATS = [
  { value: '9,033', label: 'FBS prospects' },
  { value: '3', label: 'Model ensemble' },
  { value: '16', label: 'Engineered features' },
  { value: 'Live', label: 'ESPN + CFBD data' },
];

/* Map an /api/prospects row (name, position, team, success_probability,
   production_score, conference_tier, combine_speed_score, draft_grade)
   into the shape the card renders. */
function toHeroPlayer(p) {
  if (!p || typeof p !== 'object' || !p.name) return null;
  const prob = p.success_probability ?? p.probability;
  if (typeof prob !== 'number') return null;

  const factors = [];
  if (p.production_score != null) {
    factors.push({ label: 'Production score', value: Number(p.production_score).toFixed(1) });
  }
  if (p.conference_tier != null) {
    factors.push({ label: 'Conference tier', value: `${p.conference_tier} of 10` });
  }
  if (p.combine_speed_score != null) {
    factors.push({ label: 'Combine speed', value: String(p.combine_speed_score) });
  }
  if (p.draft_grade) {
    factors.push({ label: 'Projection', value: p.draft_grade });
  }

  return {
    name: p.name,
    pos: p.position || '',
    team: p.team || '',
    prob,
    factors: factors.length ? factors.slice(0, 4) : FALLBACK_PLAYER.factors,
  };
}

/* Gates for the background-video enhancement: big screens, motion OK,
   and no data-saver / slow connection. The Ken-Burns photo is always the
   base layer, so failing any gate simply means "photo only". */
function videoGatesPass() {
  if (typeof window === 'undefined') return false;
  if (window.innerWidth < 900) return false;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false;
  const conn = navigator.connection;
  if (conn) {
    if (conn.saveData === true) return false;
    const et = conn.effectiveType;
    if (et === 'slow-2g' || et === '2g' || et === '3g') return false;
  }
  return true;
}

function safePlay(video) {
  const p = video.play();
  if (p && typeof p.catch === 'function') p.catch(() => { /* autoplay blocked — poster/photo stays */ });
}

function HeroSection() {
  const [player, setPlayer] = useState(FALLBACK_PLAYER);
  const [showVideo, setShowVideo] = useState(false); // mount <video> at all
  const [videoReady, setVideoReady] = useState(false); // 'canplay' fired → fade in
  const heroRef = useRef(null);
  const videoRef = useRef(null);

  /* Defer the video decision until after window load + an idle slice, so it
     never competes with first paint or data fetches. */
  useEffect(() => {
    let cancelled = false;
    let idleId = null;
    let timerId = null;

    const decide = () => {
      if (!cancelled && videoGatesPass()) setShowVideo(true);
    };
    const whenIdle = () => {
      if (cancelled) return;
      if (typeof window.requestIdleCallback === 'function') {
        idleId = window.requestIdleCallback(decide, { timeout: 3000 });
      } else {
        timerId = setTimeout(decide, 1200);
      }
    };

    if (document.readyState === 'complete') {
      whenIdle();
    } else {
      window.addEventListener('load', whenIdle, { once: true });
    }

    return () => {
      cancelled = true;
      window.removeEventListener('load', whenIdle);
      if (idleId != null && typeof window.cancelIdleCallback === 'function') {
        window.cancelIdleCallback(idleId);
      }
      if (timerId != null) clearTimeout(timerId);
    };
  }, []);

  /* Once the element exists, set src (never present in initial HTML),
     and fade it in over the photo when it can actually play. */
  useEffect(() => {
    if (!showVideo) return undefined;
    const v = videoRef.current;
    if (!v) return undefined;

    const onCanPlay = () => {
      setVideoReady(true);
      safePlay(v);
    };
    v.addEventListener('canplay', onCanPlay, { once: true });
    v.src = process.env.PUBLIC_URL + '/videos/hero-bg.mp4';
    v.load();

    return () => v.removeEventListener('canplay', onCanPlay);
  }, [showVideo]);

  /* Pause when the hero scrolls out of view; resume on re-entry. */
  useEffect(() => {
    if (!videoReady) return undefined;
    const host = heroRef.current;
    const v = videoRef.current;
    if (!host || !v || typeof IntersectionObserver !== 'function') return undefined;

    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) safePlay(v);
        else v.pause();
      },
      { threshold: 0.05 }
    );
    io.observe(host);
    return () => io.disconnect();
  }, [videoReady]);

  useEffect(() => {
    let cancelled = false;
    // Feature the household name, not whoever tops the raw model board —
    // the hero card is a storefront. Fall back to the board's #1, then to
    // the static card.
    (async () => {
      try {
        const r = await anonFetch('/api/prospects?q=jeremiah%20smith&limit=5');
        if (r.ok) {
          const data = await r.json();
          const hit = (data?.prospects || []).find(
            (p) => p.name === 'Jeremiah Smith' && /ohio state/i.test(p.team || '')
          );
          const mapped = toHeroPlayer(hit);
          if (mapped && !cancelled) { setPlayer(mapped); return; }
        }
      } catch { /* fall through */ }
      try {
        const r = await anonFetch('/api/prospects?limit=2');
        const data = await r.json();
        const list = Array.isArray(data?.prospects) ? data.prospects : [];
        const mapped = toHeroPlayer(list[0]);
        if (mapped && !cancelled) setPlayer(mapped);
      } catch { /* keep the static fallback */ }
    })();
    return () => { cancelled = true; };
  }, []);

  return (
    <section className={videoReady ? 'hero has-video' : 'hero'} ref={heroRef}>
      <div className="hero-media" aria-hidden="true">
        <img
          src={process.env.PUBLIC_URL + '/images/CFB Content/action-1.jpg'}
          alt=""
        />
        {showVideo && (
          <video
            ref={videoRef}
            className={videoReady ? 'hero-video is-ready' : 'hero-video'}
            muted
            loop
            playsInline
            preload="none"
            poster={process.env.PUBLIC_URL + '/videos/hero-poster.jpg'}
            tabIndex={-1}
          />
        )}
      </div>
      <div className="hero-tint" aria-hidden="true" />
      <div className="hero-scrim-v" aria-hidden="true" />
      <div className="hero-scrim-h" aria-hidden="true" />

      <div className="hero-inner">
        <div className="hero-copy">
          <div className="hero-eyebrow">
            <span className="hero-eyebrow-dot" aria-hidden="true" />
            <span className="hero-eyebrow-text">Machine-learned draft intelligence</span>
          </div>
          <h1 className="hero-title">
            Draft<span className="hero-title-accent">Vision</span>
          </h1>
          <div className="hero-divider" aria-hidden="true" />
          <p className="hero-lede">
            A three-model ensemble scores every FBS prospect on production,
            athleticism, conference tier and accolades — then tells you what the
            number is made of.
          </p>
          <p className="hero-meta">
            9,033 college prospects · 10,000+ high-school recruits · live ESPN and CFBD data
          </p>
          <div className="hero-ctas">
            <Link to="/predict" className="dv-cta">Run a prediction</Link>
            <Link to="/leaderboard" className="dv-cta dv-cta-ghost">Browse the board</Link>
          </div>
        </div>

        <div className="hero-cardwrap">
          <div className="hero-card">
            <div className="hero-card-head">
              <span className="hero-card-kicker">Live model output</span>
              <span className="hero-card-version">ensemble v3</span>
            </div>
            <div className="hero-card-name">{player.name}</div>
            <div className="hero-card-meta">
              {player.pos} · {player.team}
            </div>
            <div className="hero-card-prob">
              <span className="hero-card-prob-num">{player.prob.toFixed(1)}</span>
              <span className="hero-card-prob-label">% success probability</span>
            </div>
            <div className="hero-card-track" aria-hidden="true">
              <div
                className="hero-card-fill"
                style={{ width: `${Math.min(100, Math.max(0, player.prob))}%` }}
              />
            </div>
            {player.factors.map((f) => (
              <div className="hero-card-factor" key={f.label}>
                <span className="hero-card-factor-label">{f.label}</span>
                <span className="hero-card-factor-value">{f.value}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="hero-stats">
          {STATS.map((s) => (
            <div className="hero-stat" key={s.label}>
              <div className="hero-stat-value">{s.value}</div>
              <div className="hero-stat-label">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default HeroSection;

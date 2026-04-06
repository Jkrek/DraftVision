import React, { useState } from 'react';
import '../../App.css';

export default function SignUp() {
  const [form, setForm] = useState({ name: '', email: '', role: 'fan' });
  const [submitted, setSubmitted] = useState(false);

  const backgroundStyle = {
    backgroundImage: `linear-gradient(135deg, rgba(11,19,43,0.88), rgba(14,184,166,0.35), rgba(11,110,124,0.88)), url(${process.env.PUBLIC_URL}/images/Top-NFL-Players.jpeg)`,
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    minHeight: '100vh',
    paddingTop: '80px',
    paddingBottom: '60px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const glassCard = {
    background: 'rgba(15,23,42,0.88)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: '16px',
    backdropFilter: 'blur(16px)',
    padding: '2.5rem',
    maxWidth: '480px',
    width: '100%',
    margin: '0 20px',
  };

  const inputStyle = {
    width: '100%', padding: '11px 14px', borderRadius: '8px',
    border: '1px solid rgba(255,255,255,0.12)', background: 'rgba(255,255,255,0.05)',
    color: '#e2e8f0', fontSize: '14px', outline: 'none', boxSizing: 'border-box',
  };

  const labelStyle = { color: '#94a3b8', fontSize: '13px', display: 'block', marginBottom: '6px' };

  const handleChange = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  const handleSubmit = e => {
    e.preventDefault();
    if (form.name && form.email) setSubmitted(true);
  };

  return (
    <div style={backgroundStyle}>
      <div style={glassCard}>
        {submitted ? (
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>✅</div>
            <h2 style={{ color: '#22c55e', fontWeight: 800, margin: '0 0 0.5rem' }}>You're on the list!</h2>
            <p style={{ color: '#94a3b8' }}>
              Welcome, <strong style={{ color: '#e2e8f0' }}>{form.name}</strong>. We'll send DraftVision updates to <strong style={{ color: '#e2e8f0' }}>{form.email}</strong>.
            </p>
          </div>
        ) : (
          <>
            <h2 style={{ color: '#f1f5f9', fontWeight: 800, fontSize: '1.8rem', margin: '0 0 0.4rem' }}>
              Join DraftVision
            </h2>
            <p style={{ color: '#94a3b8', marginBottom: '1.75rem', fontSize: '0.95rem' }}>
              Get early access to new features and prospect insights
            </p>

            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label style={labelStyle}>Full Name</label>
                <input name="name" value={form.name} onChange={handleChange} placeholder="Jared Krekeler" required style={inputStyle} />
              </div>
              <div>
                <label style={labelStyle}>Email Address</label>
                <input name="email" type="email" value={form.email} onChange={handleChange} placeholder="you@example.com" required style={inputStyle} />
              </div>
              <div>
                <label style={labelStyle}>I am a…</label>
                <select name="role" value={form.role} onChange={handleChange} style={{ ...inputStyle, cursor: 'pointer' }}>
                  <option value="fan">Football Fan</option>
                  <option value="analyst">Scout / Analyst</option>
                  <option value="coach">Coach / GM</option>
                  <option value="media">Media / Journalist</option>
                  <option value="dev">Developer</option>
                </select>
              </div>
              <button
                type="submit"
                style={{
                  marginTop: '0.5rem', padding: '12px', borderRadius: '8px', fontWeight: 700,
                  fontSize: '15px', border: 'none', background: '#3b82f6', color: '#fff', cursor: 'pointer',
                }}
              >
                Get Early Access
              </button>
            </form>

            <p style={{ color: '#475569', fontSize: '12px', textAlign: 'center', marginTop: '1rem' }}>
              No spam. Unsubscribe anytime.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

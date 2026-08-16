import React, { useState } from 'react';
import './SignUp.css';

export default function SignUp() {
  const [form, setForm] = useState({ name: '', email: '', role: 'fan' });
  const [submitted, setSubmitted] = useState(false);

  const handleChange = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  const handleSubmit = e => {
    e.preventDefault();
    if (form.name && form.email) setSubmitted(true);
  };

  return (
    <div className="signup-page">
      <div className="signup-panel">
        {submitted ? (
          <div className="signup-success">
            <div className="signup-success-mark" aria-hidden="true">✓</div>
            <h2>You're on the list</h2>
            <p className="signup-sub">
              Welcome, <strong>{form.name}</strong>. We'll send DraftVision updates
              to <strong>{form.email}</strong>.
            </p>
          </div>
        ) : (
          <>
            <div className="signup-eyebrow">Early access</div>
            <h2>Join DraftVision</h2>
            <p className="signup-sub">Get early access to new features and prospect insights.</p>

            <form onSubmit={handleSubmit} className="signup-form">
              <div className="field">
                <label htmlFor="signup-name">Full name</label>
                <input
                  id="signup-name"
                  className="input"
                  name="name"
                  value={form.name}
                  onChange={handleChange}
                  placeholder="Jared Krekeler"
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="signup-email">Email address</label>
                <input
                  id="signup-email"
                  className="input"
                  name="email"
                  type="email"
                  value={form.email}
                  onChange={handleChange}
                  placeholder="you@example.com"
                  required
                />
              </div>
              <div className="field">
                <label htmlFor="signup-role">I am a…</label>
                <select
                  id="signup-role"
                  className="input signup-select"
                  name="role"
                  value={form.role}
                  onChange={handleChange}
                >
                  <option value="fan">Football Fan</option>
                  <option value="analyst">Scout / Analyst</option>
                  <option value="coach">Coach / GM</option>
                  <option value="media">Media / Journalist</option>
                  <option value="dev">Developer</option>
                </select>
              </div>
              <button type="submit" className="btn btn-primary btn-block">
                Get early access
              </button>
            </form>

            <p className="signup-fineprint">No spam. Unsubscribe anytime.</p>
          </>
        )}
      </div>
    </div>
  );
}

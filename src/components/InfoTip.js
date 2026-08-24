import React from 'react';
import './InfoTip.css';

/*
 * Tiny "i" info icon — hover (or tap/focus on mobile) reveals a plain-language
 * explanation of how a metric is actually calculated.
 *
 *   <InfoTip text="…" />                 — tooltip opens above, centered
 *   <InfoTip text="…" place="bottom" />  — opens below (for labels near a
 *                                          clipped card top)
 *   place: top | bottom | top-left | bottom-left
 *   (-left variants right-align the bubble so it stays inside cards that
 *    sit against the viewport/card edge)
 */
export default function InfoTip({ text, place = 'top' }) {
  return (
    <span className={`itip itip--${place}`}>
      <button
        type="button"
        className="itip-btn"
        aria-label="What does this metric mean?"
        onClick={(e) => e.currentTarget.focus()}
      >
        i
      </button>
      <span role="tooltip" className="itip-pop">{text}</span>
    </span>
  );
}

import React from 'react';
import { Link } from 'react-router-dom';

/* One "Top of the board" prospect cell — rank, grade pill, name,
   pos · team, and a big weight-300 probability numeral. */
function CardItem({ rank, grade, name, pos, team, prob }) {
  return (
    <Link className="board-cell" to="/leaderboard">
      <div className="board-cell-top">
        <span className="board-cell-rank">{rank}</span>
        {grade && <span className="board-cell-grade">{grade}</span>}
      </div>
      <div className="board-cell-name">{name}</div>
      <div className="board-cell-meta">
        {pos} · {team}
      </div>
      <div className="board-cell-prob">
        <span className="board-cell-prob-num">{prob}</span>
        <span className="board-cell-prob-label">% success</span>
      </div>
    </Link>
  );
}

export default CardItem;

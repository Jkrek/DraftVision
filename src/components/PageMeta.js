import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { trackPageView } from '../lib/api';

/*
 * PageMeta — per-route <title> + description, and one page-view beacon per
 * navigation. Every route used to share one title, so nothing but the home
 * page had anything distinct for a search result or a shared link to show.
 */

const SITE = 'DraftVision';

const META = {
  '/': ['DraftVision — a machine-learned NFL draft board',
    'Every FBS prospect graded weekly by a model that publishes its own track record.'],
  '/predict': ['Predict any prospect',
    'Run the model on any FBS player: success odds, projected pick, comps and the reasons behind the number.'],
  '/leaderboard': ['Model Board — every FBS prospect ranked',
    'The model’s full board: 15,000 college players graded and ranked, refreshed every Tuesday.'],
  '/big-board': ['2027 NFL Draft Big Board',
    'Class-by-class draft boards: the model’s ranking, a curated board, and one you build yourself.'],
  '/mock-draft': ['2027 NFL Mock Draft simulator',
    'Play out draft night pick by pick against the model’s projections.'],
  '/hs-prospects': ['High-school recruiting classes',
    'Composite-ranked recruiting classes with commitments, the first step of the draft pipeline.'],
  '/services': ['College Stars', 'The college players the model rates highest right now.'],
  '/compare': ['Compare two prospects',
    'Two prospects head to head: probabilities, the factors behind them and historical comps.'],
  '/futures': ['2027 NFL Draft futures — model vs. market',
    'Kalshi’s 2027 draft markets lined up against the model, with every disagreement recorded and scored after the draft.'],
  '/backtest': ['Backtest — how the model did on drafts it never saw',
    'The model scored against the 2019–20 drafts and the consensus big board, every row downloadable.'],
};

function setDescription(text) {
  const el = document.querySelector('meta[name="description"]');
  if (el && text) el.setAttribute('content', text);
}

function PageMeta() {
  const { pathname } = useLocation();

  useEffect(() => {
    const m = META[pathname];
    if (m) {
      document.title = pathname === '/' ? m[0] : `${m[0]} · ${SITE}`;
      setDescription(m[1]);
    } else if (!pathname.startsWith('/player/')) {
      // player pages set their own title once the player has loaded
      document.title = SITE;
    }
    if (pathname !== '/insights') trackPageView(pathname);
  }, [pathname]);

  return null;
}

export default PageMeta;

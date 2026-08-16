// NFL team → ESPN logo abbreviation. Mock-draft picks arrive as free text from
// Claude vision ("Kansas City Chiefs", "KC", "Chiefs"), so matching tries the
// full name, the city, the nickname, and the abbreviation itself.
const TEAMS = [
  { abbr: 'ari', city: 'arizona',      nick: 'cardinals' },
  { abbr: 'atl', city: 'atlanta',      nick: 'falcons' },
  { abbr: 'bal', city: 'baltimore',    nick: 'ravens' },
  { abbr: 'buf', city: 'buffalo',      nick: 'bills' },
  { abbr: 'car', city: 'carolina',     nick: 'panthers' },
  { abbr: 'chi', city: 'chicago',      nick: 'bears' },
  { abbr: 'cin', city: 'cincinnati',   nick: 'bengals' },
  { abbr: 'cle', city: 'cleveland',    nick: 'browns' },
  { abbr: 'dal', city: 'dallas',       nick: 'cowboys' },
  { abbr: 'den', city: 'denver',       nick: 'broncos' },
  { abbr: 'det', city: 'detroit',      nick: 'lions' },
  { abbr: 'gb',  city: 'green bay',    nick: 'packers' },
  { abbr: 'hou', city: 'houston',      nick: 'texans' },
  { abbr: 'ind', city: 'indianapolis', nick: 'colts' },
  { abbr: 'jax', city: 'jacksonville', nick: 'jaguars' },
  { abbr: 'kc',  city: 'kansas city',  nick: 'chiefs' },
  { abbr: 'lac', city: 'los angeles',  nick: 'chargers' },
  { abbr: 'lar', city: 'los angeles',  nick: 'rams' },
  { abbr: 'lv',  city: 'las vegas',    nick: 'raiders' },
  { abbr: 'mia', city: 'miami',        nick: 'dolphins' },
  { abbr: 'min', city: 'minnesota',    nick: 'vikings' },
  { abbr: 'ne',  city: 'new england',  nick: 'patriots' },
  { abbr: 'no',  city: 'new orleans',  nick: 'saints' },
  { abbr: 'nyg', city: 'new york',     nick: 'giants' },
  { abbr: 'nyj', city: 'new york',     nick: 'jets' },
  { abbr: 'phi', city: 'philadelphia', nick: 'eagles' },
  { abbr: 'pit', city: 'pittsburgh',   nick: 'steelers' },
  { abbr: 'sea', city: 'seattle',      nick: 'seahawks' },
  { abbr: 'sf',  city: 'san francisco', nick: '49ers' },
  { abbr: 'tb',  city: 'tampa bay',    nick: 'buccaneers' },
  { abbr: 'wsh', city: 'washington',   nick: 'commanders' },
];

export function nflTeamAbbr(name) {
  const t = (name || '').toLowerCase().trim();
  if (!t) return null;
  // Nickname is unambiguous (both LA and NY pairs differ by nickname), so
  // prefer it; city alone only matches when it is unambiguous.
  const byNick = TEAMS.find(x => t.includes(x.nick));
  if (byNick) return byNick.abbr;
  const byAbbr = TEAMS.find(x => t === x.abbr || t === x.abbr.toUpperCase().toLowerCase());
  if (byAbbr) return byAbbr.abbr;
  const cityMatches = TEAMS.filter(x => t.includes(x.city));
  return cityMatches.length === 1 ? cityMatches[0].abbr : null;
}

export function nflLogoUrl(name, size = 500) {
  const abbr = nflTeamAbbr(name);
  return abbr ? `https://a.espncdn.com/i/teamlogos/nfl/${size}/${abbr}.png` : null;
}

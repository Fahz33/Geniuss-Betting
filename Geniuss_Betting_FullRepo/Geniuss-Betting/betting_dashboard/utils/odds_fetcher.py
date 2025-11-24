import os
import requests
from typing import List, Dict, Any

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "37ec113c962988a4591233c742b5e2c7")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

def _last_word(name: str) -> str:
    if not name:
        return ""
    return str(name).strip().split(" ")[-1].upper()

def fetch_odds() -> List[Dict[str, Any]]:
    if not ODDS_API_KEY:
        raise RuntimeError("Set ODDS_API_KEY as env var.")
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american"
    }
    resp = requests.get(ODDS_BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()

def simplify_odds(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten Odds API events into a simpler home/away structure with one main book (DraftKings if present)."""
    simplified = []
    for ev in raw_events:
        home = ev.get("home_team")
        away = None
        teams = ev.get("teams") or []
        if teams and home:
            away = teams[0] if teams[1] == home else teams[1]
        else:
            away = ev.get("away_team")
        bookmakers = ev.get("bookmakers", [])
        main_book = None
        for b in bookmakers:
            if b.get("key") == "draftkings":
                main_book = b
                break
        if not main_book and bookmakers:
            main_book = bookmakers[0]

        home_ml = away_ml = None
        home_spread = away_spread = None
        for m in main_book.get("markets", []):
            mkey = m.get("key")
            if mkey == "h2h":
                for out in m.get("outcomes", []):
                    if out.get("name") == home:
                        home_ml = out.get("price")
                    elif out.get("name") == away:
                        away_ml = out.get("price")
            elif mkey == "spreads":
                for out in m.get("outcomes", []):
                    if out.get("name") == home:
                        home_spread = out.get("point")
                    elif out.get("name") == away:
                        away_spread = out.get("point")
        simplified.append({
            "id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "home_team": home,
            "away_team": away,
            "home_nick": _last_word(home),
            "away_nick": _last_word(away),
            "home_ml": home_ml,
            "away_ml": away_ml,
            "home_spread": home_spread,
            "away_spread": away_spread,
            "book": main_book.get("key") if main_book else None
        })
    return simplified

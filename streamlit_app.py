import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# =========================
# CONFIG
# =========================

PLAYS_CSV = "2025_plays.csv"  # <-- you must upload this file to the repo root
ROW_START_REGULAR_SEASON = 7988  # 0-based index => row 7989 is first regular-season play

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "37ec113c962988a4591233c742b5e2c7")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


# =========================
# HELPER FUNCTIONS
# =========================

def american_to_decimal(american: float):
    if american is None or pd.isna(american):
        return None
    american = float(american)
    if american > 0:
        return 1.0 + (american / 100.0)
    else:
        return 1.0 + (100.0 / abs(american))


def implied_prob_from_american(american: float):
    dec = american_to_decimal(american)
    if not dec or dec <= 1:
        return None
    return 1.0 / dec


def expected_value(prob: float, american: float, stake: float = 1.0):
    if prob is None or american is None or pd.isna(american):
        return None
    dec = american_to_decimal(american)
    win_return = (dec - 1.0) * stake
    lose_amount = stake
    return prob * win_return - (1 - prob) * lose_amount


def kelly_fraction(prob: float, american: float):
    if prob is None or american is None or pd.isna(american):
        return 0.0
    dec = american_to_decimal(american)
    b = dec - 1.0
    q = 1 - prob
    edge = (b * prob - q)
    if b <= 0 or edge <= 0:
        return 0.0
    return edge / b


def fetch_odds():
    """Fetch odds from The Odds API. Returns list of events or raises."""
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set in environment/secrets.")
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }
    resp = requests.get(ODDS_BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _last_word(name: str) -> str:
    if not name:
        return ""
    return str(name).strip().split(" ")[-1].upper()


def simplify_odds(raw_events):
    """Flatten Odds API events into home/away structure, try DraftKings first."""
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
        if main_book:
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

        simplified.append(
            {
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
                "book": main_book.get("key") if main_book else None,
            }
        )
    return simplified


def build_game_summary(df):
    """Build per-game summary using Season, Week, AwayTeam, HomeTeam, Day, Date."""
    required_cols = ["Season", "Week", "AwayTeam", "HomeTeam", "Day", "Date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing expected columns in 2025_plays.csv: {missing}")
        return None

    for col in ["IsScoringDrive", "IsScoringPlay", "PlayOutcome"]:
        if col not in df.columns:
            st.error(f"Missing expected column in 2025_plays.csv: {col}")
            return None

    group_cols = required_cols
    game_grp = df.groupby(group_cols, dropna=False)
    game_summary = game_grp.agg(
        total_plays=("PlayOutcome", "count"),
        scoring_drives=("IsScoringDrive", "sum"),
        scoring_plays=("IsScoringPlay", "sum"),
    ).reset_index()

    game_summary["scoring_drive_rate"] = (
        game_summary["scoring_drives"]
        / game_summary["total_plays"].replace(0, np.nan)
    )
    game_summary["scoring_play_rate"] = (
        game_summary["scoring_plays"]
        / game_summary["total_plays"].replace(0, np.nan)
    )

    return game_summary


def attach_matchup_and_model(game_summary, odds_df):
    """Join internal game summary to odds by nickname and build simple model + EV."""
    gs = game_summary.copy()
    gs["home_nick"] = gs["HomeTeam"].astype(str).str.upper()
    gs["away_nick"] = gs["AwayTeam"].astype(str).str.upper()

    merged = pd.merge(
        gs,
        odds_df,
        on=["home_nick", "away_nick"],
        how="left",
        suffixes=("", "_odds"),
    )

    # simple signal: scoring_drive_rate vs league average that week
    merged["home_edge"] = (
        merged["scoring_drive_rate"]
        - merged.groupby("Week")["scoring_drive_rate"].transform("mean")
    )
    merged["home_edge"] = merged["home_edge"].fillna(0)

    # squash edge into probability range (0,1)
    merged["model_prob_home"] = 1 / (1 + np.exp(-4 * merged["home_edge"]))

    merged["market_prob_home"] = merged["home_ml"].apply(
        implied_prob_from_american
    )
    merged["market_prob_away"] = merged["away_ml"].apply(
        implied_prob_from_american
    )

    merged["ev_home"] = merged.apply(
        lambda r: expected_value(r["model_prob_home"], r["home_ml"])
        if pd.notna(r["home_ml"])
        else np.nan,
        axis=1,
    )
    merged["ev_away"] = merged.apply(
        lambda r: expected_value(1 - r["model_prob_home"], r["away_ml"])
        if pd.notna(r["away_ml"])
        else np.nan,
        axis=1,
    )

    merged["kelly_home"] = merged.apply(
        lambda r: kelly_fraction(r["model_prob_home"], r["home_ml"])
        if pd.notna(r["home_ml"])
        else 0.0,
        axis=1,
    )
    merged["kelly_away"] = merged.apply(
        lambda r: kelly_fraction(1 - r["model_prob_home"], r["away_ml"])
        if pd.notna(r["away_ml"])
        else 0.0,
        axis=1,
    )

    return merged


# =========================
# STREAMLIT APP
# =========================

st.set_page_config(
    page_title="Geniuss NFL Betting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    body { background-color: #05070a; color: #e6f7ff; }
    .stApp { background: linear-gradient(135deg, #020617, #050816, #020617); }
    .big-title { font-size: 30px; color: #39ff14; font-weight: 800; }
    .sub-title { font-size: 18px; color: #7dd3fc; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="big-title">🏈 Geniuss NFL Betting Dashboard — Dark Neon</div>',
    unsafe_allow_html=True,
)

st.write(f"**Session time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

# ---- Load base data ----
if not os.path.exists(PLAYS_CSV):
    st.error(
        f"Could not find `{PLAYS_CSV}` in the repo root. "
        f"Upload 2025_plays.csv to the same folder as streamlit_app.py."
    )
    st.stop()

full_df = pd.read_csv(PLAYS_CSV, low_memory=False)

if len(full_df) <= ROW_START_REGULAR_SEASON:
    st.error(
        f"`{PLAYS_CSV}` has only {len(full_df)} rows; cannot slice from row 7989 "
        f"for regular season. Check the file."
    )
    st.stop()

# Apply your rule: regular season starts at row 7989 (1-based)
df = full_df.iloc[ROW_START_REGULAR_SEASON:].reset_index(drop=True)

# Optionally, try to drop preseason / HoF further if labeled
for col in df.columns:
    if col.lower() in ["seasontype", "season_type", "gametype", "game_type"]:
        df = df[~df[col].astype(str).str.upper().isin(["PRE", "HOF"])].reset_index(
            drop=True
        )

game_summary = build_game_summary(df)
if game_summary is None:
    st.stop()

tab_overview, tab_games, tab_upsets, tab_raw = st.tabs(
    ["Overview", "Games & EV", "Upset Alerts", "Raw Data"]
)

# ====================================
# TAB: OVERVIEW
# ====================================
with tab_overview:
    st.markdown('<div class="sub-title">Model Summary</div>', unsafe_allow_html=True)
    st.write("Base 2025 regular season (no preseason / HoF).")
    st.write(f"Plays in dataset: **{len(df):,}**")
    st.write(f"Games in summary: **{len(game_summary):,}**")

    st.write("Sample of game-level summary:")
    st.dataframe(
        game_summary[
            [
                "Season",
                "Week",
                "AwayTeam",
                "HomeTeam",
                "total_plays",
                "scoring_drive_rate",
                "scoring_play_rate",
            ]
        ].head(20)
    )

    st.markdown("### Fetch Odds & Build EV Model")
    st.write(
        "Press the button below to hit The Odds API using your stored key "
        "(ODDS_API_KEY) and build the model in real-time."
    )

    if st.button("🔄 Fetch Odds & Build Model"):
        try:
            raw_events = fetch_odds()
            odds_df = pd.DataFrame(simplify_odds(raw_events))
            model_df = attach_matchup_and_model(game_summary, odds_df)

            st.success("Model built successfully.")
            st.write("Top EV edges:")
            tmp = model_df.copy()
            tmp["BestSide"] = tmp.apply(
                lambda r: "HOME"
                if (r.get("ev_home", 0) or 0) > (r.get("ev_away", 0) or 0)
                else "AWAY",
                axis=1,
            )
            tmp["BestEV"] = tmp[["ev_home", "ev_away"]].max(axis=1)
            cols = [
                "Season",
                "Week",
                "AwayTeam",
                "HomeTeam",
                "home_ml",
                "away_ml",
                "BestSide",
                "BestEV",
                "model_prob_home",
                "market_prob_home",
            ]
            cols = [c for c in cols if c in tmp.columns]
            st.dataframe(tmp[cols].sort_values("BestEV", ascending=False).head(20))

            st.session_state["model_df"] = model_df
        except Exception as e:
            st.error(f"Error fetching odds or building model: {e}")


# ====================================
# TAB: GAMES & EV
# ====================================
with tab_games:
    st.markdown('<div class="sub-title">Games & Model View</div>', unsafe_allow_html=True)
    model_df = st.session_state.get("model_df")
    if model_df is not None:
        st.dataframe(model_df.head(100))
    else:
        st.info(
            "No model data yet. Go to the **Overview** tab and click "
            "**Fetch Odds & Build Model**."
        )
        st.write("Showing raw game summary meanwhile:")
        st.dataframe(game_summary.head(100))


# ====================================
# TAB: UPSET ALERTS
# ====================================
with tab_upsets:
    st.markdown('<div class="sub-title">Potential Upset Alerts</div>', unsafe_allow_html=True)
    model_df = st.session_state.get("model_df")
    if model_df is not None and "model_prob_home" in model_df.columns:
        dfm = model_df.copy()
        dfm = dfm.dropna(
            subset=["home_ml", "away_ml", "model_prob_home", "market_prob_home"]
        )
        dfm["home_is_dog"] = dfm.apply(
            lambda r: abs(r["home_ml"]) > abs(r["away_ml"])
            if (r["home_ml"] is not None and r["away_ml"] is not None)
            else False,
            axis=1,
        )
        dfm["home_edge_prob"] = dfm["model_prob_home"] - dfm["market_prob_home"]

        upset_home = dfm[(dfm["home_is_dog"]) & (dfm["home_edge_prob"] > 0.05)]
        upset_away = dfm[(~dfm["home_is_dog"]) & (-dfm["home_edge_prob"] > 0.05)]
        upsets = pd.concat([upset_home, upset_away], ignore_index=True, sort=False)

        if not upsets.empty:
            cols = [
                "Season",
                "Week",
                "AwayTeam",
                "HomeTeam",
                "home_ml",
                "away_ml",
                "model_prob_home",
                "market_prob_home",
                "home_edge_prob",
            ]
            cols = [c for c in cols if c in upsets.columns]
            st.dataframe(upsets[cols])
        else:
            st.info("No strong upset signals based on this simple model.")
    else:
        st.info(
            "No model data yet. Go to the **Overview** tab and click "
            "**Fetch Odds & Build Model**."
        )


# ====================================
# TAB: RAW DATA
# ====================================
with tab_raw:
    st.markdown('<div class="sub-title">Raw 2025 Data</div>', unsafe_allow_html=True)
    st.write("First 200 plays (regular season only):")
    st.dataframe(df.head(200))

    st.write("First 100 games summary:")
    st.dataframe(game_summary.head(100))

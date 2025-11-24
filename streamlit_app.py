import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="NFL Betting Dashboard", layout="wide", initial_sidebar_state="expanded")

# Dark neon theme styling
st.markdown(
    '''
    <style>
    body { background-color: #05070a; color: #e6f7ff; }
    .stApp { background: linear-gradient(135deg, #020617, #050816, #020617); }
    .big-title { font-size: 30px; color: #39ff14; font-weight: 800; }
    .sub-title { font-size: 18px; color: #7dd3fc; }
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">🏈 NFL Betting Dashboard — Dark Neon</div>', unsafe_allow_html=True)

# Load data
primary_path = os.path.join(DATA_DIR, "primary_weekly_data_2025.csv")
game_summary_path = os.path.join(DATA_DIR, "game_summary_2025.csv")
model_path = os.path.join(DATA_DIR, "games_model_2025.csv")

primary_df = pd.read_csv(primary_path) if os.path.exists(primary_path) else None
game_summary_df = pd.read_csv(game_summary_path) if os.path.exists(game_summary_path) else None
model_df = pd.read_csv(model_path) if os.path.exists(model_path) else None

tab_overview, tab_games, tab_upsets, tab_raw = st.tabs(["Overview", "Games & EV", "Upset Alerts", "Raw Data"])

with tab_overview:
    st.markdown('<div class="sub-title">Model Summary</div>', unsafe_allow_html=True)
    if model_df is not None and not model_df.empty:
        # derive best side & EV
        tmp = model_df.copy()
        tmp["BestSide"] = tmp.apply(lambda r: "HOME" if (r.get("ev_home", 0) or 0) > (r.get("ev_away", 0) or 0) else "AWAY", axis=1)
        tmp["BestEV"] = tmp[["ev_home","ev_away"]].max(axis=1)
        tmp = tmp.sort_values("BestEV", ascending=False)
        cols = ["Season","Week","AwayTeam","HomeTeam","home_ml","away_ml","BestSide","BestEV","model_prob_home","market_prob_home"]
        cols = [c for c in cols if c in tmp.columns]
        st.write("Top EV edges (model vs market):")
        st.dataframe(tmp[cols].head(20))
    else:
        st.info("Run weekly_updater.py to fetch odds and compute EV/model data.")

with tab_games:
    st.markdown('<div class="sub-title">Games & Model View</div>', unsafe_allow_html=True)
    if model_df is not None and not model_df.empty:
        st.dataframe(model_df.head(100))
    elif game_summary_df is not None:
        st.info("Model view not found. Showing internal game summary instead.")
        st.dataframe(game_summary_df.head(100))
    else:
        st.warning("No game data available.")

with tab_upsets:
    st.markdown('<div class="sub-title">Potential Upset Alerts</div>', unsafe_allow_html=True)
    if model_df is not None and not model_df.empty and "ev_home" in model_df.columns:
        df = model_df.copy()
        # find dogs with positive EV
        df = df.dropna(subset=["home_ml","away_ml","model_prob_home","market_prob_home"])
        df["home_is_dog"] = df.apply(
            lambda r: abs(r["home_ml"]) > abs(r["away_ml"]) if (r["home_ml"] is not None and r["away_ml"] is not None) else False,
            axis=1
        )
        df["home_edge_prob"] = df["model_prob_home"] - df["market_prob_home"]
        upset_home = df[(df["home_is_dog"]) & (df["home_edge_prob"] > 0.05)]
        upset_away = df[(~df["home_is_dog"]) & (-df["home_edge_prob"] > 0.05)]
        upsets = pd.concat([upset_home, upset_away], ignore_index=True, sort=False)
        if not upsets.empty:
            cols = ["Season","Week","AwayTeam","HomeTeam","home_ml","away_ml","model_prob_home","market_prob_home","home_edge_prob"]
            cols = [c for c in cols if c in upsets.columns]
            st.dataframe(upsets[cols])
        else:
            st.info("No strong upset signals detected by the simple model.")
    else:
        st.info("Run weekly_updater.py first to populate model data.")

with tab_raw:
    st.markdown('<div class="sub-title">Raw 2025 Data</div>', unsafe_allow_html=True)
    if primary_df is not None:
        st.write("Play-level data (first 200 rows):")
        st.dataframe(primary_df.head(200))
    if game_summary_df is not None:
        st.write("Game summary data (first 100 rows):")
        st.dataframe(game_summary_df.head(100))

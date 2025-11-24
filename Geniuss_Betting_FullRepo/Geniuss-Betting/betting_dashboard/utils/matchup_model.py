import pandas as pd
import numpy as np
from .ev_model import implied_prob_from_american, expected_value, kelly_fraction

def attach_matchup_and_model(game_summary: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    gs = game_summary.copy()
    gs["home_nick"] = gs["HomeTeam"].astype(str).str.upper()
    gs["away_nick"] = gs["AwayTeam"].astype(str).str.upper()

    merged = pd.merge(
        gs,
        odds_df,
        on=["home_nick", "away_nick"],
        how="left",
        suffixes=("", "_odds")
    )

    merged["home_edge"] = merged["scoring_drive_rate"] - merged["scoring_drive_rate"].groupby(merged["Week"]).transform("mean")
    merged["home_edge"] = merged["home_edge"].fillna(0)

    merged["model_prob_home"] = 1 / (1 + np.exp(-4 * merged["home_edge"]))

    merged["market_prob_home"] = merged["home_ml"].apply(implied_prob_from_american)
    merged["market_prob_away"] = merged["away_ml"].apply(implied_prob_from_american)

    merged["ev_home"] = merged.apply(
        lambda r: expected_value(r["model_prob_home"], r["home_ml"]) if pd.notna(r["home_ml"]) else np.nan,
        axis=1
    )
    merged["ev_away"] = merged.apply(
        lambda r: expected_value(1 - r["model_prob_home"], r["away_ml"]) if pd.notna(r["away_ml"]) else np.nan,
        axis=1
    )

    merged["kelly_home"] = merged.apply(
        lambda r: kelly_fraction(r["model_prob_home"], r["home_ml"]) if pd.notna(r["home_ml"]) else 0.0,
        axis=1
    )
    merged["kelly_away"] = merged.apply(
        lambda r: kelly_fraction(1 - r["model_prob_home"], r["away_ml"]) if pd.notna(r["away_ml"]) else 0.0,
        axis=1
    )

    return merged

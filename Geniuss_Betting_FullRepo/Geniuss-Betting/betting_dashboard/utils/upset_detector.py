import pandas as pd

def detect_upsets(model_df: pd.DataFrame, edge_threshold: float = 0.05) -> pd.DataFrame:
    df = model_df.copy()
    df = df.dropna(subset=["home_ml","away_ml","model_prob_home","market_prob_home"])
    df["home_is_dog"] = df.apply(
        lambda r: abs(r["home_ml"]) > abs(r["away_ml"]) if (r["home_ml"] is not None and r["away_ml"] is not None) else False,
        axis=1
    )
    df["home_edge_prob"] = df["model_prob_home"] - df["market_prob_home"]
    upset_home = df[(df["home_is_dog"]) & (df["home_edge_prob"] > edge_threshold)]
    upset_away = df[(~df["home_is_dog"]) & (-df["home_edge_prob"] > edge_threshold)]
    return pd.concat([upset_home, upset_away], ignore_index=True, sort=False)

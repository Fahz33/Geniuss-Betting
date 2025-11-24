import os
import json
import pandas as pd

from utils.odds_fetcher import fetch_odds, simplify_odds
from utils.matchup_model import attach_matchup_and_model

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

def run_update():
    game_summary_path = os.path.join(DATA_DIR, "game_summary_2025.csv")
    if not os.path.exists(game_summary_path):
        raise FileNotFoundError("game_summary_2025.csv not found in data/.")

    game_summary = pd.read_csv(game_summary_path)

    raw_events = fetch_odds()
    simplified = simplify_odds(raw_events)
    odds_df = pd.DataFrame(simplified)

    with open(os.path.join(DATA_DIR, "odds_raw.json"), "w") as f:
        json.dump(simplified, f, indent=2)

    model_df = attach_matchup_and_model(game_summary, odds_df)

    out_csv = os.path.join(DATA_DIR, "games_model_2025.csv")
    model_df.to_csv(out_csv, index=False)
    print("Updated:", out_csv)

if __name__ == "__main__":
    run_update()


# NFL Betting Dashboard (Dark Neon)

This project is ready to drop into GitHub and run locally or on Streamlit Cloud.

## Folder Structure

```
betting_dashboard/
    streamlit_app.py          # Main Streamlit app (dark neon, tabs)
    weekly_updater.py         # Pulls odds via The Odds API & builds model CSV
    README.md

    data/
        primary_weekly_data_2025.csv   # 2025 play-level, regular season only
        game_summary_2025.csv          # 2025 game-level summary
        games_model_2025.csv           # (generated) joined odds + model output
        odds_raw.json                  # (generated) raw odds snapshot

    utils/
        ev_model.py         # EV, odds conversions, Kelly
        odds_fetcher.py     # The Odds API integration (uses your key)
        weather.py          # Weather integration hooks (OpenWeather)
        injuries.py         # Injury integration placeholder
        matchup_model.py    # Simple internal model + EV calc
        upset_detector.py   # Upset candidate detection (not required by app)
```

## Setup

Python 3.9+ recommended.

```bash
cd betting_dashboard
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install streamlit pandas numpy requests scipy
```

### Environment Variables

Create a `.env` file or export env vars (recommended):

- `ODDS_API_KEY` — The Odds API key (already defaulted in code to your key `37ec...`, but you can override)
- `OPENWEATHER_API_KEY` — (optional) OpenWeather key for stadium weather

Example `.env`:

```text
ODDS_API_KEY=37ec113c962988a4591233c742b5e2c7
OPENWEATHER_API_KEY=YOUR_OPENWEATHER_KEY_HERE
```

## Workflow

1. **Update weekly data & odds**

   Run:

   ```bash
   python weekly_updater.py
   ```

   This will:

   - Pull live NFL odds from The Odds API
   - Simplify to a `odds_raw.json`
   - Join with internal `data/game_summary_2025.csv`
   - Compute a simple model win probability & EV
   - Save `data/games_model_2025.csv`

2. **Run the dashboard**

   ```bash
   streamlit run streamlit_app.py
   ```

   Tabs:

   - **Overview**: Top EV edges
   - **Games & EV**: Full model/odds table
   - **Upset Alerts**: Potential upset signals
   - **Raw Data**: 2025 play & game-level tables

## Deploy to GitHub

From **inside** `betting_dashboard/`:

```bash
git init
git add .
git commit -m "Initial NFL Betting Dashboard with odds + EV model"
git branch -M main
git remote add origin https://github.com/YOURNAME/NFL-Betting-Dashboard.git
git push -u origin main
```

Replace `YOURNAME` and repo name as desired.

## Notes

- This is a **starting point**: the model is intentionally simple so you can tweak logic in `utils/matchup_model.py` and `utils/ev_model.py`.
- Weather and injuries modules are wired conceptually; you'll need to add your API keys and flesh out injury scraping/integration.


# Geniuss Betting — NFL Dark Neon Dashboard

This repo is ready to deploy to Streamlit Cloud and uses a weekly auto-update workflow.

## Structure

```text
Geniuss-Betting/
├── betting_dashboard/
│   ├── streamlit_app.py
│   ├── weekly_updater.py
│   ├── data/
│   │   ├── primary_weekly_data_2025.csv
│   │   └── game_summary_2025.csv
│   └── utils/
│       ├── ev_model.py
│       ├── odds_fetcher.py
│       ├── weather.py
│       ├── injuries.py
│       ├── matchup_model.py
│       └── upset_detector.py
└── .github/workflows/weekly_update.yml
```

Main Streamlit app file: `betting_dashboard/streamlit_app.py`

## Running Locally

```bash
cd betting_dashboard
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r ../requirements.txt
python weekly_updater.py   # build odds/model file
streamlit run streamlit_app.py
```

## Deploying to Streamlit Cloud

- Repo: `Fahz33/Geniuss-Betting`
- Main file path: `betting_dashboard/streamlit_app.py`

### Secrets (on Streamlit Cloud)

Set in **App → Settings → Secrets**:

```toml
ODDS_API_KEY = "37ec113c962988a4591233c742b5e2c7"
OPENWEATHER_API_KEY = "your-openweather-key-here"
```

## GitHub Actions — Weekly Auto Update

The workflow in `.github/workflows/weekly_update.yml` runs every Wednesday at 00:00 UTC.

It:

- Installs dependencies
- Runs `betting_dashboard/weekly_updater.py`
- Commits updated `games_model_2025.csv` and `odds_raw.json`
- Pushes back to `main`

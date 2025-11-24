import os
import requests

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

STADIUM_COORDS = {
    "EAGLES": (39.9008, -75.1675),
    "COWBOYS": (32.7473, -97.0945),
}

def fetch_weather_for_team(team_nick: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "No OPENWEATHER_API_KEY set"
    coords = STADIUM_COORDS.get(team_nick.upper())
    if not coords:
        return "No coords for team"
    lat, lon = coords
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "imperial"}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        return f"Weather error {r.status_code}"
    j = r.json()
    desc = j["weather"][0]["description"]
    temp = j["main"]["temp"]
    wind = j.get("wind", {}).get("speed", 0)
    return f"{desc}, {temp}F, wind {wind} mph"

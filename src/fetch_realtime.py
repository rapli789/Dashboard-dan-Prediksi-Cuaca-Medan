import os
import requests
import pandas as pd
from datetime import datetime

# Koordinat Medan: Lintang 3.5952, Bujur 98.6722
MEDAN_LAT = 3.5952
MEDAN_LON = 98.6722


def fetch_today_weather():
    """Mengambil data cuaca real-time hari ini di Medan dari Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={MEDAN_LAT}&longitude={MEDAN_LON}&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean&timezone=Asia%2FJakarta"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        current = data["current"]
        daily = data["daily"]

        today_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "prcp": current.get("precipitation", 0.0),
            "tavg": daily.get("temperature_2m_mean", [current.get("temperature_2m")])[0],
            "tmin": daily.get("temperature_2m_min", [current.get("temperature_2m")])[0],
            "tmax": daily.get("temperature_2m_max", [current.get("temperature_2m")])[0],
            "wspd": current.get("wind_speed_10m", 0.0),
            "pres": current.get("surface_pressure", 1013.0),
            "wdir": current.get("wind_direction_10m", 0.0)
        }
        return pd.DataFrame([today_data])
    else:
        raise Exception(
            f"Gagal mengambil data dari API (Status Code: {response.status_code})")


def update_history_file(new_data_df, history_path="C:/Users/ASUS/.vscode/rain-medan/data/medan_weather_cleaned.csv"):
    """Menambahkan data hari ini ke histori tanpa membuat data ganda."""
    os.makedirs("data", exist_ok=True)

    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        # Hapus jika tanggal hari ini sudah pernah tercatat sebelumnya, lalu gabungkan
        history_df = history_df[history_df["date"]
                                != new_data_df["date"].iloc[0]]
        updated_df = pd.concat([history_df, new_data_df], ignore_index=True)
    else:
        updated_df = new_data_df

    updated_df.to_csv(history_path, index=False)
    return updated_df

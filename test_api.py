import requests
from datetime import datetime

print("Tanggal PC:", datetime.now().date())
try:
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": 3.5952, "longitude": 98.6722,
                "daily": "precipitation_sum,temperature_2m_mean",
                "past_days": 3, "forecast_days": 1, "timezone": "Asia/Jakarta"},
        timeout=15,
    )
    print("Status:", r.status_code)
    print("Tanggal API:", r.json()["daily"]["time"])
except Exception as e:
    print("GAGAL:", repr(e))
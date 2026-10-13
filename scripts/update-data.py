"""Perbarui data/medan_weather_cleaned.csv: isi hari yang kosong sampai kemarin,
lalu hitung kolom turunan (lag, rata-rata bergerak, target) yang masih kosong.

Pakai:  python scripts/update_data.py            (tulis ke CSV)
        python scripts/update_data.py --dry-run  (hanya tampilkan hasil)
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

CSV = Path(__file__).resolve().parent.parent / "data" / "medan_weather_cleaned.csv"
TZ = ZoneInfo("Asia/Jakarta")
LAT, LON = 3.5952, 98.6722
BASE = ["prcp", "tavg", "tmin", "tmax", "wspd", "pres"]
BASE_RE = "|".join(BASE)
API_VARS = {
    "temperature_2m_mean": "tavg", "temperature_2m_min": "tmin",
    "temperature_2m_max": "tmax", "precipitation_sum": "prcp",
    "wind_speed_10m_mean": "wspd", "surface_pressure_mean": "pres",
}


def call(url: str, **extra) -> pd.DataFrame:
    r = requests.get(url, timeout=60, params={
        "latitude": LAT, "longitude": LON, "daily": ",".join(API_VARS),
        "timezone": "Asia/Jakarta", **extra})
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"])}).set_index("date")
    for api_name, col in API_VARS.items():
        df[col] = d[api_name]
    return df


def fetch(start, end, today) -> pd.DataFrame:
    # arsip untuk tanggal lama, prakiraan (past_days maks 92) untuk tanggal terbaru
    parts = []
    past = min((today - start).days, 92)
    cover_start = today - timedelta(days=past)
    if start < cover_start:
        parts.append(call("https://archive-api.open-meteo.com/v1/archive",
                          start_date=start.isoformat(),
                          end_date=(cover_start - timedelta(days=1)).isoformat()))
    parts.append(call("https://api.open-meteo.com/v1/forecast", past_days=past, forecast_days=1))
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="first")]
    return out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))].dropna(subset=["tavg"])


def candidates(df: pd.DataFrame, col: str):
    # kemungkinan rumus untuk sebuah kolom turunan
    m = re.fullmatch(rf"({BASE_RE})_lag(\d+)", col)
    if m:
        yield df[m[1]].shift(int(m[2]))
    m = re.fullmatch(rf"({BASE_RE})_roll_mean_(\d+)d", col)
    if m:
        b, n = df[m[1]], int(m[2])
        for series in (b.shift(1), b):
            for min_p in (n, 1):
                yield series.rolling(n, min_periods=min_p).mean()
    if col in ("month", "bulan"):
        yield pd.Series(df.index.month, index=df.index, dtype=float)
    for b in BASE:
        yield df[b].shift(-1)
    nxt = df["prcp"].shift(-1)
    for th in (0.1, 0.5, 1.0, 5.0):
        for flag in (nxt >= th, nxt > th):
            yield flag.astype(float).where(nxt.notna())


def matches(old: pd.Series, cand: pd.Series) -> bool:
    # rumus dianggap benar jika cocok dengan nilai lama di hampir semua baris
    mask = old.notna() & cand.notna()
    return mask.sum() >= 30 and np.isclose(old[mask], cand[mask], atol=0.051, rtol=1e-3).mean() >= 0.995


def main(dry_run: bool = False) -> None:
    today = datetime.now(TZ).date()
    end = today - timedelta(days=1)  # hari ini belum selesai, jadi sampai kemarin

    raw = pd.read_csv(CSV)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.drop_duplicates("date").set_index("date").sort_index()
    for c in raw.columns:
        conv = pd.to_numeric(raw[c], errors="coerce")
        if conv.notna().sum() >= raw[c].notna().sum():
            raw[c] = conv

    full = raw.reindex(pd.date_range(raw.index.min(), end, freq="D"))
    full.index.name = "date"

    missing = full.index[full[BASE].isna().any(axis=1)]
    if len(missing):
        fresh = fetch(missing.min().date(), end, today)
        for c in BASE:
            full.loc[fresh.index, c] = full.loc[fresh.index, c].fillna(fresh[c])
        print(f"Hari kosong diisi: {len(missing)}, tersedia dari sumber: {len(fresh)}")

    for c in [c for c in full.columns if c not in BASE]:
        old = full[c]
        for cand in candidates(full, c):
            if matches(old, cand):
                full[c] = old.where(old.notna(), cand)
                break
        else:
            if old.isna().any():
                print(f"PERINGATAN: rumus kolom '{c}' tidak dikenali, baris baru dibiarkan kosong")

    if full.equals(raw.reindex(full.index)):
        print("Tidak ada perubahan.")
        return
    out = full.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    print(out.tail(5).to_string())
    if dry_run:
        print("Mode --dry-run: file tidak ditulis.")
    else:
        out.to_csv(CSV, index=False)
        print(f"Tersimpan: {CSV} ({len(out)} baris, sampai {out['date'].iloc[-1]})")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)

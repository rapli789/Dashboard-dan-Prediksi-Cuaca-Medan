"""Aplikasi Cuaca Medan: dashboard, prediksi hujan, dan chatbot AI.

Jalankan: streamlit run app.py
File    : app.py, assets/styles.css, assets/templates.html, .streamlit/config.toml
"""

import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from google import genai
except ImportError:  # chatbot nonaktif jika pustaka belum terpasang
    genai = None

log = logging.getLogger("cuaca-medan")

st.set_page_config(page_title="Cuaca Medan", layout="wide",
                   initial_sidebar_state="collapsed")

# konfigurasi
ASSETS_DIR = Path(__file__).parent / "assets"
MODEL_PATH = Path(
    "models/rain_prediction_rf.pkl")
DATA_CANDIDATES = [
    Path("data/medan_weather_cleaned.csv")]
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS",
    "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite",
).split(",") if m.strip()]
BUSY_CODES = ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE")
LAT, LON = 3.5952, 98.6722  
COLS = ["prcp", "tavg", "tmin", "tmax", "wspd", "pres"]
HOT_C = 32.0  # batas suhu untuk label "panas" (derajat C)
_HUJAN_RINGAN = ("Hujan ringan", "Siapkan payung lipat untuk berjaga-jaga.")
_HUJAN_SEDANG = ("Hujan sedang", "Bawa payung atau jas hujan saat keluar rumah.")
_HUJAN_LEBAT = ("Hujan lebat", "Hindari bepergian jika tidak mendesak dan waspadai genangan di jalan.")
_CERAH = ("Cerah", "Langit cerah. Gunakan tabir surya jika beraktivitas di luar ruangan.")
_KABUT = ("Berkabut", "Jarak pandang terbatas, berkendaralah dengan hati-hati.")
_PETIR = ("Badai petir", "Hindari area terbuka dan pohon tinggi, tunda aktivitas di luar ruangan.")
_GERIMIS = ("Gerimis", "Siapkan payung lipat untuk berjaga-jaga.")
WMO_ID = {
    0: _CERAH, 1: _CERAH,
    2: ("Cerah berawan", "Cuaca cukup nyaman untuk beraktivitas di luar ruangan."),
    3: ("Berawan", "Langit tertutup awan, tetapi belum ada tanda hujan."),
    45: _KABUT, 48: _KABUT,
    51: _GERIMIS, 53: _GERIMIS, 55: _GERIMIS, 56: _GERIMIS, 57: _GERIMIS,
    61: _HUJAN_RINGAN, 63: _HUJAN_SEDANG, 65: _HUJAN_LEBAT,
    66: _HUJAN_SEDANG, 67: _HUJAN_LEBAT,
    80: _HUJAN_RINGAN, 81: _HUJAN_SEDANG, 82: _HUJAN_LEBAT,
    95: _PETIR, 96: _PETIR, 99: _PETIR,
}
DAILY_VARS = {
    "temperature_2m_mean": "tavg", "temperature_2m_min": "tmin",
    "temperature_2m_max": "tmax", "precipitation_sum": "prcp",
    "wind_speed_10m_mean": "wspd", "surface_pressure_mean": "pres",
}
FEATURES = [
    "prcp_lag1", "prcp_lag2", "prcp_lag3",
    "tavg_lag1", "tavg_lag2", "tavg_lag3",
    "tmin_lag1", "tmin_lag2", "tmin_lag3",
    "tmax_lag1", "tmax_lag2", "tmax_lag3",
    "wspd_lag1", "wspd_lag2", "wspd_lag3",
    "pres_lag1", "pres_lag2", "pres_lag3",
    "prcp_roll_mean_3d", "prcp_roll_mean_7d",
]
VAR_LABEL = {"prcp": "Curah hujan", "tavg": "Suhu rata-rata", "tmin": "Suhu minimum",
             "tmax": "Suhu maksimum", "wspd": "Kecepatan angin", "pres": "Tekanan udara"}
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]

# palet warna (juga dikirim ke CSS sebagai variabel, mis. --accent-hover)
LIGHT = {
    "bg": "#E3EFF9", "side": "#D3E6F5", "panel": "rgba(255,255,255,0.80)", "pop": "#FFFFFF",
    "soft": "#E6F0F8", "border": "#CBDDEC", "text": "#25405B", "muted": "#5E7B97",
    "accent": "#3A78AD", "accent_hover": "#2F6794", "on_accent": "#FFFFFF",
    "rain": "#5B8FC4", "dry": "#5BAE9A", "warm": "#E3AE62", "neutral": "#9CB0C4",
    "grid": "#DAE7F2", "input_bg": "#FFFFFF", "input_border": "#B7CCE0",
    "chat_bg": "#FFFFFF", "chat_border": "#B7CCE0",
    "chat_text": "#25405B", "chat_placeholder": "#5E7B97",
}
DARK = {
    "bg": "#0F1A35", "side": "#1C2B52", "panel": "rgba(28,41,78,0.80)", "pop": "#1B2850",
    "soft": "#243560", "border": "#34466F", "text": "#E7EDF8", "muted": "#9FB1CF",
    "accent": "#E4D7A4", "accent_hover": "#F0E6BF", "on_accent": "#1A2342",
    "rain": "#7EA7E3", "dry": "#7CCBB6", "warm": "#E8CB8E", "neutral": "#7D8BAA",
    "grid": "#2B3C64", "input_bg": "#142146", "input_border": "#4B6090",
    "chat_bg": "rgba(255,255,255,0.06)", "chat_border": "rgba(214,224,245,0.35)",
    "chat_text": "#F4F7FF", "chat_placeholder": "#C9D3E8",
}
C = dict(LIGHT)  # palet aktif, diperbarui di main()


# aset: CSS dan template HTML
def read_asset(name: str) -> str:
    path = find_file(Path(name).name, "assets")
    if not path.exists():
        raise FileNotFoundError(
            f"{name} tidak ada. BASE_DIR={BASE_DIR}, isi folder: {sorted(os.listdir(BASE_DIR))}")
    return path.read_text(encoding="utf-8")


def squash(text: str) -> str:
    # hapus baris kosong agar markdown tidak memutus blok HTML/CSS
    return re.sub(r"\n\s*\n", "\n", text)


def load_templates() -> dict[str, str]:
    text = read_asset("assets/templates.html")
    blocks = {}
    for m in re.finditer(r"<!--\s*@(\w+)\s*-->(.*?)(?=<!--\s*@|\Z)", text, re.S):
        body = re.sub(r"<!--.*?-->", "", m.group(2), flags=re.S)
        blocks[m.group(1)] = "".join(line.strip()
                                     for line in body.splitlines())
    return blocks


BASE_DIR = Path(__file__).resolve().parent

def find_file(name: str, *folders: str) -> Path:
    standard = BASE_DIR.joinpath(*folders, name)
    if standard.exists():
        return standard
    skip = {".venv", "venv", ".git", "__pycache__"}
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if f.lower() == name.lower():
                return Path(root) / f
    return standard


TEMPLATES = load_templates()


def render_html(name: str, **values) -> str:
    out = TEMPLATES[name]
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def inject_css() -> None:
    dark = st.session_state.get("dark", False)
    theme = {**C, "scene": "var(--scene-dark)" if dark else "var(--scene-light)",
             "color_scheme": "dark" if dark else "light"}
    variables = ";".join(
        f"--{k.replace('_', '-')}:{v}" for k, v in theme.items())
    css = read_asset("assets/style.css") + f"\n:root{{{variables}}}"
    st.markdown(f"<style>{squash(css)}</style>", unsafe_allow_html=True)


# data: real-time, historis, model
def _daily_to_df(js: dict) -> pd.DataFrame:
    d = js["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"])})
    for api_name, col in DAILY_VARS.items():
        df[col] = d[api_name]
    return df


def sync_history_csv(api_df: pd.DataFrame) -> None:
    # tambahkan data terbaru ke CSV riwayat; baris hari ini ditimpa tiap pembaruan
    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        return
    try:
        old = pd.read_csv(path)
        old.columns = [c.strip().lower() for c in old.columns]
        if "date" not in old.columns or not all(c in old.columns for c in COLS):
            log.warning("CSV %s tidak punya kolom lengkap, sinkronisasi dilewati", path)
            return
        old["date"] = pd.to_datetime(old["date"])
        new = api_df[(api_df["date"] > old["date"].max()) |
                     (api_df["date"].dt.date == date.today())]
        new = new[["date"] + COLS].dropna(subset=["tavg"])
        if new.empty:
            return
        merged = pd.concat([old[~old["date"].isin(new["date"])], new], ignore_index=True)
        merged = merged.sort_values("date")
        merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
        tmp = path.with_suffix(".tmp")
        merged.to_csv(tmp, index=False)
        tmp.replace(path)
    except Exception:
        log.exception("Gagal memperbarui %s", path)


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_recent_cached(past_days: int) -> pd.DataFrame:
    # error dilempar agar hasil gagal tidak tersimpan di cache
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": LAT, "longitude": LON, "daily": ",".join(DAILY_VARS),
                "past_days": past_days, "forecast_days": 1, "timezone": "Asia/Jakarta"},
        timeout=8,
    )
    r.raise_for_status()
    df = _daily_to_df(r.json())
    if df.empty:
        raise ValueError("API mengembalikan data kosong")
    if past_days >= 30:
        sync_history_csv(df)
    return df


def fetch_recent(past_days: int = 7) -> pd.DataFrame | None:
    # cuaca real-time Medan: coba rentang penuh, lalu rentang lebih pendek
    if time.time() - st.session_state.get("fetch_fail_ts", 0) < 60:
        return None  # baru gagal, tunggu semenit agar halaman tidak lambat
    error = None
    for days in dict.fromkeys([past_days, 14, 7]):
        try:
            df = _fetch_recent_cached(days)
            st.session_state["fetch_error"] = None
            return df
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            log.exception("Gagal mengambil data real-time (past_days=%s)", days)
    st.session_state["fetch_fail_ts"] = time.time()
    st.session_state["fetch_error"] = error
    return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_current() -> dict | None:
    # kondisi langit saat ini dari Open-Meteo
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude": LAT, "longitude": LON,
                    "current": "temperature_2m,weather_code,cloud_cover,precipitation",
                    "timezone": "Asia/Jakarta"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["current"]
    except Exception:
        log.exception("Gagal mengambil kondisi saat ini")
        return None

def _demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    wet = 0.5 + 0.25 * np.cos(2 * np.pi * (doy - 305) / 365)
    prcp = np.where(rng.random(len(dates)) < wet, rng.gamma(
        1.4, 9.0, len(dates)), 0.0).round(1)
    tavg = 27.3 + 0.8 * np.sin(2 * np.pi * (doy - 90) / 365) - \
        0.03 * prcp + rng.normal(0, .5, len(dates))
    return pd.DataFrame({
        "date": dates, "prcp": prcp, "tavg": tavg.round(1),
        "tmin": (tavg - 3.4).round(1), "tmax": (tavg + 4.6 - 0.05 * prcp).round(1),
        "wspd": np.clip(rng.normal(7.5, 2.2, len(dates)), 1, None).round(1),
        "pres": (1009.5 + rng.normal(0, 1.6, len(dates)) - 0.04 * prcp).round(1),
    })


@st.cache_data(ttl=900, show_spinner=False)
def load_history() -> pd.DataFrame:
    # urutan sumber: CSV lokal, arsip Open-Meteo, data cadangan
    for path in DATA_CANDIDATES:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            if "date" in df.columns and all(c in df.columns for c in COLS):
                df["date"] = pd.to_datetime(df["date"])
                return df[["date"] + COLS].sort_values("date").reset_index(drop=True)
        except Exception:
            log.exception("Gagal membaca %s", path)
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": LAT, "longitude": LON, "daily": ",".join(DAILY_VARS),
                    "start_date": "2022-01-01", "end_date": (date.today() - timedelta(days=7)).isoformat(),
                    "timezone": "Asia/Jakarta"},
            timeout=20,
        )
        r.raise_for_status()
        return _daily_to_df(r.json()).dropna(subset=["tavg"]).reset_index(drop=True)
    except Exception:
        log.exception("Gagal mengambil arsip historis")
        return _demo_data()


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        log.exception("Gagal memuat model")
        return None


def date_label(ts: pd.Timestamp) -> str:
    return f"{HARI[ts.weekday()]}, {ts.day} {BULAN[ts.month - 1]} {ts.year}"


def get_context():
    # mengembalikan (hist, recent, today, clim)
    hist = load_history()
    recent = fetch_recent(92)  # batas maksimum Open-Meteo
    if recent is not None and not recent.empty:
        # sambung arsip yang tertinggal dengan data terbaru sampai hari ini
        newer = recent[recent["date"] > hist["date"].max()]
        hist = pd.concat([hist, newer], ignore_index=True)
        recent = recent.tail(8).copy()
    else:
        recent = hist.tail(8).copy()
    clim = hist[COLS].mean()
    recent = recent.sort_values("date").reset_index(drop=True)
    for c in COLS:
        recent[c] = recent[c].ffill().bfill().fillna(clim[c])
    row = recent.iloc[-1]
    today = {"date": row["date"].strftime("%Y-%m-%d"), "date_label": date_label(row["date"]),
         "stale": row["date"].date() < date.today(),
         **{c: float(row[c]) for c in COLS}}
    return hist, recent, today, clim


# prediksi dan penjelasan
def build_features(inp: dict) -> pd.DataFrame:
    p0, p1, p2 = inp["prcp"], inp["prcp_h1"], inp["prcp_h2"]
    row = {}
    for var in ["tavg", "tmin", "tmax", "wspd", "pres"]:
        for lag in (1, 2, 3):
            row[f"{var}_lag{lag}"] = inp[var]
    row.update({"prcp_lag1": p0, "prcp_lag2": p1, "prcp_lag3": p2})
    row["prcp_roll_mean_3d"] = float(np.mean([p0, p1, p2]))
    row["prcp_roll_mean_7d"] = inp["prcp_7d"]
    return pd.DataFrame([row])[FEATURES]


def run_prediction(model, inp: dict) -> tuple[int, float]:
    x = build_features(inp)
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        x = x[list(cols)]
    return int(model.predict(x)[0]), float(model.predict_proba(x)[0][1])


def get_importances(model) -> pd.DataFrame:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    names = getattr(model, "feature_names_in_", None)
    names = list(names) if names is not None else FEATURES
    imp = pd.DataFrame(
        {"feature": names, "importance": est.feature_importances_})
    imp["variable"] = imp["feature"].str.split("_").str[0].map(VAR_LABEL)
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def pretty_feature(name: str) -> str:
    if "roll_mean" in name:
        return f"Rata-rata curah hujan {name.split('_')[-1].replace('d', '')} hari"
    var, lag = name.split("_lag")
    return f"{VAR_LABEL.get(var, var)} (H-{int(lag) - 1})"


def build_findings(inp: dict, clim: pd.Series, imp) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    p = inp["prcp"]
    if p >= 20:
        out.append(
            ("up", f"<b>Curah hujan hari ini tinggi ({p:.1f} mm).</b> Atmosfer masih lembap dan hujan di Medan cenderung berlanjut beberapa hari."))
    elif p >= 1:
        out.append(
            ("up", f"<b>Hujan tercatat hari ini ({p:.1f} mm).</b> Kondisi basah meningkatkan peluang hujan lagi besok."))
    else:
        out.append(
            ("down", "<b>Hari ini relatif kering.</b> Tanpa hujan baru-baru ini, peluang hujan besok cenderung turun."))

    d = inp["pres"] - clim["pres"]
    if d <= -1.5:
        out.append(
            ("up", f"<b>Tekanan udara rendah ({inp['pres']:.1f} hPa, {abs(d):.1f} hPa di bawah rata-rata).</b> Udara cenderung naik dan membentuk awan hujan."))
    elif d >= 1.5:
        out.append(
            ("down", f"<b>Tekanan udara tinggi ({inp['pres']:.1f} hPa, {d:.1f} hPa di atas rata-rata).</b> Udara stabil sehingga awan hujan sulit terbentuk."))
    else:
        out.append(
            ("neutral", f"<b>Tekanan udara mendekati normal ({inp['pres']:.1f} hPa).</b> Faktor ini tidak condong ke salah satu arah."))

    r = inp["tmax"] - inp["tmin"]
    if r < 6:
        out.append(
            ("up", f"<b>Rentang suhu sempit ({r:.1f} derajat C).</b> Biasanya menandakan tutupan awan tebal dan kelembapan tinggi."))
    elif r > 9:
        out.append(
            ("down", f"<b>Rentang suhu lebar ({r:.1f} derajat C).</b> Langit cenderung cerah sepanjang hari."))
    if inp["tavg"] >= clim["tavg"] + 1:
        out.append(
            ("up", f"<b>Suhu rata-rata di atas normal ({inp['tavg']:.1f} derajat C).</b> Pemanasan permukaan dapat memicu hujan konvektif sore hari."))
    if inp["wspd"] >= clim["wspd"] * 1.3:
        out.append(
            ("up", f"<b>Angin cukup kencang ({inp['wspd']:.1f} km/jam).</b> Dapat membawa massa udara lembap ke wilayah Medan."))
    if imp is not None and len(imp):
        out.append(
            ("neutral", f"<b>Fitur paling berpengaruh pada model:</b> {pretty_feature(imp.iloc[0]['feature'])}."))
    return out


def describe_today(t: dict, now: dict | None = None) -> tuple[str, str]:
    # kondisi langit saat ini dari kode cuaca; cadangan dari data harian
    if now and now.get("weather_code") is not None:
        label, tip = WMO_ID.get(int(now["weather_code"]), (
            "Berawan", "Cuaca cukup nyaman untuk beraktivitas di luar ruangan."))
        if label in ("Cerah", "Cerah berawan") and float(now.get("temperature_2m") or 0) >= HOT_C:
            label += " dan panas"
            tip = "Cuaca terik. Perbanyak minum dan hindari berada di luar ruangan terlalu lama."
        if t["prcp"] >= 1:
            tip += f" Total hujan hari ini {t['prcp']:.1f} mm."
        return label, tip
    if t["prcp"] >= 20:
        return "Hujan lebat", "Hindari bepergian jika tidak mendesak dan waspadai genangan di jalan."
    if t["prcp"] >= 5:
        return "Hujan sedang", "Bawa payung atau jas hujan saat keluar rumah."
    if t["prcp"] >= 1:
        return "Hujan ringan", "Siapkan payung lipat untuk berjaga-jaga."
    if t["tmax"] >= HOT_C + 1:
        return "Cerah dan panas", "Cuaca terik. Perbanyak minum dan hindari berada di luar ruangan terlalu lama."
    if t["tmax"] - t["tmin"] > 9:
        return "Cerah", "Gunakan tabir surya dan perbanyak minum jika beraktivitas di luar ruangan."
    return "Berawan", "Cuaca cukup nyaman untuk beraktivitas di luar ruangan."


# komponen visual
def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito Sans, sans-serif", color=C["muted"], size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, x=0, font=dict(color=C["text"])),
        hoverlabel=dict(
            bgcolor=C["panel"], bordercolor=C["border"], font=dict(color=C["text"])),
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"], automargin=True)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False, automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


def panel_header(title: str, sub: str = "") -> None:
    st.markdown(render_html("panel_header", title=title, sub=sub),
                unsafe_allow_html=True)


def page_header(title: str, sub: str) -> None:
    st.markdown(render_html("page_header", title=title, sub=sub),
                unsafe_allow_html=True)


def gauge_chart(pred: int, prob: float) -> go.Figure:
    color = C["rain"] if pred == 1 else C["dry"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number=dict(suffix="%", valueformat=".1f",
                    font=dict(size=44, color=C["text"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], tickcolor=C["border"],
                      tickfont=dict(color=C["muted"], size=11)),
            bar=dict(color=color, thickness=0.3), bgcolor=C["soft"], borderwidth=0,
            threshold=dict(
                line=dict(color=C["text"], width=2), thickness=0.85, value=50),
        ),
    ))
    fig.update_layout(height=250, margin=dict(
        l=20, r=20, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def importance_chart(imp: pd.DataFrame, mode: str, top_n: int) -> go.Figure:
    if mode == "Per variabel":
        data = imp.groupby("variable", as_index=False)[
            "importance"].sum().sort_values("importance")
        labels = data["variable"]
    else:
        data = imp.head(top_n).sort_values("importance")
        labels = data["feature"].map(pretty_feature)
    fig = go.Figure(go.Bar(
        x=data["importance"], y=labels, orientation="h",
        marker=dict(color=data["importance"], colorscale=[
                    [0, C["neutral"]], [1, C["accent"]]]),
        text=[f"{v * 100:.1f}%" for v in data["importance"]], textposition="outside",
        textfont=dict(color=C["text"]),
        hovertemplate="%{y}<br>Pengaruh: %{x:.3f}<extra></extra>",
    ))
    fig.update_xaxes(title="Feature importance", range=[
                     0, data["importance"].max() * 1.22])
    fig.update_yaxes(showgrid=False)
    return style_fig(fig, max(320, 30 * len(data) + 80))


# menu 1: dashboard utama
def page_dashboard() -> None:
    _, recent, today, clim = get_context()
    page_header("Cuaca Medan Hari Ini", today["date_label"])

    cond, tip = describe_today(today, fetch_current())
    st.markdown(render_html("hero", cond=cond, tip=tip),
                unsafe_allow_html=True)

    m = st.columns(4)
    m[0].metric("Suhu rata-rata", f"{today['tavg']:.1f} C",
                f"Min {today['tmin']:.1f} / Maks {today['tmax']:.1f} C", delta_color="off")
    m[1].metric("Curah hujan", f"{today['prcp']:.1f} mm",
                f"{today['prcp'] - clim['prcp']:+.1f} dari normal", delta_color="off")
    m[2].metric("Kecepatan angin", f"{today['wspd']:.1f} km/jam",
                f"{today['wspd'] - clim['wspd']:+.1f} dari normal", delta_color="off")
    m[3].metric("Tekanan udara", f"{today['pres']:.1f} hPa",
                f"{today['pres'] - clim['pres']:+.1f} dari normal", delta_color="off")
    st.write("")

    with st.container(border=True):
        panel_header("Delapan hari terakhir",
                     "Curah hujan (batang) dan suhu rata-rata (garis)")
        fig = go.Figure()
        fig.add_bar(x=recent["date"], y=recent["prcp"], name="Curah hujan (mm)",
                    marker_color=C["rain"], opacity=0.55)
        fig.add_scatter(x=recent["date"], y=recent["tavg"], name="Suhu rata-rata (C)", yaxis="y2",
                        mode="lines+markers", line=dict(color=C["warm"], width=2.6))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                      tickfont=dict(color=C["muted"])), hovermode="x unified")
        fig.update_xaxes(tickformat="%d %b")
        show(style_fig(fig, 320))


# menu 2: prediksi dan histori
def _num(label, key, default, lo, hi, container=st):
    return container.number_input(label, lo, hi, float(np.clip(round(default, 1), lo, hi)), 0.1, key=key)


def collect_inputs(recent: pd.DataFrame, today: dict) -> dict:
    # nilai awal dari data real-time, bisa disesuaikan manual
    p = recent["prcp"].to_numpy()
    h1 = p[-2] if len(p) >= 2 else today["prcp"]
    h2 = p[-3] if len(p) >= 3 else today["prcp"]
    p7 = float(np.mean(p[-7:]))
    with st.expander("Sesuaikan parameter cuaca"):
        st.caption(
            "Terisi otomatis dari kondisi hari ini. Ubah nilai untuk mencoba skenario lain.")
        a = st.columns(3)
        prcp = _num("Curah hujan hari ini (mm)", "in_prcp",
                    today["prcp"], 0.0, 500.0, a[0])
        tavg = _num("Suhu rata-rata (C)", "in_tavg",
                    today["tavg"], 10.0, 40.0, a[1])
        wspd = _num("Kecepatan angin (km/jam)", "in_wspd",
                    today["wspd"], 0.0, 200.0, a[2])
        b = st.columns(3)
        tmin = _num("Suhu minimum (C)", "in_tmin",
                    today["tmin"], 10.0, 40.0, b[0])
        tmax = _num("Suhu maksimum (C)", "in_tmax",
                    today["tmax"], 10.0, 40.0, b[1])
        pres = _num("Tekanan udara (hPa)", "in_pres",
                    today["pres"], 900.0, 1100.0, b[2])
        c = st.columns(3)
        prcp_h1 = _num("Curah hujan kemarin (mm)",
                       "in_h1", h1, 0.0, 500.0, c[0])
        prcp_h2 = _num("Curah hujan 2 hari lalu (mm)",
                       "in_h2", h2, 0.0, 500.0, c[1])
        prcp_7d = _num("Rata-rata hujan 7 hari (mm)",
                       "in_7d", p7, 0.0, 500.0, c[2])
        if not (tmin <= tavg <= tmax):
            st.warning(
                "Urutan suhu tidak wajar: pastikan minimum <= rata-rata <= maksimum.")
    return dict(prcp=prcp, tavg=tavg, tmin=tmin, tmax=tmax, wspd=wspd, pres=pres,
                prcp_h1=prcp_h1, prcp_h2=prcp_h2, prcp_7d=prcp_7d)


def render_result(pred: int, prob: float) -> None:
    pill = "Hujan" if pred == 1 else "Tidak hujan"
    status = "Besok diprediksi hujan" if pred == 1 else "Besok diprediksi tidak hujan"
    desc = ("Siapkan payung atau jas hujan untuk aktivitas di luar ruangan." if pred == 1
            else "Kondisi diperkirakan cerah atau berawan tanpa hujan signifikan.")
    with st.container(border=True):
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            panel_header("Probabilitas hujan",
                         "Garis gelap menandai ambang keputusan 50%")
            show(gauge_chart(pred, prob))
        with right:
            st.markdown(
                render_html("result", tone="rain" if pred == 1 else "dry", pill=pill,
                            status=status, desc=desc, prob=f"{prob * 100:.1f}",
                            prob_inv=f"{(1 - prob) * 100:.1f}"),
                unsafe_allow_html=True)


def render_analysis(inp: dict, pred: int, prob: float, model, clim: pd.Series) -> None:
    try:
        imp = get_importances(model)
    except Exception:
        imp = None
    left, right = st.columns([1, 1.1], gap="medium")

    with left, st.container(border=True):
        panel_header("Alasan prediksi",
                     "Input hari ini dibandingkan rata-rata historis")
        labels = {"up": "Mendorong hujan",
                  "down": "Menekan hujan", "neutral": "Netral"}
        html = render_html("legend")
        for direction, text in build_findings(inp, clim, imp):
            html += render_html("finding", direction=direction,
                                tag=labels[direction], text=text)
        verdict = "hujan" if pred == 1 else "tidak hujan"
        html += render_html("summary", verdict=verdict,
                            prob=f"{prob * 100:.1f}")
        st.markdown(html, unsafe_allow_html=True)

    with right, st.container(border=True):
        panel_header("Pengaruh fitur pada model",
                     "Feature importance Random Forest")
        if imp is None:
            st.info("Rincian pengaruh fitur tidak tersedia untuk model ini.")
        else:
            mode = st.radio("Tampilkan", ["Per fitur", "Per variabel"], horizontal=True,
                            label_visibility="collapsed")
            top_n = st.slider("Jumlah fitur teratas", 5, len(
                imp), 10) if mode == "Per fitur" else 10
            show(importance_chart(imp, mode, top_n))
            st.caption("Nilai bersifat global: peran tiap fitur di seluruh pohon keputusan, "
                       "bukan hanya pada input ini.")


def render_history(hist: pd.DataFrame) -> None:
    d_min, d_max = hist["date"].min().date(), hist["date"].max().date()
    c1, c2 = st.columns([2, 1])
    rng = c1.date_input("Rentang tanggal", (d_min, d_max), d_min, d_max)
    window = c2.select_slider("Perataan (hari)", [1, 3, 7, 14, 30], value=7)
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        st.info("Pilih tanggal akhir untuk menampilkan grafik.")
        return
    df = hist[(hist["date"].dt.date >= rng[0]) & (
        hist["date"].dt.date <= rng[1])].copy()
    if df.empty:
        st.info("Tidak ada data pada rentang tanggal tersebut.")
        return

    # statistik ringkas
    df["bulan"] = df["date"].dt.month
    wettest_day = df.loc[df["prcp"].idxmax()]
    wet_month = df.groupby("bulan")["prcp"].sum().idxmax()
    k = st.columns(4)
    k[0].metric("Total curah hujan", f"{df['prcp'].sum():,.0f} mm")
    k[1].metric("Hari hujan", f"{(df['prcp'] >= 1).mean() * 100:.0f}%")
    k[2].metric("Hujan terlebat", f"{wettest_day['prcp']:.1f} mm",
                f"{wettest_day['date']:%d %b %Y}", delta_color="off")
    k[3].metric("Bulan terbasah", BULAN[wet_month - 1], f"Suhu rata-rata {df['tavg'].mean():.1f} C",
                delta_color="off")
    st.write("")

    def roll(s): return s.rolling(window, min_periods=1).mean()

    with st.container(border=True):
        panel_header("Curah hujan harian",
                     "Batang: harian. Garis: rata-rata bergerak.")
        fig = go.Figure()
        fig.add_bar(x=df["date"], y=df["prcp"], name="Harian",
                    marker_color=C["rain"], opacity=0.45)
        fig.add_scatter(x=df["date"], y=roll(df["prcp"]), name=f"Rata-rata {window} hari",
                        line=dict(color=C["accent"], width=2.5))
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(title="mm")
        show(style_fig(fig))

    with st.container(border=True):
        panel_header("Tren suhu", "Suhu minimum, rata-rata, dan maksimum")
        fig = go.Figure()
        fig.add_scatter(x=df["date"], y=roll(df["tmax"]),
                        name="Maks", line=dict(color=C["warm"], width=1.8))
        fig.add_scatter(x=df["date"], y=roll(df["tmin"]), name="Min", line=dict(color=C["rain"], width=1.8),
                        fill="tonexty", fillcolor="rgba(150,170,190,0.12)")
        fig.add_scatter(x=df["date"], y=roll(
            df["tavg"]), name="Rata-rata", line=dict(color=C["text"], width=2.4))
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(title="derajat C")
        show(style_fig(fig))

    with st.container(border=True):
        panel_header("Pola musiman",
                     "Total hujan dan persentase hari hujan per bulan")
        mon = df.groupby("bulan").agg(hujan=("prcp", "sum"),
                                      hari=("prcp", lambda s: (s >= 1).mean() * 100)).reset_index()
        mon["label"] = mon["bulan"].map(lambda m: BULAN[m - 1][:3])
        fig = go.Figure()
        fig.add_bar(x=mon["label"], y=mon["hujan"],
                    name="Total hujan (mm)", marker_color=C["rain"])
        fig.add_scatter(x=mon["label"], y=mon["hari"], name="Hari hujan (%)", yaxis="y2",
                        mode="lines+markers", line=dict(color=C["accent"], width=2.5))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, range=[0, 100],
                                      tickfont=dict(color=C["muted"])))
        show(style_fig(fig, 320))


def page_predict() -> None:
    hist, recent, today, clim = get_context()
    page_header("Prediksi dan Histori",
                "Perkiraan hujan besok serta tren dan statistik cuaca Medan.")
    tab_pred, tab_hist = st.tabs(
        ["Prediksi besok", "Tren dan statistik historis"])

    with tab_pred:
        inp = collect_inputs(recent, today)
        model = load_model()
        if model is None:
            st.info(
                "Fitur prediksi belum tersedia saat ini. Silakan coba beberapa saat lagi.")
        else:
            try:
                pred, prob = run_prediction(model, inp)
            except Exception:
                log.exception("Prediksi gagal")
                st.info("Prediksi belum dapat diproses saat ini. Silakan coba lagi.")
            else:
                render_result(pred, prob)
                st.write("")
                render_analysis(inp, pred, prob, model, clim)

    with tab_hist:
        render_history(hist)


# menu 3: chatbot ai
def get_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def generate_with_fallback(client, prompt: str) -> str:
    # coba tiap model berurutan
    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                return client.models.generate_content(
                    model=model_name, contents=prompt).text
            except Exception as e:
                last_error = e
                text = str(e)
                if "404" in text or "NOT_FOUND" in text:
                    break  # model tidak tersedia, langsung ke model berikutnya
                if any(code in text for code in BUSY_CODES):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise  # error lain (misalnya kunci salah) tidak perlu diulang
    raise last_error


def page_chat() -> None:
    _, _, today, _ = get_context()
    GEMINI_API_KEY = get_api_key()

    head = st.columns([5, 1])
    with head[0]:
        page_header("Asisten Cuaca Medan",
                    "Tanyakan apa saja seputar cuaca Medan hari ini.")
    if head[1].button("Hapus riwayat", use_container_width=True):
        st.session_state.chat_history = []

    if today:
        context_cuaca = f"""
    Kamu adalah asisten cuaca cerdas untuk kota Medan.
    Berikut adalah data cuaca Medan real-time hari ini ({today['date']}):
    - Suhu Rata-rata: {today['tavg']:.1f} °C
    - Suhu Minimum/Maksimum: {today['tmin']:.1f} °C / {today['tmax']:.1f} °C
    - Curah Hujan: {today['prcp']:.1f} mm
    - Kecepatan Angin: {today['wspd']:.1f} km/jam
    - Tekanan Udara: {today['pres']:.1f} hPa

    Jawab pertanyaan pengguna dengan ramah, komunikatif, dan berikan saran praktis.
    """
    else:
        context_cuaca = "Kamu adalah asisten cuaca cerdas untuk kota Medan."

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input(
        "Tanyakan sesuatu seputar cuaca Medan hari ini...")

    if user_query:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if not GEMINI_API_KEY:
                st.error(
                    "API Key Gemini belum dipasang. Harap set environment variable GEMINI_API_KEY.")
            elif genai is None:
                st.error(
                    "Pustaka google-genai belum terpasang. Jalankan: pip install google-genai")
            else:
                try:
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    prompt = f"{context_cuaca}\n\nPertanyaan User: {user_query}"
                    with st.spinner("Asisten sedang menyiapkan jawaban..."):
                        bot_reply = generate_with_fallback(client, prompt)
                    st.markdown(bot_reply)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": bot_reply})
                except Exception as e:
                    if any(code in str(e) for code in BUSY_CODES):
                        st.error("Layanan AI sedang sibuk. Silakan coba lagi dalam beberapa saat.")
                    else:
                        st.error(f"Gagal memproses respon AI: {e}")


# navigasi
PAGES = {
    "Dashboard Utama": page_dashboard,
    "Prediksi dan Histori": page_predict,
    "Chatbot AI": page_chat,
}


def _go(page: str) -> None:
    st.session_state.page = page


def _toggle_theme() -> None:
    st.session_state.dark = not st.session_state.dark
    try:
        st.query_params["mode"] = "dark" if st.session_state.dark else "light"
    except Exception:
        pass


def render_navbar() -> None:
    # navbar atas dengan tombol mode terang/gelap
    st.session_state.setdefault("page", "Dashboard Utama")
    cols = st.columns([2.4, 1.4, 1.7, 1.3, 1.2])
    cols[0].markdown(render_html("brand"), unsafe_allow_html=True)
    for i, name in enumerate(PAGES):
        cols[i + 1].button(name, key=f"nav_{i}", use_container_width=True,
                           type="primary" if st.session_state.page == name else "secondary",
                           on_click=_go, args=(name,))
    label = "🌙" if st.session_state.dark else "⛅"
    cols[4].button(label, key="theme_toggle",
                   use_container_width=True, on_click=_toggle_theme)
    st.markdown('<hr class="nav-line">', unsafe_allow_html=True)


def main() -> None:
    if "dark" not in st.session_state:
        try:
            st.session_state.dark = st.query_params.get("mode") == "dark"
        except Exception:
            st.session_state.dark = False
    C.update(DARK if st.session_state.dark else LIGHT)
    inject_css()
    render_navbar()
    PAGES[st.session_state.page]()


main()

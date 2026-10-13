import pandas as pd


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan awal dataset cuaca."""
    df_clean = df.copy()

    # Hapus kolom yang semuanya NaN
    cols_empty = [
        col
        for col in df_clean.columns
        if df_clean[col].isna().all() or col in ["snow", "wpgt", "tsun"]
    ]
    df_clean = df_clean.drop(columns=cols_empty, errors="ignore")

    # Fillna untuk prcp
    if "prcp" in df_clean.columns:
        df_clean["prcp"] = df_clean["prcp"].fillna(0.0)

    # Pastikan kolom date berformat datetime
    if "date" in df_clean.columns:
        df_clean["date"] = pd.to_datetime(df_clean["date"])
        df_clean = df_clean.sort_values("date").reset_index(drop=True)

    return df_clean

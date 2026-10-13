import pandas as pd


def create_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat fitur lag, rolling mean, dan target hujan besok."""
    df_feat = df.copy()

    # Target Hujan Besok
    df_feat["target_prcp_tomorrow"] = df_feat["prcp"].shift(-1)
    df_feat["target_rain_tomorrow"] = (
        df_feat["target_prcp_tomorrow"] > 0.5
    ).astype(int)

    # Fitur Lag
    weather_cols = ["prcp", "tavg", "tmin", "tmax", "wspd", "pres", "wdir"]
    for col in weather_cols:
        if col in df_feat.columns:
            df_feat[f"{col}_lag1"] = df_feat[col].shift(1)
            df_feat[f"{col}_lag2"] = df_feat[col].shift(2)
            df_feat[f"{col}_lag3"] = df_feat[col].shift(3)

    # Fitur Rolling Mean
    if "prcp" in df_feat.columns:
        df_feat["prcp_roll_mean_3d"] = (
            df_feat["prcp"].shift(1).rolling(window=3).mean()
        )
        df_feat["prcp_roll_mean_7d"] = (
            df_feat["prcp"].shift(1).rolling(window=7).mean()
        )

    df_model = df_feat.dropna().reset_index(drop=True)
    return df_model

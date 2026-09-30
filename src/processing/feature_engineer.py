"""
CRIP — Feature Engineering
"""
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    if "day_of_week" not in df.columns:
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wind_dir_sin"] = np.sin(np.radians(df.get("wind_dir", 0)))
    df["wind_dir_cos"] = np.cos(np.radians(df.get("wind_dir", 0)))
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_crop_burn_season"] = df["month"].isin([10, 11]).astype(int)
    ts = pd.to_datetime(df["timestamp"])
    df["is_festival"] = (
        ((ts.dt.month == 10) & ts.dt.day.between(20, 28)) |
        ((ts.dt.month == 11) & ts.dt.day.between(1, 5))
    ).astype(int)
    df["pm_ratio"] = (df["pm10"] / (df["pm25"] + 1)).clip(0, 10)

    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag{lag}h"] = df["aqi"].shift(lag)
        df[f"pm25_lag{lag}h"] = df["pm25"].shift(lag)
    for w in [3, 6, 24]:
        df[f"aqi_roll{w}h_mean"] = df["aqi"].shift(1).rolling(w).mean()
        df[f"aqi_roll{w}h_std"] = df["aqi"].shift(1).rolling(w).std().fillna(0)
        df[f"pm25_roll{w}h_mean"] = df["pm25"].shift(1).rolling(w).mean()
    df = df.dropna(subset=[c for c in df.columns if "lag" in c or "roll" in c])
    return df

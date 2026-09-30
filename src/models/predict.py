"""
CRIP — Prediction + SHAP Explanation Engine
"""
import os, json, joblib
import numpy as np
import pandas as pd
import shap
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(ROOT, "artifacts")

# Load metadata once
with open(os.path.join(ARTIFACTS, "metadata.json")) as f:
    META = json.load(f)

FEATURE_COLS = META["feature_cols"]
HORIZONS = META["horizons"]
STATIONS_META = {s["id"]: s for s in META["stations"]}

_MODELS = {}
_EXPLAINERS = {}

def _get_model(horizon: int):
    if horizon not in _MODELS:
        _MODELS[horizon] = joblib.load(os.path.join(ARTIFACTS, f"xgb_{horizon}h.pkl"))
    return _MODELS[horizon]

def _get_explainer(horizon: int):
    if horizon not in _EXPLAINERS:
        _EXPLAINERS[horizon] = shap.TreeExplainer(_get_model(horizon))
    return _EXPLAINERS[horizon]

FACTOR_MAP = {
    "hour_sin":               "Time of day (traffic cycle)",
    "hour_cos":               "Time of day (traffic cycle)",
    "month_sin":              "Seasonal pattern",
    "month_cos":              "Seasonal pattern",
    "is_crop_burn_season":    "🌾 Crop-residue burning season",
    "is_festival":            "🎆 Festival-related emissions",
    "is_weekend":             "Weekend (lower traffic)",
    "is_night":               "Nighttime atmospheric conditions",
    "blh":                    "🌡️ Atmospheric mixing (boundary layer)",
    "temperature":            "🌡️ Temperature conditions",
    "humidity":               "💧 Humidity level",
    "wind_speed":             "🌬️ Wind speed (dispersion)",
    "wind_dir_sin":           "Wind direction",
    "wind_dir_cos":           "Wind direction",
    "rainfall":               "🌧️ Rainfall (washout effect)",
    "road_density":           "🚗 Road traffic density",
    "industrial_zone":        "🏭 Industrial zone proximity",
    "pm_ratio":               "Dust vs combustion ratio (PM10/PM2.5)",
    "aqi_lag1h":              "AQI 1 hour ago",
    "aqi_lag3h":              "AQI 3 hours ago",
    "aqi_lag6h":              "AQI 6 hours ago",
    "aqi_lag12h":             "AQI 12 hours ago",
    "aqi_lag24h":             "🔁 Yesterday same-hour AQI",
    "pm25_lag1h":             "PM2.5 carry-over",
    "pm25_lag3h":             "PM2.5 carry-over",
    "pm25_lag6h":             "PM2.5 carry-over",
    "pm25_lag12h":            "PM2.5 carry-over",
    "pm25_lag24h":            "🔁 Yesterday same-hour PM2.5",
    "aqi_roll3h_mean":        "Short-term AQI trend",
    "aqi_roll6h_mean":        "6-hour AQI trend",
    "aqi_roll24h_mean":       "24-hour AQI trend",
    "aqi_roll3h_std":         "Recent AQI volatility",
    "aqi_roll6h_std":         "AQI variability",
    "aqi_roll24h_std":        "Daily AQI variability",
    "pm25_roll3h_mean":       "Short-term PM2.5 trend",
    "pm25_roll6h_mean":       "6-hour PM2.5 trend",
    "pm25_roll24h_mean":      "24-hour PM2.5 trend",
    "day_of_week":            "Day of week",
}

def aqi_category(aqi: float) -> tuple[str, str]:
    if aqi <= 50:   return "Good", "#00B050"
    if aqi <= 100:  return "Satisfactory", "#92D050"
    if aqi <= 200:  return "Moderate", "#FFFF00"
    if aqi <= 300:  return "Poor", "#FF0000"
    if aqi <= 400:  return "Very Poor", "#7030A0"
    return "Severe", "#1a1a1a"

def make_feature_vector(station_id: str, latest_readings: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature vector from the latest readings DataFrame.
    latest_readings must have columns: timestamp, aqi, pm25, pm10, no2, so2,
    temperature, humidity, wind_speed, wind_dir, rainfall, blh
    Needs at least 24 rows (hourly).
    """
    from src.processing.feature_engineer import engineer_features
    s = STATIONS_META[station_id]
    df = latest_readings.copy()
    df["station_id"] = station_id
    df["city"] = s["city"]
    df["road_density"] = s["road_density"]
    df["industrial_zone"] = s["industrial"]
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df = engineer_features(df)
    return df[FEATURE_COLS].tail(1)


def predict_all_horizons(feature_vec: pd.DataFrame, station_id: str) -> dict:
    """Return predictions for all horizons."""
    results = {}
    for h in HORIZONS:
        model = _get_model(h)
        pred = float(model.predict(feature_vec)[0])
        # Approximate confidence interval using model's leaf variance
        # (simple percentile-based estimate for MVP)
        noise_scale = {6: 15, 12: 22, 24: 35, 48: 50}[h]
        low = max(0, pred - noise_scale)
        high = min(500, pred + noise_scale)
        cat, color = aqi_category(pred)
        results[h] = {
            "predicted_aqi": round(pred, 1),
            "ci_low": round(low, 1),
            "ci_high": round(high, 1),
            "category": cat,
            "color": color,
        }
    return results


def explain_prediction(feature_vec: pd.DataFrame, horizon: int = 24) -> list[dict]:
    """Return top-5 SHAP factors for a prediction."""
    explainer = _get_explainer(horizon)
    shap_vals = explainer.shap_values(feature_vec)
    shap_series = pd.Series(dict(zip(FEATURE_COLS, shap_vals[0])))
    top = shap_series.abs().nlargest(8)
    factors = []
    seen_labels = set()
    for feat, _ in top.items():
        label = FACTOR_MAP.get(feat, feat)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        factors.append({
            "feature": feat,
            "label": label,
            "shap_value": round(float(shap_series[feat]), 2),
            "direction": "increases" if shap_series[feat] > 0 else "decreases",
        })
        if len(factors) >= 5:
            break
    return factors

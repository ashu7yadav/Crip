"""
CRIP — Climate Resilience Score
Computes a 5-factor composite score (0–100) per station.
Higher score = more resilient (lower risk).
"""
import numpy as np
import pandas as pd


def compute_resilience(station: dict, readings_df: pd.DataFrame) -> dict:
    """
    station: dict with id, lat, lon, road_density, industrial_zone, etc.
    readings_df: historical readings for this station.
    Returns a dict with overall_score (0–100) and sub-scores.
    """
    sid = station["id"]
    df = readings_df[readings_df["station_id"] == sid].copy()
    if df.empty:
        return _empty_score(sid)

    # Factor 1: AQI Trend (improving = good)
    df = df.sort_values("timestamp")
    if len(df) >= 48:
        recent_avg = df["aqi"].tail(24).mean()
        older_avg = df["aqi"].iloc[-48:-24].mean()
        trend_delta = older_avg - recent_avg  # positive = improving
        trend_score = np.clip(50 + trend_delta / 3, 0, 100)
    else:
        trend_score = 50

    # Factor 2: Extreme weather exposure (fewer high-AQI days = better)
    total_days = max((df["timestamp"].max() - df["timestamp"].min()).days, 1)
    bad_hours = (df["aqi"] > 200).sum()
    bad_ratio = bad_hours / len(df)
    weather_exposure_score = np.clip(100 - bad_ratio * 150, 0, 100)

    # Factor 3: Population vulnerability (proxy: city density)
    city_vuln = {"Delhi": 30, "Mumbai": 35, "Kolkata": 40,
                 "Chennai": 55, "Bengaluru": 60, "Hyderabad": 55}
    vulnerability_score = city_vuln.get(station.get("city", ""), 50)

    # Factor 4: Green cover (inverse of road_density as proxy)
    green_score = np.clip(100 - station.get("road_density", 3) * 15, 10, 90)

    # Factor 5: Pollution persistence (lower autocorrelation = faster recovery)
    if len(df) >= 24:
        autocorr = df["aqi"].autocorr(lag=6)
        persist_score = np.clip(100 - abs(autocorr) * 60, 10, 90)
    else:
        persist_score = 50

    # Weighted composite
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    scores = [trend_score, weather_exposure_score, vulnerability_score, green_score, persist_score]
    overall = round(float(np.dot(weights, scores)), 1)

    return {
        "station_id": sid,
        "overall_score": overall,
        "grade": _grade(overall),
        "aqi_trend_score": round(trend_score, 1),
        "weather_exposure_score": round(weather_exposure_score, 1),
        "vulnerability_score": round(vulnerability_score, 1),
        "green_cover_score": round(green_score, 1),
        "pollution_persistence_score": round(persist_score, 1),
        "disclaimer": "Informational indicator only. Not an official health or safety rating.",
    }


def _grade(score: float) -> str:
    if score >= 75: return "High Resilience"
    if score >= 50: return "Moderate Resilience"
    if score >= 30: return "Low Resilience"
    return "Very Low Resilience"


def _empty_score(sid: str) -> dict:
    return {
        "station_id": sid, "overall_score": 0, "grade": "No Data",
        "aqi_trend_score": 0, "weather_exposure_score": 0, "vulnerability_score": 0,
        "green_cover_score": 0, "pollution_persistence_score": 0,
        "disclaimer": "Insufficient data.",
    }

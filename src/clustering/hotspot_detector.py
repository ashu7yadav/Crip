"""
CRIP — Hotspot Detector (DBSCAN clustering on station AQI data)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def compute_hotspots(stations_df: pd.DataFrame, readings_df: pd.DataFrame,
                     days: int = 30) -> pd.DataFrame:
    """
    Identifies high-pollution hotspot clusters.
    stations_df: id, name, city, lat, lon
    readings_df: station_id, timestamp, aqi, pm25
    Returns stations_df with cluster label and avg_aqi.
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = readings_df[readings_df["timestamp"] >= cutoff]
    agg = recent.groupby("station_id").agg(
        avg_aqi=("aqi", "mean"),
        max_aqi=("aqi", "max"),
        p90_aqi=("aqi", lambda x: np.percentile(x, 90)),
        avg_pm25=("pm25", "mean"),
    ).reset_index()
    merged = stations_df.merge(agg, left_on="id", right_on="station_id", how="left")
    merged["avg_aqi"] = merged["avg_aqi"].fillna(0)

    # DBSCAN on (lat, lon) weighted by avg_aqi
    coords = merged[["lat", "lon"]].values
    # Scale coords + aqi together
    features = np.column_stack([coords, merged["avg_aqi"].values / 100])
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    db = DBSCAN(eps=0.5, min_samples=2).fit(X)
    merged["cluster"] = db.labels_

    # Risk label per cluster
    cluster_risk = merged.groupby("cluster")["avg_aqi"].mean()
    def risk_label(row):
        if row["avg_aqi"] > 250:  return "🔴 Critical"
        if row["avg_aqi"] > 180:  return "🟠 High"
        if row["avg_aqi"] > 100:  return "🟡 Moderate"
        return "🟢 Low"
    merged["risk_label"] = merged.apply(risk_label, axis=1)
    return merged


def get_station_history(station_id: str, readings_df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    return readings_df[
        (readings_df["station_id"] == station_id) &
        (readings_df["timestamp"] >= cutoff)
    ].sort_values("timestamp")

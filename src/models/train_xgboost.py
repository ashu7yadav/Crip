"""
CRIP — Synthetic Data + Model Training Script
Generates realistic AQI data for multiple Indian cities, engineers features,
trains XGBoost models for 4 forecast horizons, and serialises artifacts.
Run: python src/models/train_xgboost.py
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROC = os.path.join(ROOT, "data", "processed")
ARTIFACTS = os.path.join(ROOT, "artifacts")
for p in [DATA_RAW, DATA_PROC, ARTIFACTS]:
    os.makedirs(p, exist_ok=True)

# ── Station metadata ──────────────────────────────────────────────────────────
STATIONS = [
    {"id": "DL_001", "name": "Anand Vihar",       "city": "Delhi",     "lat": 28.6469, "lon": 77.3164, "road_density": 5, "industrial": 1, "state": "Delhi"},
    {"id": "DL_002", "name": "ITO",                "city": "Delhi",     "lat": 28.6289, "lon": 77.2411, "road_density": 4, "industrial": 0, "state": "Delhi"},
    {"id": "DL_003", "name": "Dwarka Sector 8",    "city": "Delhi",     "lat": 28.5795, "lon": 77.0585, "road_density": 3, "industrial": 0, "state": "Delhi"},
    {"id": "DL_004", "name": "Jahangirpuri",       "city": "Delhi",     "lat": 28.7298, "lon": 77.1624, "road_density": 3, "industrial": 1, "state": "Delhi"},
    {"id": "MU_001", "name": "Bandra Kurla Complex","city": "Mumbai",   "lat": 19.0632, "lon": 72.8677, "road_density": 4, "industrial": 0, "state": "Maharashtra"},
    {"id": "MU_002", "name": "Chembur",            "city": "Mumbai",    "lat": 19.0522, "lon": 72.8994, "road_density": 3, "industrial": 1, "state": "Maharashtra"},
    {"id": "MU_003", "name": "Worli",              "city": "Mumbai",    "lat": 19.0148, "lon": 72.8160, "road_density": 3, "industrial": 0, "state": "Maharashtra"},
    {"id": "KO_001", "name": "Jadavpur",           "city": "Kolkata",   "lat": 22.4988, "lon": 88.3714, "road_density": 3, "industrial": 1, "state": "West Bengal"},
    {"id": "KO_002", "name": "Rabindra Sarani",    "city": "Kolkata",   "lat": 22.5726, "lon": 88.3639, "road_density": 4, "industrial": 0, "state": "West Bengal"},
    {"id": "CH_001", "name": "Manali",             "city": "Chennai",   "lat": 13.1654, "lon": 80.2640, "road_density": 2, "industrial": 1, "state": "Tamil Nadu"},
    {"id": "BG_001", "name": "BTM Layout",         "city": "Bengaluru", "lat": 12.9141, "lon": 77.6101, "road_density": 4, "industrial": 0, "state": "Karnataka"},
    {"id": "HY_001", "name": "Bollaram",           "city": "Hyderabad", "lat": 17.4806, "lon": 78.3747, "road_density": 2, "industrial": 1, "state": "Telangana"},
]

# ── Synthetic AQI generation ──────────────────────────────────────────────────
def generate_station_data(station: dict, days: int = 730) -> pd.DataFrame:
    """Generate realistic hourly AQI data with seasonal, diurnal, and weather patterns."""
    rng = np.random.default_rng(seed=hash(station["id"]) % 2**32)
    n = days * 24
    timestamps = pd.date_range(end=datetime.now().replace(minute=0, second=0, microsecond=0),
                               periods=n, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Base AQI — city baseline
    city_base = {"Delhi": 180, "Mumbai": 120, "Kolkata": 140, "Chennai": 90, "Bengaluru": 100, "Hyderabad": 110}
    base = city_base.get(station["city"], 130)

    # Seasonal component (worse Oct–Jan for North India)
    if station["state"] in ["Delhi", "Punjab", "Haryana", "West Bengal"]:
        seasonal = 80 * np.sin(np.pi * (df["month"] - 4) / 6)  # peak Dec
    else:
        seasonal = 30 * np.sin(np.pi * (df["month"] - 5) / 6)

    # Diurnal pattern (morning/evening peaks)
    diurnal = (
        40 * np.exp(-0.5 * ((df["hour"] - 8) / 2) ** 2) +  # morning rush
        35 * np.exp(-0.5 * ((df["hour"] - 19) / 2) ** 2)   # evening rush
    )

    # Industrial zone bonus
    industrial_bonus = station["industrial"] * rng.uniform(20, 40, n)
    road_bonus = station["road_density"] * rng.uniform(5, 10, n)

    # Crop burning season (Oct–Nov, North India only)
    is_crop_burn = ((df["month"].isin([10, 11])) &
                    (station["state"] in ["Delhi", "Punjab", "Haryana"])).astype(float)
    crop_bonus = is_crop_burn * rng.exponential(50, n)

    # Random weather noise
    noise = rng.normal(0, 15, n)

    # Compose AQI
    aqi = (base + seasonal.values + diurnal.values + industrial_bonus +
           road_bonus + crop_bonus + noise).clip(0, 500)

    # Weekend dip
    weekend_mask = df["day_of_week"].isin([5, 6])
    aqi[weekend_mask.values] *= rng.uniform(0.85, 0.95, weekend_mask.sum())

    # Weather variables
    temp = 25 + 10 * np.sin(np.pi * (df["month"] - 3) / 6) + rng.normal(0, 3, n)
    humidity = 60 + 20 * np.sin(np.pi * (df["month"] - 6) / 6) + rng.normal(0, 8, n)
    humidity = humidity.clip(10, 100)
    wind_speed = np.abs(rng.normal(3, 2, n)).clip(0.1, 20)
    wind_dir = rng.uniform(0, 360, n)
    rainfall = np.zeros(n)
    # Monsoon rain (Jun–Sep)
    monsoon = df["month"].isin([6, 7, 8, 9]).values
    rainfall[monsoon] = rng.exponential(2, monsoon.sum())

    # Boundary layer height (low in winter mornings → trapping pollution)
    blh = 1200 + 600 * np.sin(np.pi * (df["month"] - 3) / 6) - \
          300 * np.exp(-0.5 * ((df["hour"] - 4) / 3) ** 2) + rng.normal(0, 100, n)
    blh = blh.clip(100, 3000)

    # PM2.5 / PM10 derived from AQI with ratio variation
    pm25 = (aqi * 0.35 + rng.normal(0, 5, n)).clip(0, 300)
    pm10 = (pm25 * rng.uniform(1.3, 2.2, n) + rng.normal(0, 8, n)).clip(0, 500)
    no2 = (aqi * 0.08 + rng.normal(0, 3, n)).clip(0, 100)
    so2 = (station["industrial"] * aqi * 0.04 + rng.normal(0, 2, n)).clip(0, 50)

    # Rain washes out pollution
    rain_washout = np.exp(-rainfall * 0.3)
    aqi = (aqi * rain_washout).clip(0, 500)
    pm25 = (pm25 * rain_washout).clip(0, 300)
    pm10 = (pm10 * rain_washout).clip(0, 500)

    df["aqi"] = aqi.round(1)
    df["pm25"] = pm25.round(1)
    df["pm10"] = pm10.round(1)
    df["no2"] = no2.round(1)
    df["so2"] = so2.round(1)
    df["temperature"] = temp.round(1)
    df["humidity"] = humidity.round(1)
    df["wind_speed"] = wind_speed.round(2)
    df["wind_dir"] = wind_dir.round(1)
    df["rainfall"] = rainfall.round(2)
    df["blh"] = blh.round(0)
    df["station_id"] = station["id"]
    df["city"] = station["city"]
    df["road_density"] = station["road_density"]
    df["industrial_zone"] = station["industrial"]
    return df


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wind_dir_sin"] = np.sin(np.radians(df["wind_dir"]))
    df["wind_dir_cos"] = np.cos(np.radians(df["wind_dir"]))
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_crop_burn_season"] = (df["month"].isin([10, 11])).astype(int)
    df["is_festival"] = (
        ((df["timestamp"].dt.month == 10) & (df["timestamp"].dt.day.between(20, 28))) |
        ((df["timestamp"].dt.month == 11) & (df["timestamp"].dt.day.between(1, 5)))
    ).astype(int)
    df["pm_ratio"] = (df["pm10"] / (df["pm25"] + 1)).clip(0, 10)
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag{lag}h"] = df["aqi"].shift(lag)
        df[f"pm25_lag{lag}h"] = df["pm25"].shift(lag)
    for w in [3, 6, 24]:
        df[f"aqi_roll{w}h_mean"] = df["aqi"].shift(1).rolling(w).mean()
        df[f"aqi_roll{w}h_std"] = df["aqi"].shift(1).rolling(w).std()
        df[f"pm25_roll{w}h_mean"] = df["pm25"].shift(1).rolling(w).mean()
    df = df.dropna().reset_index(drop=True)
    return df

FEATURE_COLS = [
    "hour_sin","hour_cos","month_sin","month_cos",
    "day_of_week","is_weekend","is_night","is_crop_burn_season","is_festival",
    "temperature","humidity","wind_speed","wind_dir_sin","wind_dir_cos",
    "rainfall","blh","pm_ratio","road_density","industrial_zone",
    "aqi_lag1h","aqi_lag3h","aqi_lag6h","aqi_lag12h","aqi_lag24h",
    "pm25_lag1h","pm25_lag3h","pm25_lag6h","pm25_lag12h","pm25_lag24h",
    "aqi_roll3h_mean","aqi_roll6h_mean","aqi_roll24h_mean",
    "aqi_roll3h_std","aqi_roll6h_std","aqi_roll24h_std",
    "pm25_roll3h_mean","pm25_roll6h_mean","pm25_roll24h_mean",
]

HORIZONS = [6, 12, 24, 48]


# ── Training ──────────────────────────────────────────────────────────────────
def train_models():
    print("=" * 60)
    print("CRIP — Model Training Pipeline")
    print("=" * 60)

    # 1. Generate data
    all_dfs = []
    print("\n[1/4] Generating synthetic training data for all stations...")
    for s in STATIONS:
        df = generate_station_data(s, days=730)
        df = engineer_features(df)
        all_dfs.append(df)
        print(f"  [OK] {s['id']} — {s['name']}, {s['city']} ({len(df):,} rows)")

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Save processed data
    full_df.to_parquet(os.path.join(DATA_PROC, "all_stations.parquet"), index=False)
    print(f"\n  Saved processed data: {len(full_df):,} total rows")

    # 2. Train per horizon
    print("\n[2/4] Training XGBoost models (4 horizons)...")
    metrics_all = {}
    for h in HORIZONS:
        target_col = f"aqi_target_{h}h"
        # Create target: AQI h-hours ahead
        full_df[target_col] = full_df.groupby("station_id")["aqi"].shift(-h)
        df_h = full_df.dropna(subset=[target_col])
        X = df_h[FEATURE_COLS]
        y = df_h[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Persistence baseline
        pers_pred = X_test["aqi_lag1h"].values
        pers_rmse = np.sqrt(mean_squared_error(y_test, pers_pred))
        skill = 1 - (rmse / pers_rmse)

        metrics_all[f"{h}h"] = {"rmse": round(rmse, 2), "mae": round(mae, 2),
                                  "r2": round(r2, 3), "skill_score": round(skill, 3)}
        path = os.path.join(ARTIFACTS, f"xgb_{h}h.pkl")
        joblib.dump(model, path)
        print(f"  [OK] {h}h model — RMSE={rmse:.1f}, MAE={mae:.1f}, R²={r2:.3f}, Skill={skill:.3f} → saved")

    # 3. Anomaly detector
    print("\n[3/4] Training Isolation Forest (anomaly detection)...")
    recent = full_df.tail(50000)[FEATURE_COLS + ["aqi"]]
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(recent)
    joblib.dump(iso, os.path.join(ARTIFACTS, "isolation_forest.pkl"))
    print("  [OK] Isolation Forest saved")

    # 4. Save metadata
    meta = {
        "trained_at": datetime.now().isoformat(),
        "stations": STATIONS,
        "feature_cols": FEATURE_COLS,
        "horizons": HORIZONS,
        "metrics": metrics_all,
    }
    with open(os.path.join(ARTIFACTS, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[4/4] Saving station metadata...")
    pd.DataFrame(STATIONS).to_csv(os.path.join("data", "static", "stations.csv"), index=False)

    print("\n" + "=" * 60)
    print("[DONE] Training complete! Artifacts saved to /artifacts/")
    print("Metrics summary:")
    for h, m in metrics_all.items():
        print(f"  {h}: RMSE={m['rmse']}, R²={m['r2']}, Skill={m['skill_score']}")
    print("\nRun: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    train_models()

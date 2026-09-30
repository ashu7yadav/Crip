"""
CRIP — What-If Scenario Simulator
Perturbs input features and re-runs the model to produce counterfactual predictions.
DISCLAIMER: Model simulation only — not a physical atmospheric model.
"""
import numpy as np
import pandas as pd
import joblib, os, json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS = os.path.join(ROOT, "artifacts")

with open(os.path.join(ARTIFACTS, "metadata.json")) as f:
    META = json.load(f)
FEATURE_COLS = META["feature_cols"]


def simulate(feature_vec: pd.DataFrame, horizon: int = 24,
             traffic_reduction: float = 0.0,
             green_cover_increase: float = 0.0,
             rainfall_mm: float = 0.0,
             industrial_reduction: float = 0.0) -> dict:
    """
    Apply scenario adjustments to feature vector and predict counterfactual AQI.

    traffic_reduction: 0–1 (fraction e.g. 0.3 = 30% traffic reduction)
    green_cover_increase: 0–1 (fraction e.g. 0.2 = 20% more green cover)
    rainfall_mm: additional rainfall in mm
    industrial_reduction: 0–1 (fraction e.g. 0.5 = 50% industrial cut)

    Returns baseline_aqi, scenario_aqi, delta, and explanation text.
    """
    model = joblib.load(os.path.join(ARTIFACTS, f"xgb_{horizon}h.pkl"))
    X_baseline = feature_vec.copy()
    X_scenario = feature_vec.copy()

    baseline_aqi = float(model.predict(X_baseline)[0])

    # Traffic reduction → lower road_density proxy + lag AQIs partially reduced
    if traffic_reduction > 0:
        X_scenario["road_density"] = X_scenario["road_density"] * (1 - traffic_reduction * 0.6)
        for col in ["aqi_lag1h", "aqi_lag3h", "aqi_roll3h_mean"]:
            if col in X_scenario.columns:
                X_scenario[col] = X_scenario[col] * (1 - traffic_reduction * 0.25)

    # Green cover → slightly improved BLH (better dispersion), lower aqi baselines
    if green_cover_increase > 0:
        if "blh" in X_scenario.columns:
            X_scenario["blh"] = X_scenario["blh"] * (1 + green_cover_increase * 0.1)
        for col in ["aqi_lag6h", "aqi_lag24h", "aqi_roll24h_mean"]:
            if col in X_scenario.columns:
                X_scenario[col] = X_scenario[col] * (1 - green_cover_increase * 0.08)

    # Rainfall → washout effect on PM lags
    if rainfall_mm > 0:
        washout = np.exp(-rainfall_mm * 0.15)
        for col in [c for c in X_scenario.columns if "pm25" in c or "aqi_lag" in c or "aqi_roll" in c]:
            X_scenario[col] = X_scenario[col] * washout
        if "rainfall" in X_scenario.columns:
            X_scenario["rainfall"] = X_scenario["rainfall"] + rainfall_mm

    # Industrial reduction → lower so2/no2 proxied via aqi lags + industrial flag
    if industrial_reduction > 0:
        X_scenario["industrial_zone"] = X_scenario["industrial_zone"] * (1 - industrial_reduction * 0.8)
        for col in ["aqi_lag3h", "aqi_lag6h", "aqi_roll6h_mean"]:
            if col in X_scenario.columns:
                X_scenario[col] = X_scenario[col] * (1 - industrial_reduction * 0.15)

    scenario_aqi = float(model.predict(X_scenario)[0])
    delta = baseline_aqi - scenario_aqi
    percent_change = (delta / max(baseline_aqi, 1)) * 100

    # Build explanation
    actions = []
    if traffic_reduction > 0:
        actions.append(f"Traffic reduced by {int(traffic_reduction*100)}%")
    if green_cover_increase > 0:
        actions.append(f"Green cover increased by {int(green_cover_increase*100)}%")
    if rainfall_mm > 0:
        actions.append(f"Simulated rainfall of {rainfall_mm:.1f} mm")
    if industrial_reduction > 0:
        actions.append(f"Industrial emissions cut by {int(industrial_reduction*100)}%")

    return {
        "horizon_h": horizon,
        "baseline_aqi": round(baseline_aqi, 1),
        "scenario_aqi": round(max(0, scenario_aqi), 1),
        "delta": round(delta, 1),
        "percent_improvement": round(percent_change, 1),
        "scenario_description": " + ".join(actions) if actions else "No change",
        "disclaimer": "⚠️ This is a statistical model simulation, not a physical atmospheric model. "
                      "Real-world effects depend on complex dynamics not captured here.",
    }

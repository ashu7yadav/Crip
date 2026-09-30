"""
CRIP — CleanAir Resilience Intelligence Platform
Main Streamlit Dashboard
Run: streamlit run app.py
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRIP — CleanAir Resilience Intelligence Platform",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1628 50%, #0a1220 100%);
    color: #e2e8f0;
}

/* Header */
.crip-header {
    background: linear-gradient(135deg, #1a2744 0%, #0f3460 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 30px rgba(0,0,0,0.4);
}
.crip-title { font-size: 1.8rem; font-weight: 800; color: #63b3ed; margin: 0; }
.crip-sub { font-size: 0.85rem; color: #90cdf4; margin: 4px 0 0; }
.crip-badge { background: #1a365d; border: 1px solid #63b3ed;
              border-radius: 20px; padding: 6px 14px; font-size: 0.75rem; color: #90cdf4; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a2744 0%, #162032 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px; padding: 18px 20px;
    text-align: center; transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 6px 30px rgba(0,0,0,0.4); }
.metric-value { font-size: 2.4rem; font-weight: 800; margin: 0; line-height: 1; }
.metric-label { font-size: 0.75rem; color: #90cdf4; margin: 6px 0 0; text-transform: uppercase;
                letter-spacing: 0.08em; }
.metric-sub { font-size: 0.85rem; font-weight: 600; margin: 4px 0 0; }

/* AQI gauge */
.aqi-gauge { text-align: center; }
.aqi-number { font-size: 4rem; font-weight: 900; line-height: 1; }
.aqi-cat { font-size: 1.1rem; font-weight: 700; margin-top: 4px; }

/* Alert banner */
.alert-banner {
    border-radius: 12px; padding: 14px 18px; margin: 8px 0;
    border-left: 4px solid; font-size: 0.9rem;
}
.alert-critical { background: rgba(197,48,48,0.15); border-color: #fc8181; color: #fc8181; }
.alert-warning  { background: rgba(214,158,46,0.15); border-color: #f6ad55; color: #f6ad55; }
.alert-info     { background: rgba(49,130,206,0.15); border-color: #63b3ed; color: #63b3ed; }
.alert-ok       { background: rgba(56,161,105,0.15); border-color: #68d391; color: #68d391; }

/* Section headers */
.section-header {
    font-size: 1rem; font-weight: 700; color: #63b3ed; text-transform: uppercase;
    letter-spacing: 0.1em; padding-bottom: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.2); margin-bottom: 14px;
}

/* Factor chips */
.factor-chip {
    display: inline-block; background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,179,0.3); border-radius: 20px;
    padding: 4px 12px; margin: 3px; font-size: 0.8rem; color: #90cdf4;
}

/* Resilience score */
.resilience-score {
    font-size: 3rem; font-weight: 900; text-align: center;
}

/* Disclaimer */
.disclaimer {
    font-size: 0.72rem; color: #718096; font-style: italic;
    border-left: 2px solid #4a5568; padding-left: 8px; margin: 8px 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1c2e 0%, #0d1628 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Plotly charts */
.js-plotly-plot { border-radius: 12px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: rgba(26,39,68,0.5); border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: #90cdf4; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #1a365d);
    color: white; border: 1px solid #63b3ed; border-radius: 10px;
    font-weight: 600; transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(99,179,237,0.3); }

/* Slider */
.stSlider [data-baseweb="slider"] { color: #63b3ed; }

/* Selectbox */
.stSelectbox label, .stMultiSelect label { color: #90cdf4; font-weight: 600; }

div[data-testid="metric-container"] {
    background: rgba(26,39,68,0.6); border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px; padding: 12px;
}

/* Data label badges */
.badge-measured { background: #276749; color: #9ae6b4; padding: 2px 8px; border-radius: 20px; font-size: 0.72rem; }
.badge-model    { background: #7b341e; color: #fbd38d; padding: 2px 8px; border-radius: 20px; font-size: 0.72rem; }
.badge-sim      { background: #2c5282; color: #bee3f8; padding: 2px 8px; border-radius: 20px; font-size: 0.72rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data + models ────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_PROC = os.path.join(os.path.dirname(__file__), "data", "processed")
MODELS_READY = os.path.exists(os.path.join(ARTIFACTS_DIR, "metadata.json"))

AQI_COLORS = {
    "Good": "#00B050", "Satisfactory": "#92D050", "Moderate": "#FFFF00",
    "Poor": "#FF4444", "Very Poor": "#AA00AA", "Severe": "#333333",
}

def aqi_category(v):
    if v <= 50:   return "Good", "#00c851"
    if v <= 100:  return "Satisfactory", "#a8e063"
    if v <= 200:  return "Moderate", "#f7c900"
    if v <= 300:  return "Poor", "#ff4444"
    if v <= 400:  return "Very Poor", "#aa44aa"
    return "Severe", "#333333"

def aqi_emoji(v):
    if v <= 50:   return "🟢"
    if v <= 100:  return "🟡"
    if v <= 200:  return "🟠"
    if v <= 300:  return "🔴"
    if v <= 400:  return "🟣"
    return "⚫"


@st.cache_data(ttl=300)
def load_data():
    if not MODELS_READY:
        return None, None
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    stations = pd.DataFrame(meta["stations"])
    stations = stations.rename(columns={"road_density": "road_density_index",
                                        "industrial": "industrial_zone"})
    readings = pd.read_parquet(os.path.join(DATA_PROC, "all_stations.parquet"))
    readings["timestamp"] = pd.to_datetime(readings["timestamp"])
    return stations, readings


@st.cache_resource
def load_models():
    if not MODELS_READY:
        return {}, None
    models = {}
    for h in [6, 12, 24, 48]:
        p = os.path.join(ARTIFACTS_DIR, f"xgb_{h}h.pkl")
        if os.path.exists(p):
            models[h] = joblib.load(p)
    iso_p = os.path.join(ARTIFACTS_DIR, "isolation_forest.pkl")
    iso = joblib.load(iso_p) if os.path.exists(iso_p) else None
    return models, iso


def get_latest_row(station_id: str, readings: pd.DataFrame) -> pd.Series:
    return readings[readings["station_id"] == station_id].sort_values("timestamp").iloc[-1]


def get_feature_vector(station_id: str, readings: pd.DataFrame, stations: pd.DataFrame):
    from src.processing.feature_engineer import engineer_features
    s = stations[stations["id"] == station_id].iloc[0]
    df = readings[readings["station_id"] == station_id].sort_values("timestamp").tail(48).copy()
    df["road_density"] = s["road_density_index"]
    df["industrial_zone"] = s["industrial_zone"]
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df = engineer_features(df)
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
        meta = json.load(f)
    fcols = meta["feature_cols"]
    available = [c for c in fcols if c in df.columns]
    vec = df[available].tail(1)
    # fill missing cols with 0
    for c in fcols:
        if c not in vec.columns:
            vec[c] = 0
    return vec[fcols]


def predict_horizons(fvec, models):
    results = {}
    for h, model in models.items():
        pred = float(model.predict(fvec)[0])
        noise = {6: 12, 12: 20, 24: 33, 48: 48}[h]
        cat, color = aqi_category(pred)
        results[h] = {"aqi": round(pred, 1), "low": round(max(0, pred - noise), 1),
                      "high": round(min(500, pred + noise), 1), "cat": cat, "color": color}
    return results


def get_shap_factors(fvec, model, horizon):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(fvec)
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
            meta = json.load(f)
        fcols = meta["feature_cols"]
        FACTOR_MAP = {
            "hour_sin": "⏰ Time of day (traffic)", "hour_cos": "⏰ Time of day (traffic)",
            "month_sin": "📅 Seasonal pattern", "month_cos": "📅 Seasonal pattern",
            "is_crop_burn_season": "🌾 Crop-residue burning season",
            "is_festival": "🎆 Festival emissions", "is_weekend": "🛣️ Weekend (lower traffic)",
            "is_night": "🌙 Nighttime conditions",
            "blh": "🌡️ Atmospheric mixing (BLH)", "temperature": "🌡️ Temperature",
            "humidity": "💧 Humidity", "wind_speed": "🌬️ Wind speed",
            "rainfall": "🌧️ Rainfall (washout)", "road_density": "🚗 Traffic density",
            "industrial_zone": "🏭 Industrial zone", "pm_ratio": "💨 Dust vs combustion ratio",
            "aqi_lag1h": "🔁 AQI 1h ago", "aqi_lag3h": "🔁 AQI 3h ago",
            "aqi_lag6h": "🔁 AQI 6h ago", "aqi_lag24h": "🔁 AQI yesterday",
            "pm25_lag24h": "🔁 PM2.5 yesterday", "aqi_roll3h_mean": "📈 3h AQI trend",
            "aqi_roll6h_mean": "📈 6h AQI trend", "aqi_roll24h_mean": "📈 24h AQI trend",
        }
        shap_series = pd.Series(dict(zip(fcols, sv[0]))).abs().nlargest(10)
        raw_shap = pd.Series(dict(zip(fcols, sv[0])))
        factors, seen = [], set()
        for feat in shap_series.index:
            label = FACTOR_MAP.get(feat, feat)
            if label in seen: continue
            seen.add(label)
            factors.append({"label": label, "shap": round(float(raw_shap[feat]), 2)})
            if len(factors) >= 5: break
        return factors
    except Exception:
        return []


def generate_alerts(preds: dict, station_name: str) -> list:
    alerts = []
    for h, p in preds.items():
        aqi = p["aqi"]
        if aqi > 300:
            alerts.append({
                "severity": "critical",
                "horizon": h,
                "msg": f"🚨 CRITICAL: AQI predicted {aqi:.0f} (Very Poor/Severe) in {h}h at {station_name}",
                "citizen": "Stay indoors. Use N95 masks outdoors. Avoid exercise.",
                "school": "Cancel all outdoor activities. Consider early school closure.",
                "worker": "Limit outdoor work. Provide N95 masks. Rotate shifts.",
                "authority": "Issue public health advisory. Deploy water sprinklers on major roads.",
            })
        elif aqi > 200:
            alerts.append({
                "severity": "warning",
                "horizon": h,
                "msg": f"⚠️ WARNING: AQI predicted {aqi:.0f} (Poor) in {h}h at {station_name}",
                "citizen": "Limit outdoor time. Sensitive groups stay indoors.",
                "school": "Cancel outdoor PE classes. Keep windows closed.",
                "worker": "Use dust masks. Take regular indoor breaks.",
                "authority": "Monitor air quality actively. Prepare health advisories.",
            })
        elif aqi > 100:
            alerts.append({
                "severity": "info",
                "horizon": h,
                "msg": f"ℹ️ NOTICE: AQI predicted {aqi:.0f} (Moderate) in {h}h at {station_name}",
                "citizen": "Sensitive individuals may experience discomfort outdoors.",
                "school": "No changes required; monitor updates.",
                "worker": "Standard precautions. Monitor if working near traffic.",
                "authority": "No immediate action required.",
            })
    return alerts


def resilience_score(station_id: str, readings: pd.DataFrame) -> dict:
    df = readings[readings["station_id"] == station_id]
    if df.empty:
        return {"overall": 0, "grade": "No Data",
                "factors": [0, 0, 0, 0, 0]}
    aqi = df["aqi"].values
    recent24 = aqi[-24:].mean() if len(aqi) >= 24 else aqi.mean()
    prev24 = aqi[-48:-24].mean() if len(aqi) >= 48 else recent24
    trend = min(100, max(0, 50 + (prev24 - recent24) / 3))
    bad_ratio = (aqi > 200).mean()
    weather_exp = max(0, 100 - bad_ratio * 150)
    vuln_map = {"DL": 30, "MU": 40, "KO": 38, "CH": 58, "BG": 62, "HY": 55}
    vuln = vuln_map.get(station_id[:2], 50)
    rd = readings[readings["station_id"] == station_id]["road_density"].iloc[-1] if "road_density" in readings.columns else 3
    green = max(10, min(90, 100 - rd * 15))
    autocorr = pd.Series(aqi[-168:]).autocorr(lag=6) if len(aqi) >= 168 else 0.5
    persist = max(10, min(90, 100 - abs(autocorr) * 60))
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    factors = [trend, weather_exp, vuln, green, persist]
    overall = round(float(np.dot(weights, factors)), 1)
    grade = ("High Resilience" if overall >= 75 else
             "Moderate Resilience" if overall >= 50 else
             "Low Resilience" if overall >= 30 else "Very Low Resilience")
    color = ("#00c851" if overall >= 75 else "#f7c900" if overall >= 50 else
             "#ff4444" if overall >= 30 else "#aa44aa")
    return {"overall": overall, "grade": grade, "color": color, "factors": factors}


# ── Not trained yet screen ────────────────────────────────────────────────────
if not MODELS_READY:
    st.markdown("""
    <div class="crip-header">
      <div>
        <p class="crip-title">🌬️ CRIP — CleanAir Resilience Intelligence Platform</p>
        <p class="crip-sub">Track 2: Clean Air & Climate Resilience</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.error("### ⚙️ Models not trained yet")
    st.info("Run the training pipeline first:")
    st.code("python src/models/train_xgboost.py", language="bash")
    st.caption("Training takes ~3–5 minutes on a standard laptop. Then reload this page.")
    st.stop()

# ── Load everything ───────────────────────────────────────────────────────────
stations, readings = load_data()
models, iso_forest = load_models()
with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
    meta_info = json.load(f)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌬️ CRIP")
    st.markdown("**CleanAir Resilience Intelligence Platform**")
    st.markdown("---")
    cities = sorted(stations["city"].unique())
    sel_city = st.selectbox("🏙️ Select City", cities, index=0)
    city_stations = stations[stations["city"] == sel_city]
    sel_station_name = st.selectbox(
        "📍 Select Station",
        city_stations["name"].tolist(),
        index=0
    )
    sel_station = city_stations[city_stations["name"] == sel_station_name].iloc[0]
    station_id = sel_station["id"]

    st.markdown("---")
    forecast_horizon = st.select_slider(
        "⏱️ Forecast Horizon",
        options=[6, 12, 24, 48],
        value=24,
        format_func=lambda x: f"{x}h"
    )
    st.markdown("---")
    st.markdown("**📊 Data Sources**")
    st.markdown("""
    - 🟢 [CPCB](https://airquality.cpcb.gov.in) — AQI data
    - 🟢 [Open-Meteo](https://open-meteo.com) — Weather
    - 🟢 [OpenAQ](https://openaq.org) — Supplementary
    - 🟢 [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) — Fire spots
    """)
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    ⚠️ Demo uses synthetic training data
    modelled on CPCB patterns.<br>
    Predictions are model estimates, not official measurements.
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
now = datetime.now().strftime("%d %b %Y, %H:%M IST")
latest = get_latest_row(station_id, readings)
cur_aqi = float(latest["aqi"])
cur_cat, cur_col = aqi_category(cur_aqi)
emoji = aqi_emoji(cur_aqi)

st.markdown(f"""
<div class="crip-header">
  <div>
    <p class="crip-title">🌬️ CRIP — CleanAir Resilience Intelligence Platform</p>
    <p class="crip-sub">📍 {sel_station_name}, {sel_city} &nbsp;|&nbsp; ⏱ {now}</p>
  </div>
  <div>
    <span class="crip-badge">Track 2: Clean Air & Climate Resilience</span>
    <br><br>
    <span class="badge-measured">🟢 Measured</span>
    <span class="badge-model">&nbsp;🟡 Model</span>
    <span class="badge-sim">&nbsp;🔵 Simulation</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🗺️ Hotspot Map", "🤖 AI Explanation",
    "⚠️ Early Warnings", "🔬 What-If Simulator"
])

# ═══════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════
with tab1:
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="aqi-number" style="color:{cur_col}">{cur_aqi:.0f}</div>
          <div class="aqi-cat" style="color:{cur_col}">{emoji} {cur_cat}</div>
          <div class="metric-label">Current AQI · {sel_station_name}</div>
          <div style="font-size:0.72rem;color:#718096;margin-top:6px;">
            🟢 Measured data · as of {latest['timestamp'].strftime('%H:%M') if hasattr(latest['timestamp'], 'strftime') else '–'}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#fbb6ce">{float(latest['pm25']):.1f}</div>
          <div class="metric-label">PM2.5 (µg/m³)</div>
          <div class="metric-sub" style="color:#fbb6ce">{'⚠️ High' if latest['pm25'] > 60 else '✅ Normal'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#fbd38d">{float(latest['pm10']):.1f}</div>
          <div class="metric-label">PM10 (µg/m³)</div>
          <div class="metric-sub" style="color:#fbd38d">{'⚠️ High' if latest['pm10'] > 100 else '✅ Normal'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#90cdf4">{float(latest['temperature']):.0f}°C</div>
          <div class="metric-label">Temperature</div>
          <div class="metric-sub" style="color:#a0aec0">
            💧 {float(latest['humidity']):.0f}% RH
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:#9ae6b4">{float(latest['wind_speed']):.1f}</div>
          <div class="metric-label">Wind (km/h)</div>
          <div class="metric-sub" style="color:#a0aec0">
            🌧️ {float(latest['rainfall']):.1f} mm rain
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast bar ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔮 AQI Forecast <span class="badge-model">🟡 Model Prediction</span></div>', unsafe_allow_html=True)
    fvec = get_feature_vector(station_id, readings, stations)
    preds = predict_horizons(fvec, models)

    fc_cols = st.columns(4)
    for i, (h, p) in enumerate(preds.items()):
        with fc_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{p['color']}55">
              <div style="font-size:0.75rem;color:#90cdf4;margin-bottom:4px;">+{h} hours</div>
              <div class="metric-value" style="color:{p['color']};font-size:2rem">{p['aqi']}</div>
              <div style="font-size:0.8rem;color:{p['color']};margin:4px 0">{p['cat']}</div>
              <div style="font-size:0.7rem;color:#718096">CI: {p['low']}–{p['high']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trend chart + Resilience ──────────────────────────────────────────────
    col_trend, col_resil = st.columns([2, 1])

    with col_trend:
        st.markdown('<div class="section-header">📈 72-Hour PM2.5 / AQI Trend <span class="badge-measured">🟢 Measured</span></div>', unsafe_allow_html=True)
        hist = readings[readings["station_id"] == station_id].sort_values("timestamp").tail(72)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist["aqi"], name="AQI",
            line=dict(color="#63b3ed", width=2.5),
            fill="tozeroy", fillcolor="rgba(99,179,237,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist["pm25"], name="PM2.5",
            line=dict(color="#fbb6ce", width=2), yaxis="y2"
        ))
        # Add forecast extension
        last_ts = hist["timestamp"].iloc[-1]
        fc_ts = [last_ts + timedelta(hours=h) for h in [6, 12, 24, 48]]
        fc_aqi = [preds[h]["aqi"] for h in [6, 12, 24, 48]]
        fc_low = [preds[h]["low"] for h in [6, 12, 24, 48]]
        fc_high = [preds[h]["high"] for h in [6, 12, 24, 48]]
        fig.add_trace(go.Scatter(
            x=[last_ts] + fc_ts, y=[cur_aqi] + fc_aqi, name="Forecast AQI",
            line=dict(color="#f6ad55", width=2.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=fc_ts + fc_ts[::-1],
            y=fc_high + fc_low[::-1],
            fill="toself", fillcolor="rgba(246,173,85,0.1)",
            line=dict(width=0), name="Forecast CI", showlegend=False
        ))
        # AQI category bands
        for threshold, color, label in [(50,"#00c851","Good"),(100,"#a8e063","Sat."),(200,"#f7c900","Mod."),(300,"#ff4444","Poor")]:
            fig.add_hline(y=threshold, line_dash="dash", line_color=color, opacity=0.3,
                         annotation_text=label, annotation_font_color=color, annotation_font_size=9)
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
            height=280, margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showgrid=True, title="AQI"),
            yaxis2=dict(overlaying="y", side="right", title="PM2.5 (µg/m³)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

    with col_resil:
        st.markdown('<div class="section-header">🛡️ Climate Resilience Score</div>', unsafe_allow_html=True)
        rs = resilience_score(station_id, readings)
        factor_names = ["AQI Trend", "Weather Exposure", "Vulnerability", "Green Cover", "Pollution Persist"]

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rs["overall"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": rs["grade"], "font": {"size": 12, "color": "#90cdf4"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#90cdf4"},
                "bar": {"color": rs.get("color", "#63b3ed")},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1, "bordercolor": "#2d3748",
                "steps": [
                    {"range": [0, 30], "color": "rgba(170,0,170,0.2)"},
                    {"range": [30, 50], "color": "rgba(255,68,68,0.2)"},
                    {"range": [50, 75], "color": "rgba(247,201,0,0.2)"},
                    {"range": [75, 100], "color": "rgba(0,200,81,0.2)"},
                ],
            },
            number={"font": {"color": rs.get("color", "#63b3ed"), "size": 40}},
        ))
        fig_gauge.update_layout(
            height=200, margin=dict(l=20, r=20, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")
        )
        st.plotly_chart(fig_gauge, width='stretch')

        # Radar
        fig_radar = go.Figure(go.Scatterpolar(
            r=rs["factors"] + [rs["factors"][0]],
            theta=factor_names + [factor_names[0]],
            fill="toself", fillcolor="rgba(99,179,237,0.15)",
            line=dict(color="#63b3ed", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)", tickcolor="#90cdf4"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", tickcolor="#90cdf4"),
            ),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0", size=9),
            showlegend=False, height=220, margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_radar, width='stretch')
        st.markdown('<div class="disclaimer">Informational indicator only. Not an official safety rating.</div>', unsafe_allow_html=True)

    # ── Current weather panel ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🌤️ Current Conditions</div>', unsafe_allow_html=True)
    w1, w2, w3, w4, w5, w6 = st.columns(6)
    weather_data = [
        ("🌡️", "Temperature", f"{float(latest['temperature']):.1f} °C", "#90cdf4"),
        ("💧", "Humidity", f"{float(latest['humidity']):.0f}%", "#63b3ed"),
        ("🌬️", "Wind Speed", f"{float(latest['wind_speed']):.1f} km/h", "#9ae6b4"),
        ("🌧️", "Rainfall", f"{float(latest['rainfall']):.2f} mm", "#bee3f8"),
        ("🌫️", "BLH", f"{float(latest['blh']):.0f} m", "#fbd38d"),
        ("🏭", "NO₂", f"{float(latest['no2']):.1f} µg/m³", "#fbb6ce"),
    ]
    for col, (ico, lbl, val, col_hex) in zip([w1, w2, w3, w4, w5, w6], weather_data):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:12px">
              <div style="font-size:1.5rem">{ico}</div>
              <div style="font-size:1rem;font-weight:700;color:{col_hex};margin:4px 0">{val}</div>
              <div style="font-size:0.7rem;color:#718096">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 2: HOTSPOT MAP
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🗺️ Pollution Hotspot Map — All Stations</div>', unsafe_allow_html=True)
    st.caption("Clustering based on 30-day average AQI. Click any marker for details.")

    # Compute station averages
    cutoff = readings["timestamp"].max() - timedelta(days=30)
    recent = readings[readings["timestamp"] >= cutoff]
    agg = recent.groupby("station_id").agg(
        avg_aqi=("aqi", "mean"), p90_aqi=("aqi", lambda x: np.percentile(x, 90)),
        avg_pm25=("pm25", "mean"), max_aqi=("aqi", "max"),
    ).reset_index()
    map_data = stations.merge(agg, left_on="id", right_on="station_id", how="left")
    map_data["avg_aqi"] = map_data["avg_aqi"].fillna(0)

    center_lat = map_data["lat"].mean()
    center_lon = map_data["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5,
                   tiles="CartoDB dark_matter", control_scale=True)

    for _, row in map_data.iterrows():
        aqi_val = row["avg_aqi"]
        cat, color = aqi_category(aqi_val)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(8, min(30, aqi_val / 10)),
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(f"""
                <b>{row['name']}</b><br>
                City: {row['city']}<br>
                Avg AQI (30d): <b>{aqi_val:.0f}</b> — {cat}<br>
                P90 AQI: {row['p90_aqi']:.0f}<br>
                Avg PM2.5: {row['avg_pm25']:.1f} µg/m³<br>
                Max AQI: {row['max_aqi']:.0f}<br>
                {"🏭 Industrial zone" if row.get('industrial_zone', 0) else ""}
            """, max_width=220),
            tooltip=f"{row['name']}: AQI {aqi_val:.0f} ({cat})",
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: rgba(0,0,0,0.8); padding: 12px; border-radius: 8px;
         font-family: Inter, sans-serif; font-size: 12px; color: white; border: 1px solid #444;">
      <b>AQI Legend</b><br>
      <span style="color:#00c851">●</span> Good (0–50)<br>
      <span style="color:#a8e063">●</span> Satisfactory (51–100)<br>
      <span style="color:#f7c900">●</span> Moderate (101–200)<br>
      <span style="color:#ff4444">●</span> Poor (201–300)<br>
      <span style="color:#aa44aa">●</span> Very Poor (301–400)<br>
      <span style="color:#333">●</span> Severe (401+)<br>
      <i>Circle size ∝ AQI severity</i>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=None, height=520)

    # Station table
    st.markdown('<div class="section-header">📋 Station Rankings (30-day avg AQI)</div>', unsafe_allow_html=True)
    display_df = map_data[["name", "city", "avg_aqi", "p90_aqi", "avg_pm25", "max_aqi"]].copy()
    display_df.columns = ["Station", "City", "Avg AQI", "P90 AQI", "Avg PM2.5", "Max AQI"]
    display_df = display_df.sort_values("Avg AQI", ascending=False).reset_index(drop=True)
    display_df["Avg AQI"] = display_df["Avg AQI"].round(1)
    display_df["P90 AQI"] = display_df["P90 AQI"].round(1)
    display_df["Avg PM2.5"] = display_df["Avg PM2.5"].round(1)
    display_df["Max AQI"] = display_df["Max AQI"].round(1)
    st.dataframe(display_df, width='stretch', height=300)

# ═══════════════════════════════════════════════════════
# TAB 3: AI EXPLANATION
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 AI Explanation — Why This Prediction? <span class="badge-model">🟡 Model Attribution</span></div>', unsafe_allow_html=True)

    col_exp, col_shap = st.columns([1, 1.5])
    with col_exp:
        sel_h = st.selectbox("Explain forecast horizon:", [6, 12, 24, 48],
                             index=[6, 12, 24, 48].index(forecast_horizon),
                             format_func=lambda x: f"+{x} hours")
        pred_val = preds[sel_h]["aqi"]
        pred_cat = preds[sel_h]["cat"]
        pred_col = preds[sel_h]["color"]

        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:16px">
          <div style="font-size:0.8rem;color:#90cdf4">+{sel_h}h Prediction for {sel_station_name}</div>
          <div style="font-size:2.8rem;font-weight:900;color:{pred_col}">{pred_val:.0f}</div>
          <div style="color:{pred_col};font-weight:700">{aqi_emoji(pred_val)} {pred_cat}</div>
          <div style="font-size:0.7rem;color:#718096;margin-top:6px">
            CI: {preds[sel_h]['low']} – {preds[sel_h]['high']} AQI units
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Factor context
        is_crop = latest["timestamp"].month in [10, 11]
        is_festival = (latest["timestamp"].month == 10 and 20 <= latest["timestamp"].day <= 28) or \
                      (latest["timestamp"].month == 11 and 1 <= latest["timestamp"].day <= 5)
        is_industrial = sel_station.get("industrial_zone", 0)
        is_night = latest["timestamp"].hour >= 22 or latest["timestamp"].hour <= 6
        context = []
        if is_crop:
            context.append("🌾 Crop-residue burning season active (Oct–Nov)")
        if is_festival:
            context.append("🎆 Festival period — elevated firework emissions")
        if is_industrial:
            context.append("🏭 Station is in/near an industrial zone")
        if is_night:
            context.append("🌙 Night hours — lower boundary layer traps pollutants")
        if float(latest["wind_speed"]) < 2:
            context.append("🌬️ Very low wind speed — poor dispersion")
        if float(latest["blh"]) < 400:
            context.append("⚠️ Low boundary layer height — atmospheric inversion likely")
        if context:
            st.markdown("**📌 Contextual factors active now:**")
            for c in context:
                st.markdown(f"<div class='factor-chip'>{c}</div>", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("""
        <div class="disclaimer">
        🔵 SHAP values show which features influence the model's prediction for this specific input.
        They represent model attribution, not direct causal measurement.
        </div>
        """, unsafe_allow_html=True)

    with col_shap:
        with st.spinner("Computing SHAP explanations..."):
            factors = get_shap_factors(fvec, models[sel_h], sel_h)
        if factors:
            st.markdown(f"**Top factors for +{sel_h}h prediction:**")
            for i, f in enumerate(factors):
                direction_col = "#ff6b6b" if f["shap"] > 0 else "#68d391"
                direction_icon = "📈 increases" if f["shap"] > 0 else "📉 decreases"
                bar_width = min(100, abs(f["shap"]) / max(abs(ff["shap"]) for ff in factors) * 100)
                st.markdown(f"""
                <div style="margin-bottom:14px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:0.88rem;color:#e2e8f0">{i+1}. {f['label']}</span>
                    <span style="font-size:0.82rem;color:{direction_col}">{direction_icon} AQI by ~{abs(f['shap']):.1f}</span>
                  </div>
                  <div style="background:rgba(255,255,255,0.05);border-radius:6px;height:10px;overflow:hidden">
                    <div style="width:{bar_width}%;height:100%;background:{direction_col};border-radius:6px;
                                opacity:0.85;transition:width 0.5s"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Plotly SHAP bar
            fig_shap = go.Figure(go.Bar(
                x=[f["shap"] for f in factors],
                y=[f["label"] for f in factors],
                orientation="h",
                marker=dict(
                    color=["#ff6b6b" if f["shap"] > 0 else "#68d391" for f in factors],
                    opacity=0.85
                ),
                text=[f"{f['shap']:+.1f}" for f in factors],
                textposition="outside",
            ))
            fig_shap.update_layout(
                title=f"SHAP Feature Contributions (+{sel_h}h forecast)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Inter", size=11),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="SHAP value (impact on AQI)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                height=300, margin=dict(l=0, r=60, t=40, b=0),
            )
            st.plotly_chart(fig_shap, width='stretch')
        else:
            st.info("SHAP values computing — ensure shap package is installed.")

# ═══════════════════════════════════════════════════════
# TAB 4: EARLY WARNINGS
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">⚠️ Early Warning System</div>', unsafe_allow_html=True)
    alerts = generate_alerts(preds, sel_station_name)

    if not alerts:
        st.markdown("""
        <div class="alert-banner alert-ok">
          ✅ No active warnings for the next 48 hours at this station.
          AQI forecasts are within acceptable ranges.
        </div>
        """, unsafe_allow_html=True)
    else:
        for a in alerts:
            cls = "alert-critical" if a["severity"] == "critical" else "alert-warning" if a["severity"] == "warning" else "alert-info"
            st.markdown(f'<div class="alert-banner {cls}"><b>{a["msg"]}</b></div>', unsafe_allow_html=True)

    if alerts:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🎯 Recommended Actions by User Group</div>', unsafe_allow_html=True)
        worst = max(alerts, key=lambda x: preds[x["horizon"]]["aqi"])
        ac1, ac2, ac3, ac4 = st.columns(4)
        groups = [
            (ac1, "👥 Citizens", worst["citizen"], "#63b3ed"),
            (ac2, "🏫 Schools", worst["school"], "#fbd38d"),
            (ac3, "👷 Workers", worst["worker"], "#fbb6ce"),
            (ac4, "🏛️ Authorities", worst["authority"], "#9ae6b4"),
        ]
        for col, title, action, color in groups:
            with col:
                st.markdown(f"""
                <div class="metric-card" style="text-align:left;border-color:{color}44">
                  <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:10px">{title}</div>
                  <div style="font-size:0.85rem;line-height:1.6">{action}</div>
                </div>
                """, unsafe_allow_html=True)

    # Historical alert frequency
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Historical AQI Category Distribution (Last 30 Days)</div>', unsafe_allow_html=True)
    hist30 = readings[readings["station_id"] == station_id].tail(30 * 24)
    cats = pd.cut(
        hist30["aqi"],
        bins=[0, 50, 100, 200, 300, 400, 501],
        labels=["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    ).value_counts().sort_index()
    cat_colors = ["#00c851", "#a8e063", "#f7c900", "#ff4444", "#aa44aa", "#333333"]
    fig_cat = go.Figure(go.Bar(
        x=cats.index.tolist(), y=cats.values,
        marker=dict(color=cat_colors[:len(cats)], opacity=0.85),
        text=cats.values, textposition="outside",
    ))
    fig_cat.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Hours"),
        height=250, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_cat, width='stretch')

# ═══════════════════════════════════════════════════════
# TAB 5: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">🔬 What-If Policy Simulator <span class="badge-sim">🔵 Model Simulation</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer" style="margin-bottom:16px">
    ⚠️ This is a <b>statistical model simulation</b> based on feature perturbation of a trained XGBoost model.
    It is NOT a physical atmospheric model. Results show <b>directional estimates only</b>
    and should not be used for regulatory or policy decisions.
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 1.5])

    with sim_col1:
        st.markdown("**🎛️ Scenario Controls**")
        sim_h = st.selectbox("Forecast horizon for simulation:", [6, 12, 24, 48],
                             format_func=lambda x: f"+{x} hours", index=2)
        traffic_red = st.slider("🚗 Traffic Reduction", 0.0, 0.8, 0.0, 0.05,
                                format="%.0f%%", help="Fraction of traffic removed (e.g. 0.30 = 30% fewer vehicles)")
        green_cov = st.slider("🌳 Green Cover Increase", 0.0, 0.5, 0.0, 0.05,
                              format="%.0f%%", help="Fraction increase in green/forested area")
        rain_mm = st.slider("🌧️ Simulated Rainfall (mm)", 0.0, 50.0, 0.0, 1.0)
        ind_red = st.slider("🏭 Industrial Emission Cut", 0.0, 0.8, 0.0, 0.05,
                            format="%.0f%%", help="Fraction reduction in industrial output")

        run_sim = st.button("▶️ Run Simulation", type="primary")

    with sim_col2:
        if run_sim or True:  # Always show result
            from src.simulation.whatif import simulate
            result = simulate(
                fvec.copy(), horizon=sim_h,
                traffic_reduction=traffic_red,
                green_cover_increase=green_cov,
                rainfall_mm=rain_mm,
                industrial_reduction=ind_red,
            )

            base_aqi = result["baseline_aqi"]
            sim_aqi = result["scenario_aqi"]
            delta = result["delta"]
            pct = result["percent_improvement"]
            base_cat, base_col = aqi_category(base_aqi)
            sim_cat, sim_col = aqi_category(sim_aqi)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:0.75rem;color:#90cdf4;margin-bottom:4px">Baseline AQI (+{sim_h}h)</div>
                  <div style="font-size:2.5rem;font-weight:900;color:{base_col}">{base_aqi}</div>
                  <div style="font-size:0.85rem;color:{base_col}">{base_cat}</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:0.75rem;color:#90cdf4;margin-bottom:4px">Simulated AQI</div>
                  <div style="font-size:2.5rem;font-weight:900;color:{sim_col}">{sim_aqi}</div>
                  <div style="font-size:0.85rem;color:{sim_col}">{sim_cat}</div>
                </div>
                """, unsafe_allow_html=True)
            with r3:
                delta_col = "#68d391" if delta > 0 else "#fc8181"
                arrow = "▼" if delta > 0 else "▲"
                st.markdown(f"""
                <div class="metric-card">
                  <div style="font-size:0.75rem;color:#90cdf4;margin-bottom:4px">Improvement</div>
                  <div style="font-size:2.5rem;font-weight:900;color:{delta_col}">{arrow}{abs(delta):.0f}</div>
                  <div style="font-size:0.85rem;color:{delta_col}">{abs(pct):.1f}% {"better" if delta > 0 else "worse"}</div>
                </div>
                """, unsafe_allow_html=True)

            # Before / After comparison gauge
            fig_sim = go.Figure()
            fig_sim.add_trace(go.Bar(
                name="Baseline", x=["AQI Forecast"], y=[base_aqi],
                marker=dict(color=base_col, opacity=0.8),
                text=[f"{base_aqi:.0f}"], textposition="outside",
            ))
            fig_sim.add_trace(go.Bar(
                name="Simulation", x=["AQI Forecast"], y=[sim_aqi],
                marker=dict(color=sim_col, opacity=0.8),
                text=[f"{sim_aqi:.0f}"], textposition="outside",
            ))
            for threshold, color, label in [(50,"#00c851","Good"),(100,"#a8e063","Sat."),(200,"#f7c900","Mod."),(300,"#ff4444","Poor")]:
                fig_sim.add_hline(y=threshold, line_dash="dash", line_color=color, opacity=0.4,
                                 annotation_text=label, annotation_font_size=9)
            fig_sim.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0", family="Inter"),
                barmode="group", height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, max(base_aqi, sim_aqi) * 1.3]),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_sim, width='stretch')

            if result["scenario_description"] != "No change":
                st.success(f"📌 Scenario: **{result['scenario_description']}**")
            st.markdown(f'<div class="disclaimer">{result["disclaimer"]}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#4a5568;font-size:0.78rem;padding:12px 0">
  🌬️ <b>CRIP — CleanAir Resilience Intelligence Platform</b> &nbsp;|&nbsp;
  Track 2: Clean Air & Climate Resilience &nbsp;|&nbsp;
  Built with ❤️ for India's 1.4 billion people<br>
  Data: CPCB · Open-Meteo · OpenAQ · NASA FIRMS &nbsp;|&nbsp;
  <span style="color:#718096">Model predictions are estimates. Not official measurements or health advisories.</span>
</div>
""", unsafe_allow_html=True)


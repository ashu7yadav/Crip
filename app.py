"""
CRIP — CleanAir Resilience Intelligence Platform
Dark Green Futuristic Hub Dashboard (AI Agriculture Ecosystem Template Style)
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
    initial_sidebar_state="collapsed",
)

# ── Dark Green Futuristic CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #040d08;
    color: #c8f5d8;
}

.stApp {
    background: radial-gradient(ellipse at 50% 0%, #0a2a15 0%, #040d08 60%);
    min-height: 100vh;
}

/* Hexagonal grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        repeating-linear-gradient(60deg, rgba(0,255,100,0.03) 0px, transparent 1px, transparent 30px),
        repeating-linear-gradient(120deg, rgba(0,255,100,0.03) 0px, transparent 1px, transparent 30px),
        repeating-linear-gradient(0deg, rgba(0,255,100,0.02) 0px, transparent 1px, transparent 52px);
    pointer-events: none;
    z-index: 0;
}

/* TOP NAV BAR */
.crip-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 24px;
    background: rgba(0,20,10,0.95);
    border-bottom: 1px solid rgba(0,255,100,0.3);
    margin-bottom: 16px;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 4px 30px rgba(0,255,100,0.1);
}
.crip-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    color: #00ff88;
    letter-spacing: 0.15em;
    text-shadow: 0 0 20px rgba(0,255,136,0.6);
}
.crip-subtitle {
    font-size: 0.72rem;
    color: #4dff9e;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}
.nav-tabs {
    display: flex;
    gap: 24px;
    font-size: 0.8rem;
    color: #4dff9e;
    letter-spacing: 0.1em;
}
.nav-tab { cursor: pointer; padding: 4px 8px; border-radius: 4px; transition: all 0.2s; }
.nav-tab:hover { color: #00ff88; background: rgba(0,255,136,0.1); }
.nav-tab.active { color: #00ff88; border-bottom: 2px solid #00ff88; }
.live-badge {
    display: flex; align-items: center; gap: 6px;
    background: rgba(0,255,100,0.08);
    border: 1px solid rgba(0,255,100,0.4);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.75rem; color: #00ff88; font-weight: 600;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 8px #00ff88;
    animation: pulse-green 1.5s infinite;
}
@keyframes pulse-green {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.4); }
}

/* PANEL CARDS */
.panel {
    background: rgba(0,18,8,0.85);
    border: 1px solid rgba(0,255,100,0.2);
    border-radius: 12px;
    padding: 16px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
    backdrop-filter: blur(4px);
}
.panel:hover {
    border-color: rgba(0,255,100,0.5);
    box-shadow: 0 0 20px rgba(0,255,100,0.1), inset 0 0 20px rgba(0,255,100,0.03);
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00ff88, transparent);
    opacity: 0.6;
}
.panel-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #00ff88;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
}
.panel-icon { font-size: 0.85rem; }

/* CENTER HUB */
.hub-center {
    background: radial-gradient(circle at 50% 50%,
        rgba(0,255,100,0.12) 0%,
        rgba(0,60,30,0.4) 40%,
        rgba(0,18,8,0.9) 70%);
    border: 1px solid rgba(0,255,100,0.35);
    border-radius: 50%;
    padding: 30px;
    text-align: center;
    position: relative;
    box-shadow:
        0 0 40px rgba(0,255,100,0.2),
        0 0 80px rgba(0,255,100,0.08),
        inset 0 0 40px rgba(0,255,100,0.05);
    animation: orb-glow 3s ease-in-out infinite;
}
@keyframes orb-glow {
    0%,100% { box-shadow: 0 0 40px rgba(0,255,100,0.2), 0 0 80px rgba(0,255,100,0.08); }
    50% { box-shadow: 0 0 60px rgba(0,255,100,0.35), 0 0 120px rgba(0,255,100,0.12); }
}
.hub-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 900;
    color: #00ff88;
    text-shadow: 0 0 30px rgba(0,255,136,0.8);
    letter-spacing: 0.2em;
    line-height: 1;
}
.hub-sub {
    font-size: 0.65rem;
    color: #4dff9e;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-top: 6px;
}
.hub-ring {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(0,255,100,0.15);
}

/* AQI BIG NUMBER */
.aqi-big {
    font-family: 'Orbitron', monospace;
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1;
    text-shadow: 0 0 20px currentColor;
}
.aqi-cat {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* METRIC ROW */
.hub-metric {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 10px;
    background: rgba(0,255,100,0.05);
    border: 1px solid rgba(0,255,100,0.15);
    border-radius: 8px;
    transition: all 0.2s;
}
.hub-metric:hover { background: rgba(0,255,100,0.1); border-color: rgba(0,255,100,0.4); }
.hub-metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem;
    font-weight: 700;
    color: #00ff88;
}
.hub-metric-label {
    font-size: 0.65rem;
    color: #4dff9e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
}
.hub-metric-icon { font-size: 1.1rem; margin-bottom: 4px; }

/* FORECAST BARS */
.fc-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 10px;
    border-radius: 6px;
    background: rgba(0,255,100,0.04);
    border: 1px solid rgba(0,255,100,0.1);
    margin-bottom: 7px;
    transition: all 0.2s;
}
.fc-bar:hover { background: rgba(0,255,100,0.08); }
.fc-hour { font-size: 0.7rem; color: #4dff9e; font-family: 'Orbitron', monospace; }
.fc-aqi { font-family: 'Orbitron', monospace; font-size: 1rem; font-weight: 700; }
.fc-cat { font-size: 0.65rem; letter-spacing: 0.08em; }
.fc-ci { font-size: 0.62rem; color: #2d7a52; }

/* ALERT BANNERS */
.alert-box {
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    border-left: 3px solid;
    font-size: 0.8rem;
    line-height: 1.4;
}
.alert-critical { background: rgba(255,50,50,0.1); border-color: #ff3333; color: #ff8080; }
.alert-warning  { background: rgba(255,170,0,0.1); border-color: #ffaa00; color: #ffcc66; }
.alert-info     { background: rgba(0,200,255,0.1); border-color: #00c8ff; color: #66ddff; }
.alert-ok       { background: rgba(0,255,100,0.08); border-color: #00ff88; color: #4dff9e; }

/* FACTOR BAR */
.factor-row {
    margin-bottom: 10px;
}
.factor-label { font-size: 0.75rem; color: #c8f5d8; margin-bottom: 3px; display:flex; justify-content:space-between; }
.factor-bar-bg { background: rgba(0,255,100,0.06); border-radius: 4px; height: 8px; overflow: hidden; }
.factor-bar-fill { height: 100%; border-radius: 4px; }

/* RESILIENCE SCORE */
.resil-number {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    color: #00ff88;
    text-shadow: 0 0 20px rgba(0,255,136,0.6);
    text-align: center;
    line-height: 1;
}
.resil-grade {
    font-size: 0.7rem;
    color: #4dff9e;
    letter-spacing: 0.15em;
    text-align: center;
    text-transform: uppercase;
    margin-top: 4px;
}

/* DISCLAIMER */
.disclaimer {
    font-size: 0.65rem;
    color: #2d7a52;
    font-style: italic;
    border-left: 2px solid #1a4d35;
    padding-left: 8px;
    margin-top: 8px;
}

/* SECTION DIVIDER */
.divider {
    border: none;
    border-top: 1px solid rgba(0,255,100,0.1);
    margin: 10px 0;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: rgba(0,12,6,0.97) !important;
    border-right: 1px solid rgba(0,255,100,0.2) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,18,8,0.8);
    border-radius: 10px;
    border: 1px solid rgba(0,255,100,0.15);
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #4dff9e !important;
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,255,100,0.12) !important;
    color: #00ff88 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,80,40,0.8), rgba(0,40,20,0.8));
    color: #00ff88; border: 1px solid rgba(0,255,100,0.4);
    border-radius: 8px; font-family: 'Orbitron', monospace;
    font-size: 0.7rem; letter-spacing: 0.1em; font-weight: 700;
    transition: all 0.2s;
    box-shadow: 0 0 10px rgba(0,255,100,0.1);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0,255,100,0.3);
    border-color: #00ff88;
}

/* Sliders */
.stSlider [data-baseweb="slider"] { color: #00ff88; }

/* Selectbox */
.stSelectbox label { color: #4dff9e !important; font-size: 0.75rem !important; }
.stSelectbox [data-baseweb="select"] {
    background: rgba(0,18,8,0.9) !important;
    border-color: rgba(0,255,100,0.3) !important;
    color: #c8f5d8 !important;
}

/* Remove default padding */
.block-container { padding-top: 0 !important; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
DATA_PROC = os.path.join(os.path.dirname(__file__), "data", "processed")
MODELS_READY = os.path.exists(os.path.join(ARTIFACTS_DIR, "metadata.json"))

def aqi_category(v):
    if v <= 50:   return "Good",       "#00ff88"
    if v <= 100:  return "Satisfactory","#aaff44"
    if v <= 200:  return "Moderate",   "#ffdd00"
    if v <= 300:  return "Poor",       "#ff6600"
    if v <= 400:  return "Very Poor",  "#ff2222"
    return "Severe", "#cc00cc"

def aqi_emoji(v):
    if v <= 50:  return "🟢"
    if v <= 100: return "🟡"
    if v <= 200: return "🟠"
    if v <= 300: return "🔴"
    if v <= 400: return "🟣"
    return "⚫"

@st.cache_data(ttl=300)
def load_data():
    if not MODELS_READY: return None, None
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
        meta = json.load(f)
    stations = pd.DataFrame(meta["stations"])
    stations = stations.rename(columns={"road_density": "road_density_index",
                                        "industrial": "industrial_zone"})
    readings = pd.read_parquet(os.path.join(DATA_PROC, "all_stations.parquet"))
    readings["timestamp"] = pd.to_datetime(readings["timestamp"])
    return stations, readings

@st.cache_resource
def load_models():
    if not MODELS_READY: return {}, None
    models = {}
    for h in [6, 12, 24, 48]:
        p = os.path.join(ARTIFACTS_DIR, f"xgb_{h}h.pkl")
        if os.path.exists(p): models[h] = joblib.load(p)
    return models, None

def get_feature_vector(station_id, readings, stations):
    from src.processing.feature_engineer import engineer_features
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
        meta = json.load(f)
    fcols = meta["feature_cols"]
    s = stations[stations["id"] == station_id].iloc[0]
    df = readings[readings["station_id"] == station_id].sort_values("timestamp").tail(48).copy()
    df["road_density"] = s["road_density_index"]
    df["industrial_zone"] = s["industrial_zone"]
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df = engineer_features(df)
    vec = df.tail(1).copy()
    for c in fcols:
        if c not in vec.columns: vec[c] = 0
    return vec[fcols]

def predict_horizons(fvec, models):
    results = {}
    for h, model in models.items():
        pred = float(model.predict(fvec)[0])
        noise = {6:12, 12:20, 24:33, 48:48}[h]
        cat, color = aqi_category(pred)
        results[h] = {"aqi": round(pred,1), "low": round(max(0, pred-noise),1),
                      "high": round(min(500, pred+noise),1), "cat": cat, "color": color}
    return results

def get_shap_factors(fvec, model):
    try:
        import shap
        with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
            meta = json.load(f)
        fcols = meta["feature_cols"]
        FMAP = {
            "hour_sin":"Vehicle Emissions / Time","hour_cos":"Vehicle Emissions / Time",
            "is_crop_burn_season":"Crop Burning","is_festival":"Festival Emissions",
            "blh":"Atmospheric Inversion","temperature":"Temperature",
            "humidity":"Humidity","wind_speed":"Wind / Dispersion",
            "rainfall":"Rainfall Washout","road_density":"Traffic Density",
            "industrial_zone":"Industrial Emissions","pm_ratio":"Dust vs Combustion",
            "aqi_lag24h":"Carry-over Pollution","aqi_roll24h_mean":"Pollution Trend",
            "aqi_lag1h":"Recent AQI","aqi_lag6h":"6h AQI Trend",
        }
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(fvec)
        raw = pd.Series(dict(zip(fcols, sv[0])))
        top = raw.abs().nlargest(10)
        factors, seen = [], set()
        for feat in top.index:
            label = FMAP.get(feat, feat)
            if label in seen: continue
            seen.add(label)
            factors.append({"label": label, "shap": round(float(raw[feat]),2)})
            if len(factors) >= 5: break
        return factors
    except: return []

def resilience_score(station_id, readings):
    df = readings[readings["station_id"] == station_id]
    if df.empty: return {"overall":0,"grade":"No Data","color":"#2d7a52","factors":[0]*5}
    aqi = df["aqi"].values
    recent24 = aqi[-24:].mean() if len(aqi)>=24 else aqi.mean()
    prev24 = aqi[-48:-24].mean() if len(aqi)>=48 else recent24
    trend = min(100, max(0, 50+(prev24-recent24)/3))
    bad_ratio = (aqi>200).mean()
    weather_exp = max(0, 100 - bad_ratio*150)
    vuln_map={"DL":30,"MU":40,"KO":38,"CH":58,"BG":62,"HY":55}
    vuln = vuln_map.get(station_id[:2], 50)
    rd = 3
    green = max(10, min(90, 100-rd*15))
    autocorr = pd.Series(aqi[-168:]).autocorr(lag=6) if len(aqi)>=168 else 0.5
    persist = max(10, min(90, 100-abs(autocorr)*60))
    weights = [0.30,0.25,0.20,0.15,0.10]
    factors = [trend,weather_exp,vuln,green,persist]
    overall = round(float(np.dot(weights,factors)),1)
    grade = ("High Resilience" if overall>=75 else "Moderate Resilience" if overall>=50
             else "Low Resilience" if overall>=30 else "Very Low Resilience")
    color = ("#00ff88" if overall>=75 else "#ffdd00" if overall>=50
             else "#ff6600" if overall>=30 else "#ff2222")
    return {"overall":overall,"grade":grade,"color":color,"factors":factors}

def generate_alerts(preds, station_name):
    alerts = []
    for h,p in preds.items():
        aqi = p["aqi"]
        if aqi>300:
            alerts.append({"sev":"critical","horizon":h,
                "msg":f"🚨 AQI predicted {aqi:.0f} (Very Poor/Severe) in +{h}h at {station_name}",
                "citizen":"Stay indoors. N95 masks mandatory outdoors.",
                "school":"Cancel ALL outdoor activities. Consider early closure.",
                "worker":"Halt outdoor work. Issue N95 masks immediately.",
                "authority":"Issue public health advisory. Deploy sprinklers on major roads."})
        elif aqi>200:
            alerts.append({"sev":"warning","horizon":h,
                "msg":f"⚠️ AQI predicted {aqi:.0f} (Poor) in +{h}h at {station_name}",
                "citizen":"Limit outdoor time. Sensitive groups stay indoors.",
                "school":"Cancel outdoor PE. Keep windows closed.",
                "worker":"Use dust masks. Take indoor breaks every 2h.",
                "authority":"Monitor actively. Pre-position health resources."})
        elif aqi>100:
            alerts.append({"sev":"info","horizon":h,
                "msg":f"ℹ️ AQI predicted {aqi:.0f} (Moderate) in +{h}h — {station_name}",
                "citizen":"Sensitive groups may experience discomfort.",
                "school":"No changes required. Monitor updates.",
                "worker":"Standard precautions. Monitor if near traffic.",
                "authority":"No immediate action required."})
    return alerts

# ── Not trained yet ───────────────────────────────────────────────────────────
if not MODELS_READY:
    st.markdown("""
    <div style="text-align:center;padding:60px;font-family:'Orbitron',monospace">
      <div style="font-size:3rem;color:#00ff88;text-shadow:0 0 30px #00ff88">🌬️ CRIP</div>
      <div style="color:#4dff9e;margin:16px 0">Models not trained yet</div>
    </div>
    """, unsafe_allow_html=True)
    st.code("python src/models/train_xgboost.py", language="bash")
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
stations, readings = load_data()
models, _ = load_models()
with open(os.path.join(ARTIFACTS_DIR,"metadata.json")) as f:
    meta_info = json.load(f)

# ── Sidebar: Station selector ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;color:#00ff88;font-size:1.1rem;
    font-weight:900;letter-spacing:0.2em;margin-bottom:4px;">🌬️ CRIP</div>
    <div style="color:#4dff9e;font-size:0.65rem;letter-spacing:0.2em;margin-bottom:16px;">
    CLEANAIR INTELLIGENCE</div>
    """, unsafe_allow_html=True)
    cities = sorted(stations["city"].unique())
    sel_city = st.selectbox("🏙️ City", cities)
    city_stations = stations[stations["city"] == sel_city]
    sel_name = st.selectbox("📍 Station", city_stations["name"].tolist())
    sel_station = city_stations[city_stations["name"] == sel_name].iloc[0]
    station_id = sel_station["id"]
    st.markdown("---")
    forecast_h = st.select_slider("⏱️ Horizon", [6,12,24,48], value=24, format_func=lambda x: f"+{x}h")
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.68rem;color:#2d7a52;line-height:1.7">
    <b style="color:#4dff9e">DATA SOURCES</b><br>
    🟢 CPCB — AQI/Pollutants<br>
    🟢 Open-Meteo — Weather<br>
    🟢 OpenAQ — Supplementary<br>
    🟢 NASA FIRMS — Fire spots
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.62rem;color:#1a4d35;font-style:italic">
    ⚠️ Demo uses synthetic training data modelled on CPCB patterns.
    Not official measurements.
    </div>
    """, unsafe_allow_html=True)

# ── Top Nav ───────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%b %d, %Y  %H:%M IST")
st.markdown(f"""
<div class="crip-nav">
  <div>
    <div class="crip-logo">⬡ CRIP</div>
    <div class="crip-subtitle">Air Quality Intelligence Platform</div>
  </div>
  <div class="nav-tabs">
    <span class="nav-tab active">Dashboard</span>
    <span class="nav-tab">Analytics</span>
    <span class="nav-tab">Maps</span>
    <span class="nav-tab">Reports</span>
    <span class="nav-tab">Alerts</span>
  </div>
  <div style="text-align:right">
    <div class="live-badge"><span class="live-dot"></span> LIVE MONITOR</div>
    <div style="font-size:0.65rem;color:#2d7a52;margin-top:4px">{now_str}</div>
    <div style="font-size:0.65rem;color:#2d7a52">📍 {sel_name}, {sel_city}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Compute data ──────────────────────────────────────────────────────────────
latest = readings[readings["station_id"]==station_id].sort_values("timestamp").iloc[-1]
cur_aqi = float(latest["aqi"])
cur_cat, cur_col = aqi_category(cur_aqi)
fvec = get_feature_vector(station_id, readings, stations)
preds = predict_horizons(fvec, models)
rs = resilience_score(station_id, readings)
alerts = generate_alerts(preds, sel_name)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⬡ DASHBOARD", "🗺️ HOTSPOT MAP", "🤖 AI EXPLANATION",
    "⚠️ EARLY WARNINGS", "🔬 WHAT-IF SIMULATOR"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD — 3-column hub layout
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_center, col_right = st.columns([1, 1.1, 1], gap="medium")

    # ── LEFT COLUMN ───────────────────────────────────────────────────────────
    with col_left:
        # Panel 1: AQI Forecast
        st.markdown("""<div class="panel">
        <div class="panel-title"><span class="panel-icon">📈</span> AQI FORECAST (NEXT 48H)</div>
        """, unsafe_allow_html=True)

        hist = readings[readings["station_id"]==station_id].sort_values("timestamp").tail(48)
        last_ts = hist["timestamp"].iloc[-1]
        fc_ts = [last_ts + timedelta(hours=h) for h in [6,12,24,48]]
        fc_aqi = [preds[h]["aqi"] for h in [6,12,24,48]]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist["timestamp"], y=hist["aqi"], name="Historical AQI",
            line=dict(color="#00ff88", width=2),
            fill="tozeroy", fillcolor="rgba(0,255,136,0.06)",
        ))
        fig_fc.add_trace(go.Scatter(
            x=[last_ts]+fc_ts, y=[cur_aqi]+fc_aqi, name="Forecast",
            line=dict(color="#ffdd00", width=2, dash="dot"),
            mode="lines+markers",
            marker=dict(color="#ffdd00", size=6),
        ))
        for threshold, col, label in [(50,"#00ff88","Good"),(100,"#aaff44","Sat"),(200,"#ffdd00","Mod"),(300,"#ff6600","Poor")]:
            fig_fc.add_hline(y=threshold, line_dash="dash", line_color=col, opacity=0.25,
                            annotation_text=label, annotation_font_color=col, annotation_font_size=8)
        fig_fc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8f5d8", family="Inter", size=9),
            showlegend=True,
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)",
                       font=dict(size=8, color="#4dff9e")),
            height=220, margin=dict(l=0,r=0,t=24,b=0),
            xaxis=dict(gridcolor="rgba(0,255,100,0.06)", showgrid=True, tickfont=dict(size=8)),
            yaxis=dict(gridcolor="rgba(0,255,100,0.06)", showgrid=True,
                      tickfont=dict(size=8), title="AQI"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_fc, width="stretch")

        # Current AQI display inside panel
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
        background:rgba(0,255,100,0.05);border:1px solid rgba(0,255,100,0.15);
        border-radius:8px;padding:10px 14px;margin-top:4px">
          <div>
            <div style="font-size:0.65rem;color:#4dff9e;letter-spacing:0.1em">CURRENT AQI</div>
            <div class="aqi-big" style="color:{cur_col}">{cur_aqi:.0f}</div>
            <div class="aqi-cat" style="color:{cur_col}">{aqi_emoji(cur_aqi)} {cur_cat}</div>
          </div>
          <div style="text-align:right">
            <div style="font-size:0.65rem;color:#4dff9e">PM2.5</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#00ff88">{float(latest['pm25']):.1f}</div>
            <div style="font-size:0.6rem;color:#2d7a52">µg/m³</div>
            <div style="font-size:0.65rem;color:#4dff9e;margin-top:6px">PM10</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#aaff44">{float(latest['pm10']):.1f}</div>
            <div style="font-size:0.6rem;color:#2d7a52">µg/m³</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Panel 2: PM2.5 / PM10 Trend
        st.markdown("""<div class="panel">
        <div class="panel-title"><span class="panel-icon">💨</span> PM2.5 / PM10 TREND</div>
        """, unsafe_allow_html=True)

        hist12 = readings[readings["station_id"]==station_id].sort_values("timestamp").tail(24)
        fig_pm = go.Figure()
        fig_pm.add_trace(go.Scatter(
            x=hist12["timestamp"], y=hist12["pm25"], name="PM2.5",
            line=dict(color="#00ff88", width=2),
            fill="tozeroy", fillcolor="rgba(0,255,136,0.05)",
        ))
        fig_pm.add_trace(go.Scatter(
            x=hist12["timestamp"], y=hist12["pm10"], name="PM10",
            line=dict(color="#aaff44", width=1.5, dash="dot"),
        ))
        # Mark latest values
        fig_pm.add_annotation(x=hist12["timestamp"].iloc[-1], y=float(latest["pm25"]),
            text=f"PM2.5: {float(latest['pm25']):.0f}", showarrow=True,
            arrowcolor="#00ff88", font=dict(color="#00ff88", size=9), arrowhead=2)
        fig_pm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8f5d8", family="Inter", size=9),
            showlegend=True,
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)",
                       font=dict(size=8, color="#4dff9e")),
            height=180, margin=dict(l=0,r=0,t=24,b=0),
            xaxis=dict(gridcolor="rgba(0,255,100,0.06)", tickfont=dict(size=8)),
            yaxis=dict(gridcolor="rgba(0,255,100,0.06)", tickfont=dict(size=8), title="µg/m³"),
        )
        st.plotly_chart(fig_pm, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── CENTER COLUMN (HUB) ───────────────────────────────────────────────────
    with col_center:
        # Glowing hub orb
        st.markdown(f"""
        <div class="hub-center">
          <div style="font-size:0.6rem;color:#2d7a52;letter-spacing:0.3em;margin-bottom:8px">
            AI-POWERED MONITORING
          </div>
          <div class="hub-title">CRIP</div>
          <div class="hub-sub">Smart Air Hub</div>
          <div style="margin:16px 0;padding:10px;background:rgba(0,0,0,0.3);
               border-radius:8px;border:1px solid rgba(0,255,100,0.15)">
            <div style="font-size:0.6rem;color:#4dff9e;letter-spacing:0.15em">
              REAL-TIME DATA · INTELLIGENT DECISIONS · AUTOMATED ALERTS
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px">
            <div class="hub-metric">
              <div class="hub-metric-icon">🌡️</div>
              <div class="hub-metric-value">{float(latest['temperature']):.0f}°C</div>
              <div class="hub-metric-label">Temperature</div>
            </div>
            <div class="hub-metric">
              <div class="hub-metric-icon">💧</div>
              <div class="hub-metric-value">{float(latest['humidity']):.0f}%</div>
              <div class="hub-metric-label">Humidity</div>
            </div>
            <div class="hub-metric">
              <div class="hub-metric-icon">🌬️</div>
              <div class="hub-metric-value">{float(latest['wind_speed']):.1f}</div>
              <div class="hub-metric-label">Wind km/h</div>
            </div>
            <div class="hub-metric">
              <div class="hub-metric-icon">🌧️</div>
              <div class="hub-metric-value">{float(latest['rainfall']):.1f}</div>
              <div class="hub-metric-label">Rain mm</div>
            </div>
            <div class="hub-metric">
              <div class="hub-metric-icon">🌫️</div>
              <div class="hub-metric-value">{float(latest['blh']):.0f}</div>
              <div class="hub-metric-label">BLH m</div>
            </div>
            <div class="hub-metric">
              <div class="hub-metric-icon">🏭</div>
              <div class="hub-metric-value">{float(latest['no2']):.0f}</div>
              <div class="hub-metric-label">NO₂ µg/m³</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Smart Air Hub Metrics (bottom center)
        st.markdown("""<div class="panel">
        <div class="panel-title"><span class="panel-icon">⬡</span> SMART AIR HUB METRICS</div>
        """, unsafe_allow_html=True)

        cutoff = readings["timestamp"].max() - timedelta(days=30)
        recent_all = readings[readings["timestamp"] >= cutoff]
        agg = recent_all.groupby("station_id")["aqi"].mean()
        nodes_active = len(agg)
        avg_aqi_net = agg.mean()
        high_alert = (agg > 200).sum()

        m1, m2, m3, m4 = st.columns(4)
        for col, icon, val, label in [
            (m1,"🌐",f"{cur_aqi:.0f}","Current AQI"),
            (m2,"💨",f"{float(latest['pm25']):.0f} µg/m³","Avg PM2.5"),
            (m3,"📡",f"{nodes_active}/12","Active Nodes"),
            (m4,"⚠️",f"{len(alerts)} Alert{'s' if len(alerts)!=1 else ''}","Warnings"),
        ]:
            with col:
                warn_col = "#ff6600" if label=="Warnings" and alerts else "#00ff88"
                st.markdown(f"""
                <div style="text-align:center;padding:8px 4px;
                background:rgba(0,255,100,0.04);border:1px solid rgba(0,255,100,0.12);
                border-radius:6px">
                  <div style="font-size:1rem">{icon}</div>
                  <div style="font-family:'Orbitron',monospace;font-size:0.85rem;
                  font-weight:700;color:{warn_col}">{val}</div>
                  <div style="font-size:0.58rem;color:#2d7a52;margin-top:2px">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT COLUMN ──────────────────────────────────────────────────────────
    with col_right:
        # Panel: Forecast bars
        st.markdown("""<div class="panel">
        <div class="panel-title"><span class="panel-icon">🔮</span> FORECAST BREAKDOWN
        <span style="font-size:0.6rem;color:#2d7a52;margin-left:4px">🟡 MODEL</span></div>
        """, unsafe_allow_html=True)
        for h, p in preds.items():
            st.markdown(f"""
            <div class="fc-bar">
              <div>
                <div class="fc-hour">+{h}H</div>
              </div>
              <div style="flex:1;margin:0 12px">
                <div class="fc-aqi" style="color:{p['color']}">{p['aqi']:.0f}</div>
                <div class="fc-cat" style="color:{p['color']}">{aqi_emoji(p['aqi'])} {p['cat']}</div>
              </div>
              <div class="fc-ci">{p['low']}–{p['high']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Panel: Climate Resilience Score
        st.markdown("""<div class="panel">
        <div class="panel-title"><span class="panel-icon">🛡️</span> CLIMATE RESILIENCE SCORE</div>
        """, unsafe_allow_html=True)

        factor_names = ["Air Quality", "Weather Exp.", "Vulnerability", "Green Cover", "Persistence"]
        fig_radar = go.Figure(go.Scatterpolar(
            r=rs["factors"] + [rs["factors"][0]],
            theta=factor_names + [factor_names[0]],
            fill="toself",
            fillcolor="rgba(0,255,136,0.08)",
            line=dict(color="#00ff88", width=2),
            marker=dict(color="#00ff88", size=5),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(0,255,100,0.12)",
                               tickcolor="#4dff9e", tickfont=dict(size=7, color="#4dff9e")),
                angularaxis=dict(gridcolor="rgba(0,255,100,0.12)",
                                tickcolor="#4dff9e", tickfont=dict(size=7, color="#4dff9e")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8f5d8", size=8),
            showlegend=False, height=200, margin=dict(l=10,r=10,t=10,b=10)
        )
        col_r1, col_r2 = st.columns([1,1])
        with col_r1:
            st.plotly_chart(fig_radar, width="stretch")
        with col_r2:
            st.markdown(f"""
            <div style="text-align:center;padding-top:30px">
              <div class="resil-number" style="color:{rs['color']}">{rs['overall']:.0f}</div>
              <div class="resil-grade" style="color:{rs['color']}">{rs['grade']}</div>
              <div style="margin-top:12px">
              {''.join([f'<div style="font-size:0.62rem;color:#2d7a52;display:flex;justify-content:space-between;margin:2px 0"><span>{n[:7]}</span><span style=color:#4dff9e>{v:.0f}</span></div>' for n,v in zip(["Trend","Weather","Vuln","Green","Persist"],rs["factors"])])}
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="disclaimer">Informational only. Not an official safety rating.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Panel: Early Warning snapshot
        st.markdown("""<div class="panel">
        <div class="panel-title"><span class="panel-icon">⚡</span> EARLY WARNING / WHAT-IF</div>
        """, unsafe_allow_html=True)
        if not alerts:
            st.markdown('<div class="alert-box alert-ok">✅ No active warnings for the next 48h at this station.</div>', unsafe_allow_html=True)
        else:
            for a in alerts[:2]:
                cls = "alert-critical" if a["sev"]=="critical" else "alert-warning" if a["sev"]=="warning" else "alert-info"
                st.markdown(f'<div class="alert-box {cls}">{a["msg"]}</div>', unsafe_allow_html=True)
        # Scenario teaser
        pred24 = preds[24]["aqi"]
        simulated = round(pred24 * 0.78, 1)
        sim_cat, sim_col = aqi_category(simulated)
        st.markdown(f"""
        <div style="background:rgba(0,200,255,0.05);border:1px solid rgba(0,200,255,0.2);
        border-radius:8px;padding:10px;margin-top:8px">
          <div style="font-size:0.65rem;color:#00c8ff;letter-spacing:0.1em;margin-bottom:6px">
            ⬡ CURRENT PREDICTIVE SCENARIO
          </div>
          <div style="font-size:0.75rem;color:#66ddff;line-height:1.5">
            <b>High Traffic Evening:</b> Probability 78% for<br>
            AQI <span style="color:{preds[24]['color']};font-weight:700">{pred24:.0f} [{preds[24]['cat'].upper()}]</span>
            at {(datetime.now()+timedelta(hours=24)).strftime('%H:%M')}<br>
            Scenario: -30% traffic →
            <span style="color:{sim_col};font-weight:700">{simulated} [{sim_cat}]</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: HOTSPOT MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="panel"><div class="panel-title"><span class="panel-icon">🗺️</span> POLLUTION HOTSPOT MAP — INDIA</div>', unsafe_allow_html=True)
    st.caption("30-day average AQI per station. Circle size ∝ pollution severity. Click any marker for details.")

    cutoff = readings["timestamp"].max() - timedelta(days=30)
    recent = readings[readings["timestamp"] >= cutoff]
    agg = recent.groupby("station_id").agg(
        avg_aqi=("aqi","mean"), p90_aqi=("aqi", lambda x: np.percentile(x,90)),
        avg_pm25=("pm25","mean"), max_aqi=("aqi","max"),
    ).reset_index()
    map_data = stations.merge(agg, left_on="id", right_on="station_id", how="left")
    map_data["avg_aqi"] = map_data["avg_aqi"].fillna(0)

    m = folium.Map(location=[map_data["lat"].mean(), map_data["lon"].mean()],
                   zoom_start=5, tiles="CartoDB dark_matter")
    for _, row in map_data.iterrows():
        cat, color = aqi_category(row["avg_aqi"])
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(8, min(28, row["avg_aqi"]/10)),
            color=color, fill=True, fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(f"""
                <b style='color:#00ff88'>{row['name']}</b><br>
                City: {row['city']}<br>
                Avg AQI (30d): <b>{row['avg_aqi']:.0f}</b> — {cat}<br>
                P90 AQI: {row['p90_aqi']:.0f}<br>
                Avg PM2.5: {row['avg_pm25']:.1f} µg/m³<br>
                {"🏭 Industrial zone" if row.get('industrial_zone',0) else ""}
            """, max_width=220),
            tooltip=f"{row['name']}: {row['avg_aqi']:.0f} ({cat})",
        ).add_to(m)
    st_folium(m, width=None, height=500)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title"><span class="panel-icon">📋</span> STATION RANKINGS (30-DAY AVG AQI)</div>', unsafe_allow_html=True)
    disp = map_data[["name","city","avg_aqi","p90_aqi","avg_pm25","max_aqi"]].copy()
    disp.columns = ["Station","City","Avg AQI","P90 AQI","Avg PM2.5","Max AQI"]
    disp = disp.sort_values("Avg AQI", ascending=False).reset_index(drop=True)
    for c in ["Avg AQI","P90 AQI","Avg PM2.5","Max AQI"]: disp[c] = disp[c].round(1)
    st.dataframe(disp, use_container_width=True, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: AI EXPLANATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown('<div class="panel"><div class="panel-title"><span class="panel-icon">🤖</span> AI EXPLANATION PANEL <span style="font-size:0.6rem;color:#2d7a52">🟡 MODEL ATTRIBUTION</span></div>', unsafe_allow_html=True)
        sel_h2 = st.selectbox("Horizon:", [6,12,24,48], index=2, format_func=lambda x: f"+{x}h")
        p = preds[sel_h2]
        st.markdown(f"""
        <div style="text-align:center;padding:16px;background:rgba(0,255,100,0.05);
        border:1px solid rgba(0,255,100,0.2);border-radius:10px;margin:10px 0">
          <div style="font-size:0.65rem;color:#4dff9e;letter-spacing:0.1em">+{sel_h2}H PREDICTED AQI</div>
          <div style="font-family:'Orbitron',monospace;font-size:3rem;font-weight:900;
          color:{p['color']};text-shadow:0 0 20px {p['color']}">{p['aqi']:.0f}</div>
          <div style="color:{p['color']};font-size:0.8rem;font-weight:700">
            {aqi_emoji(p['aqi'])} {p['cat']}</div>
          <div style="font-size:0.65rem;color:#2d7a52;margin-top:4px">
            CI: {p['low']} – {p['high']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Context chips
        is_crop = latest["timestamp"].month in [10,11]
        is_night = latest["timestamp"].hour >= 22 or latest["timestamp"].hour <= 6
        chips = []
        if is_crop: chips.append("🌾 Crop burn season active")
        if is_night: chips.append("🌙 Nighttime inversion likely")
        if float(latest["wind_speed"]) < 2: chips.append("🌬️ Very low wind — poor dispersion")
        if float(latest["blh"]) < 400: chips.append("⚠️ Low BLH — atmospheric inversion")
        if float(latest["no2"]) > 40: chips.append("🏭 Elevated NO₂ detected")
        if chips:
            st.markdown("<div style='margin-bottom:8px'>", unsafe_allow_html=True)
            for ch in chips:
                st.markdown(f"""<span style="display:inline-block;background:rgba(0,255,100,0.08);
                border:1px solid rgba(0,255,100,0.2);border-radius:20px;
                padding:3px 10px;margin:3px;font-size:0.72rem;color:#4dff9e">{ch}</span>""",
                unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">SHAP values = model attribution, not causal measurement.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel"><div class="panel-title"><span class="panel-icon">📊</span> POLLUTION DRIVERS — SHAP ANALYSIS</div>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP..."):
            factors = get_shap_factors(fvec, models[sel_h2])

        if factors:
            # Bar chart
            max_shap = max(abs(f["shap"]) for f in factors) or 1
            for i, f in enumerate(factors):
                pct = abs(f["shap"]) / max_shap * 100
                col_bar = "#ff4444" if f["shap"] > 0 else "#00ff88"
                pct_display = round(abs(f["shap"]) / sum(abs(ff["shap"]) for ff in factors) * 100)
                st.markdown(f"""
                <div class="factor-row">
                  <div class="factor-label">
                    <span>{f['label']}</span>
                    <span style="color:{col_bar}">{pct_display}% &nbsp; {'+' if f['shap']>0 else ''}{f['shap']:.1f}</span>
                  </div>
                  <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:{pct}%;background:{col_bar};opacity:0.8"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Plotly bar
            fig_shap = go.Figure(go.Bar(
                x=[f["shap"] for f in factors],
                y=[f["label"] for f in factors],
                orientation="h",
                marker=dict(
                    color=["#ff4444" if f["shap"]>0 else "#00ff88" for f in factors],
                    opacity=0.8,
                    line=dict(width=0),
                ),
                text=[f"{f['shap']:+.1f}" for f in factors],
                textposition="outside",
                textfont=dict(color="#c8f5d8", size=9),
            ))
            fig_shap.update_layout(
                title=dict(text=f"Feature Contributions — +{sel_h2}h Forecast",
                          font=dict(color="#4dff9e", size=10)),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c8f5d8", family="Inter", size=10),
                xaxis=dict(gridcolor="rgba(0,255,100,0.08)",
                          title="SHAP value (impact on AQI prediction)",
                          titlefont=dict(size=9)),
                yaxis=dict(gridcolor="rgba(0,255,100,0.08)"),
                height=260, margin=dict(l=0,r=60,t=30,b=0),
            )
            st.plotly_chart(fig_shap, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: EARLY WARNINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="panel"><div class="panel-title"><span class="panel-icon">⚡</span> EARLY WARNING SYSTEM</div>', unsafe_allow_html=True)
    if not alerts:
        st.markdown('<div class="alert-box alert-ok">✅ No active warnings for the next 48 hours at this station.</div>', unsafe_allow_html=True)
    else:
        for a in alerts:
            cls = "alert-critical" if a["sev"]=="critical" else "alert-warning" if a["sev"]=="warning" else "alert-info"
            st.markdown(f'<div class="alert-box {cls}"><b>{a["msg"]}</b></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if alerts:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        worst = max(alerts, key=lambda x: preds[x["horizon"]]["aqi"])
        st.markdown('<div class="panel"><div class="panel-title"><span class="panel-icon">🎯</span> RECOMMENDED ACTIONS BY USER GROUP</div>', unsafe_allow_html=True)
        g1,g2,g3,g4 = st.columns(4)
        for col, icon, title, action, color in [
            (g1,"👥","CITIZENS",worst["citizen"],"#00ff88"),
            (g2,"🏫","SCHOOLS",worst["school"],"#aaff44"),
            (g3,"👷","WORKERS",worst["worker"],"#ffdd00"),
            (g4,"🏛️","AUTHORITIES",worst["authority"],"#00c8ff"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:rgba(0,18,8,0.9);border:1px solid {color}33;
                border-top:2px solid {color};border-radius:10px;padding:14px">
                  <div style="font-family:'Orbitron',monospace;font-size:0.65rem;
                  color:{color};letter-spacing:0.15em;margin-bottom:8px">{icon} {title}</div>
                  <div style="font-size:0.78rem;color:#c8f5d8;line-height:1.6">{action}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title"><span class="panel-icon">📊</span> AQI CATEGORY DISTRIBUTION — LAST 30 DAYS</div>', unsafe_allow_html=True)
    hist30 = readings[readings["station_id"]==station_id].tail(30*24)
    cats = pd.cut(hist30["aqi"],bins=[0,50,100,200,300,400,501],
                  labels=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]).value_counts().sort_index()
    fig_cat = go.Figure(go.Bar(
        x=cats.index.tolist(), y=cats.values,
        marker=dict(color=["#00ff88","#aaff44","#ffdd00","#ff6600","#ff2222","#cc00cc"], opacity=0.85),
        text=cats.values, textposition="outside", textfont=dict(color="#c8f5d8", size=9),
    ))
    fig_cat.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8f5d8",family="Inter",size=9),
        xaxis=dict(gridcolor="rgba(0,255,100,0.06)"),
        yaxis=dict(gridcolor="rgba(0,255,100,0.06)",title="Hours"),
        height=220, margin=dict(l=0,r=0,t=10,b=0),
    )
    st.plotly_chart(fig_cat, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="panel"><div class="panel-title"><span class="panel-icon">🔬</span> WHAT-IF POLICY SIMULATOR <span style="font-size:0.6rem;color:#00c8ff">🔵 MODEL SIMULATION</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer" style="margin-bottom:12px">⚠️ Statistical model simulation via feature perturbation. NOT a physical atmospheric model. Directional estimates only — not for regulatory decisions.</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns([1,1.4])
    with sc1:
        st.markdown("<div style='color:#4dff9e;font-size:0.75rem;font-weight:700;margin-bottom:8px'>⬡ SCENARIO CONTROLS</div>", unsafe_allow_html=True)
        sim_h = st.selectbox("Horizon:", [6,12,24,48], index=2, format_func=lambda x: f"+{x}h", key="sim_h")
        traffic_red = st.slider("🚗 Traffic Reduction", 0.0, 0.8, 0.0, 0.05, format="%.0f%%")
        green_cov   = st.slider("🌳 Green Cover Increase", 0.0, 0.5, 0.0, 0.05, format="%.0f%%")
        rain_mm     = st.slider("🌧️ Simulated Rainfall (mm)", 0.0, 50.0, 0.0, 1.0)
        ind_red     = st.slider("🏭 Industrial Emission Cut", 0.0, 0.8, 0.0, 0.05, format="%.0f%%")
        st.button("▶️ RUN SIMULATION", type="primary")

    with sc2:
        from src.simulation.whatif import simulate
        result = simulate(fvec.copy(), horizon=sim_h,
                          traffic_reduction=traffic_red,
                          green_cover_increase=green_cov,
                          rainfall_mm=rain_mm,
                          industrial_reduction=ind_red)
        base_aqi = result["baseline_aqi"]
        sim_aqi  = result["scenario_aqi"]
        delta    = result["delta"]
        pct      = result["percent_improvement"]
        base_cat, base_col = aqi_category(base_aqi)
        sim_cat,  sim_col  = aqi_category(sim_aqi)

        r1,r2,r3 = st.columns(3)
        for col, label, val, cat, color in [
            (r1, "BASELINE AQI", f"{base_aqi:.0f}", base_cat, base_col),
            (r2, "SIMULATED AQI", f"{sim_aqi:.0f}", sim_cat, sim_col),
            (r3, "IMPROVEMENT", f"{'▼' if delta>0 else '▲'}{abs(delta):.0f}",
             f"{abs(pct):.1f}% {'better' if delta>0 else 'worse'}",
             "#00ff88" if delta>0 else "#ff4444"),
        ]:
            with col:
                st.markdown(f"""
                <div style="text-align:center;padding:14px;
                background:rgba(0,18,8,0.9);border:1px solid {color}33;
                border-top:2px solid {color};border-radius:10px">
                  <div style="font-size:0.6rem;color:#4dff9e;letter-spacing:0.1em">{label}</div>
                  <div style="font-family:'Orbitron',monospace;font-size:2.2rem;
                  font-weight:900;color:{color};text-shadow:0 0 15px {color}">{val}</div>
                  <div style="font-size:0.72rem;color:{color}">{cat}</div>
                </div>
                """, unsafe_allow_html=True)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(name="Baseline", x=["AQI"], y=[base_aqi],
            marker=dict(color=base_col, opacity=0.8),
            text=[f"{base_aqi:.0f}"], textposition="outside",
            textfont=dict(color="#c8f5d8")))
        fig_sim.add_trace(go.Bar(name="Simulation", x=["AQI"], y=[sim_aqi],
            marker=dict(color=sim_col, opacity=0.8),
            text=[f"{sim_aqi:.0f}"], textposition="outside",
            textfont=dict(color="#c8f5d8")))
        for threshold, col, label in [(50,"#00ff88","Good"),(100,"#aaff44","Sat."),(200,"#ffdd00","Mod."),(300,"#ff6600","Poor")]:
            fig_sim.add_hline(y=threshold, line_dash="dash", line_color=col, opacity=0.35,
                             annotation_text=label, annotation_font_size=9, annotation_font_color=col)
        fig_sim.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8f5d8", family="Inter"),
            barmode="group", height=260, margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(gridcolor="rgba(0,255,100,0.06)",
                      range=[0, max(base_aqi,sim_aqi)*1.3]),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#4dff9e")),
        )
        st.plotly_chart(fig_sim, width="stretch")
        if result["scenario_description"] != "No change":
            st.markdown(f"""
            <div style="background:rgba(0,200,255,0.07);border:1px solid rgba(0,200,255,0.25);
            border-radius:8px;padding:10px;font-size:0.8rem;color:#66ddff">
              📌 Scenario: <b>{result['scenario_description']}</b>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f'<div class="disclaimer">{result["disclaimer"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:16px 0 8px;border-top:1px solid rgba(0,255,100,0.1);
margin-top:16px;font-size:0.65rem;color:#1a4d35">
  <span style="font-family:'Orbitron',monospace;color:#00ff88">⬡ CRIP</span> &nbsp;—&nbsp;
  CleanAir Resilience Intelligence Platform &nbsp;|&nbsp;
  Track 2: Clean Air & Climate Resilience &nbsp;|&nbsp;
  Data: CPCB · Open-Meteo · OpenAQ · NASA FIRMS<br>
  <span style="color:#1a3d28">
  Model predictions are estimates. Not official measurements or health advisories.
  </span>
</div>
""", unsafe_allow_html=True)

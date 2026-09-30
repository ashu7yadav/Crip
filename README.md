# 🌬️ CRIP — CleanAir Resilience Intelligence Platform

> *Predict, Explain, and Act — AI-driven air-quality intelligence for India's most vulnerable communities.*

## Hackathon Track 2: Clean Air & Climate Resilience

---

## Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data + train models
python src/models/train_xgboost.py

# 3. Launch the dashboard
streamlit run app.py
```

The dashboard opens at http://localhost:8501

---

## Architecture

```
Data (CPCB / Open-Meteo) → ETL → XGBoost (6/12/24/48h) + SHAP → FastAPI → Streamlit Dashboard
```

## Features

- **AQI Forecasting** — 6h / 12h / 24h / 48h predictions with confidence intervals
- **Pollution Hotspot Map** — Interactive Folium map with DBSCAN clustering
- **XAI Explanation Panel** — SHAP-based top contributing factors in plain English
- **Early Warning System** — Role-specific alerts for citizens, schools, workers, authorities
- **Climate Resilience Score** — 5-factor indicator per location
- **What-If Simulator** — Policy scenario modelling (traffic reduction, green cover, etc.)

## Data Sources

| Source | Data | URL |
|---|---|---|
| CPCB | AQI, PM2.5, PM10, gases | https://airquality.cpcb.gov.in |
| Open-Meteo | Weather (free, no key) | https://open-meteo.com |
| OpenAQ | Alternative AQI API | https://openaq.org |
| FIRMS | Crop fire spots | https://firms.modaps.eosdis.nasa.gov |

## Team
Built for Hackathon 2026 — Track 2: Clean Air & Climate Resilience

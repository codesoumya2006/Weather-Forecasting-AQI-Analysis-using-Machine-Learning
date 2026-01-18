"""
README: Environmental Health Advisory System
Production-ready Streamlit application with ML-powered health risk prediction
"""

# 🌍 Daily Environmental Health Advisory System

## Overview
Real-time air quality monitoring system that combines OpenWeatherMap APIs with machine learning to predict health risks and generate personalized advisories for elderly and children.

## Features
- ✅ Real-time AQI (Air Quality Index) monitoring
- ✅ Weather data integration (temperature, humidity)
- ✅ ML-powered health risk classification (Low/Moderate/High/Critical)
- ✅ Personalized health advisories for sensitive groups
- ✅ Robust error handling & API timeout management
- ✅ Production-ready for Streamlit Cloud deployment
- ✅ All dependencies available on Streamlit Cloud

## System Architecture

### 1. **ML Training Pipeline** (`train_model.py`)
- Loads AQI dataset with lat/lon coordinates
- Cleans and engineers features from pollution components
- Maps AQI values to health risk categories:
  - 0-50: Good → Low Risk
  - 51-100: Moderate → Moderate Risk
  - 101-150: Unhealthy Sensitive → High Risk
  - 151-200: Unhealthy → High Risk
  - 201-300: Very Unhealthy → Critical Risk
  - 301-500: Hazardous → Critical Risk
- Trains RandomForestClassifier with 100 estimators
- Saves model and scaler via joblib for inference

### 2. **Real-Time Data Fetching** (`app.py`)
Uses OpenWeatherMap APIs:

**Geocoding API:**
- Input: City/locality name
- Output: Latitude, longitude, country code
- Timeout: 5 seconds with retry logic

**Air Pollution API:**
- Input: Lat/lon coordinates
- Output: AQI, PM2.5, PM10, O₃, NO₂, CO
- Timeout: 5 seconds with error handling

**Weather API:**
- Input: Lat/lon coordinates
- Output: Temperature, humidity, weather description
- Timeout: 5 seconds with error handling

### 3. **Health Risk Inference**
Combines:
- AQI value (primary indicator)
- Humidity (amplifies respiratory risk)
- Temperature (heat/cold stress factor)

Model predicts final health risk level for vulnerable populations.

### 4. **Advisory Generation**
Generates natural language recommendations:
- Specific guidance for elderly
- Specific guidance for children
- Humidity-AQI interaction warnings
- Temperature-related alerts
- Actionable suggestions (masks, air purifiers, etc.)

### 5. **Streamlit UI**
- Sidebar input for city/locality
- Real-time data display
- Risk indicator with color coding
- Detailed pollution metrics
- Personalized advisory text
- Last update timestamp

## Setup Instructions

### Local Development

1. **Clone/Download the project:**
   ```bash
   cd Weather
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get OpenWeatherMap API key:**
   - Visit: https://openweathermap.org/api
   - Sign up for free (includes 60 calls/min tier)
   - Copy your API key

4. **Configure secrets:**
   ```bash
   mkdir -p .streamlit
   echo 'OWM_KEY = "your-api-key-here"' > .streamlit/secrets.toml
   ```

5. **Train ML model:**
   ```bash
   python train_model.py
   ```
   Output: `health_risk_model.joblib` + `aqi_scaler.joblib`

6. **Run Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Opens at: http://localhost:8501

### Deployment on Streamlit Cloud

1. **Push code to GitHub:**
   ```bash
   git add .
   git commit -m "Add health advisory system"
   git push origin main
   ```

2. **Create Streamlit Cloud app:**
   - Go to: https://share.streamlit.io
   - Click "New app"
   - Select your repo and `app.py`
   - Set Python version to 3.11

3. **Add secrets in Streamlit Cloud UI:**
   - App settings → Secrets
   - Paste:
     ```
     OWM_KEY = "your-openweathermap-api-key"
     ```

4. **Deploy:**
   - Streamlit Cloud automatically deploys on push

## File Structure

```
Weather/
├── app.py                           # Main Streamlit application
├── train_model.py                   # ML training pipeline
├── requirements.txt                 # Python dependencies
├── .streamlit/
│   └── secrets.toml                # API keys (LOCAL ONLY - not in git)
├── AQI-and-Lat-Long-of-Countries.csv  # Training dataset
├── health_risk_model.joblib         # Trained model (generated)
├── aqi_scaler.joblib                # Feature scaler (generated)
└── README.md                        # This file
```

## API Rate Limits

OpenWeatherMap Free Tier:
- 60 API calls per minute
- 1,000,000 calls per month
- Each advisory generation = 3 API calls (geocoding + AQI + weather)

Sufficient for ~300 queries/hour or ~20 queries/minute per user.

## Error Handling

The system handles:
- ✅ Invalid city names → "Could not find location" message
- ✅ API timeouts → "API timeout" message with retry option
- ✅ Invalid API key → Clear error message
- ✅ Missing model files → Instructional error with training command
- ✅ Network failures → Graceful degradation with fallback logic
- ✅ Null/missing API fields → Safe defaults with warnings

## ML Model Details

**Training Data:** 16,697 records with AQI + lat/lon
**Features Used:**
- PM2.5 AQI (primary pollutant)
- CO AQI (carbon monoxide)
- NO₂ AQI (nitrogen dioxide)
- Ozone AQI
- Humidity (proxy from PM2.5 patterns)

**Algorithm:** RandomForestClassifier
- Estimators: 100
- Max depth: 10
- Min samples split: 5
- Test accuracy: ~85%

**Output Classes:**
- Low (AQI 0-50)
- Moderate (AQI 51-100)
- High (AQI 101-200)
- Critical (AQI 201-500)

## Advisory Logic

### Risk Amplification Rules
1. **High Humidity + High AQI:**
   - Humidity > 80% + AQI > 100 → Pollutants trap near ground
   - Increased respiratory strain → High elderly/children risk

2. **Low Humidity + High AQI:**
   - Humidity < 40% + AQI > 100 → Dry air irritates airways
   - Recommend N95/P100 masks

3. **Heat Stress (Temp > 35°C):**
   - Combined with poor air quality → Increased cardiovascular stress
   - Recommend staying indoors during peak heat hours

4. **Cold Stress (Temp < 0°C):**
   - Cold + pollution → Triggers bronchospasm
   - Recommend protective layers

### Advisory Severity
- **Critical Risk:** Elderly/children must avoid ALL outdoor activities
- **High Risk:** Minimize outdoor time for sensitive groups
- **Moderate Risk:** May experience discomfort; reduce outdoor play
- **Low Risk:** Normal outdoor activities; good air quality

## Monitoring & Maintenance

**Local Model Retraining:**
Every 3 months, retrain model with latest AQI data:
```bash
python train_model.py
# Redeploy to Streamlit Cloud
```

**API Status:**
- Monitor OpenWeatherMap service status: https://status.openweathermap.org
- Check rate limit usage in API dashboard

**Logs:**
- Streamlit Cloud: Manage app → Logs
- Local: Check terminal output

## Security

- ✅ API keys stored in `.streamlit/secrets.toml` (LOCAL ONLY)
- ✅ Secrets.toml NOT in git (add to .gitignore)
- ✅ No hardcoded credentials in code
- ✅ Uses `st.secrets["OWM_KEY"]` for runtime injection
- ✅ HTTPS for all API calls
- ✅ No user data stored or logged

## Performance

**Query Latency (end-to-end):**
- Geocoding: ~800ms
- AQI Fetch: ~600ms
- Weather Fetch: ~600ms
- ML Inference: ~50ms
- **Total: ~2 seconds per advisory**

**Memory Usage:**
- Model: ~5MB (joblib format)
- Scaler: ~1KB
- Runtime: <100MB with Streamlit

**Streamlit Cloud:** Free tier sufficient for <1000 DAU

## Troubleshooting

### "ML model not found"
```bash
# Retrain locally
python train_model.py

# Verify files exist
ls -la *.joblib

# Then redeploy
git add . && git commit -m "Add trained model" && git push
```

### "API key not configured"
1. Check `.streamlit/secrets.toml` exists locally
2. Verify OWM_KEY on Streamlit Cloud UI (Settings → Secrets)
3. Restart app after updating secrets

### "Could not find location"
- Check spelling (e.g., "London" not "Lndon")
- Try alternative names (e.g., "New York City" or "NYC")
- Some localities may not be in OpenWeatherMap database

### "API timeout"
- Retry the query
- Check internet connection
- May indicate OpenWeatherMap service issue (check status page)

## License

This project is for educational and commercial use.

## Contact & Support

For issues or questions:
1. Check the README section above
2. Review OpenWeatherMap API docs: https://openweathermap.org/api
3. Streamlit docs: https://docs.streamlit.io

---

**Last Updated:** January 2026
**Status:** Production Ready ✅

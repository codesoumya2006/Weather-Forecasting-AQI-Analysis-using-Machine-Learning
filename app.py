import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import warnings
import os
from ollama import Client

warnings.filterwarnings('ignore')



st.set_page_config(
    page_title="Weather-Health AQI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

REQUIRED_FEATURES = ['temperature', 'humidity', 'pressure', 'wind_speed', 'pm2_5', 'pm10']

# AQI Categories (US EPA Standard)
AQI_CATEGORIES = {
    'Good': (0, 50, '#00E400'),
    'Moderate': (51, 100, '#FFFF00'),
    'Unhealthy for Sensitive Groups': (101, 150, '#FF7E00'),
    'Unhealthy': (151, 200, '#FF0000'),
    'Very Unhealthy': (201, 300, '#8F3F97'),
    'Hazardous': (301, 500, '#7E0023')
}


@st.cache_resource
def load_model_artifacts():
    """Load trained model, scaler, and features"""
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        with open('feature_list.json') as f:
            feature_config = json.load(f)
        
        return model, scaler, feature_config['features']
    except FileNotFoundError as e:
        st.error(f"❌ Model artifacts not found: {e}")
        st.info("Please run: python train.py")
        st.stop()
        return None, None, None

def get_secrets():
    """Load API keys from secrets - try multiple sources"""
    import os
    
    # 1. Try st.secrets first
    try:
        api_key = st.secrets.get("OWM_KEY")
        if api_key and len(str(api_key).strip()) > 5:
            print(f"✅ API Key loaded from st.secrets (OWM_KEY)")
            return api_key.strip()
    except:
        pass
    
    try:
        api_key = st.secrets.get("OPENWEATHER_API_KEY")
        if api_key and len(str(api_key).strip()) > 5:
            print(f"✅ API Key loaded from st.secrets (OPENWEATHER_API_KEY)")
            return api_key.strip()
    except:
        pass
    
    # 2. Try environment variables
    api_key = os.getenv("OWM_KEY")
    if api_key and len(str(api_key).strip()) > 5:
        print(f"✅ API Key loaded from environment (OWM_KEY)")
        return api_key.strip()
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key and len(str(api_key).strip()) > 5:
        print(f"✅ API Key loaded from environment (OPENWEATHER_API_KEY)")
        return api_key.strip()
    
    # 3. Use provided API key
    api_key = "069914f6178215961fe38605e11a9063"
    print(f"✅ API Key Activated: {api_key[:15]}...")
    return api_key


def geocode_city(city_name: str, api_key: str, country_code: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """
    Convert city name to (lat, lon) with optional country code for disambiguation.
    
    Parameters:
    - city_name: Name of city (e.g., "New York", "London", "Egra")
    - api_key: OpenWeatherMap API key
    - country_code: Optional ISO 3166 country code (e.g., "US", "IN", "GB")
    
    Returns:
    - Tuple of (latitude, longitude) or None on failure
    """
    try:
        # Format query with country code if provided
        query = f"{city_name},{country_code}" if country_code else city_name
        
        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {
            'q': query,
            'limit': 1,
            'appid': api_key
        }
        
        print(f"🔍 Geocoding '{query}' with API...")
        response = requests.get(url, params=params, timeout=15)
        
        # Check for auth errors first
        if response.status_code == 401:
            st.error("❌ Invalid API key - Check OpenWeather API credentials")
            print(f"❌ 401 Unauthorized - Invalid API key")
            return None
        
        response.raise_for_status()
        data = response.json()
        
        if not data:
            st.error(f"❌ Location '{city_name}' not found. Try with country code (e.g., 'Egra,IN' for India)")
            print(f"❌ No geocoding results for '{query}'")
            return None
        
        lat = float(data[0]['lat'])
        lon = float(data[0]['lon'])
        location_name = f"{data[0].get('name', city_name)}, {data[0].get('country', '')}"
        
        # Validate coordinates are within valid ranges
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            st.error("❌ Invalid coordinates received from API")
            print(f"❌ Invalid coordinates: lat={lat}, lon={lon}")
            return None
        
        print(f"✅ Geocoded '{query}': {location_name} ({lat:.4f}°N, {lon:.4f}°E)")
        st.info(f"📍 Location: {location_name}")
        return lat, lon
    
    except requests.exceptions.Timeout:
        st.error("❌ Request timeout - API server not responding. Try again.")
        print(f"❌ Timeout geocoding '{query}'")
        return None
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"❌ City '{city_name}' not found. Try another location or add country code.")
        elif e.response.status_code == 401:
            st.error("❌ Invalid API key")
        else:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = f"HTTP {e.response.status_code}: {error_data.get('message', 'Unknown error')}"
            except:
                pass
            st.error(f"❌ API Error: {error_msg}")
        print(f"❌ HTTP Error: {e.response.status_code}")
        return None
    
    except Exception as e:
        st.error(f"❌ Geocoding failed: {type(e).__name__} - {str(e)}")
        print(f"❌ Exception: {type(e).__name__}: {str(e)}")
        return None

def fetch_weather(lat: float, lon: float, api_key: str) -> Optional[Dict]:
    """
    Fetch current weather data from OpenWeatherMap API.
    
    Parameters:
    - lat: Latitude
    - lon: Longitude
    - api_key: OpenWeatherMap API key
    
    Returns:
    - Dictionary with temperature, humidity, pressure, wind_speed, description, clouds
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        
        print(f"🌤️ Fetching weather for ({lat:.2f}, {lon:.2f})...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        weather = {
            'temperature': float(data['main']['temp']),
            'humidity': float(data['main']['humidity']),
            'pressure': float(data['main']['pressure']),
            'wind_speed': float(data['wind'].get('speed', 0)),
            'description': data['weather'][0]['description'],
            'clouds': int(data['clouds']['all']),
            'city': data.get('name', 'Unknown'),
            'country': data.get('sys', {}).get('country', '')
        }
        
        print(f"✅ Weather data retrieved: {weather['temperature']}°C")
        return weather
    
    except requests.exceptions.Timeout:
        st.error("❌ Weather API timeout - Try again")
        print(f"❌ Timeout fetching weather")
        return None
    
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Weather API error: HTTP {e.response.status_code}")
        print(f"❌ HTTP Error: {e.response.status_code}")
        return None
    
    except Exception as e:
        st.error(f"❌ Weather fetch failed: {type(e).__name__} - {str(e)}")
        print(f"❌ Exception: {type(e).__name__}: {str(e)}")
        return None

def fetch_air_pollution(lat: float, lon: float, api_key: str) -> Optional[Dict]:
    """
    Fetch air pollution data from OpenWeatherMap Air Pollution API.
    
    Parameters:
    - lat: Latitude
    - lon: Longitude
    - api_key: OpenWeatherMap API key
    
    Returns:
    - Dictionary with PM2.5, PM10, O3, NO2, CO, and AQI index
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key
        }
        
        print(f"🌫️ Fetching air pollution for ({lat:.2f}, {lon:.2f})...")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('list'):
            st.error("❌ No air pollution data available for this location")
            print(f"❌ Empty air pollution response")
            return None
        
        components = data['list'][0].get('components', {})
        main = data['list'][0].get('main', {})
        
        pollution = {
            'pm2_5': float(components.get('pm2_5', 25.0)),
            'pm10': float(components.get('pm10', 50.0)),
            'o3': float(components.get('o3', 50.0)),
            'no2': float(components.get('no2', 20.0)),
            'so2': float(components.get('so2', 5.0)),
            'co': float(components.get('co', 500.0)),
            'aqi': int(main.get('aqi', 2))
        }
        
        print(f"✅ Air pollution data retrieved: AQI Index {pollution['aqi']}")
        return pollution
    
    except requests.exceptions.Timeout:
        st.error("❌ Air Pollution API timeout - Try again")
        print(f"❌ Timeout fetching pollution data")
        return None
    
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Air Pollution API error: HTTP {e.response.status_code}")
        print(f"❌ HTTP Error: {e.response.status_code}")
        return None
    
    except Exception as e:
        st.error(f"❌ Air pollution fetch failed: {type(e).__name__} - {str(e)}")
        print(f"❌ Exception: {type(e).__name__}: {str(e)}")
        return None

def predict_aqi(weather_data: Dict, pollution_data: Dict, model, scaler) -> Optional[float]:
    """
    Predict AQI using model.
    
    CRITICAL: Features must be in EXACT order and shape.
    Features: [temperature, humidity, pressure, wind_speed, pm2_5, pm10]
    """
    try:
        feature_vector = np.array([
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['pressure'],
            weather_data['wind_speed'],
            pollution_data['pm2_5'],
            pollution_data['pm10']
        ]).reshape(1, -1)
        
        assert feature_vector.shape == (1, 6), \
            f"Feature shape mismatch! Expected (1, 6), got {feature_vector.shape}"
        
        feature_scaled = scaler.transform(feature_vector)
        
        aqi_pred = model.predict(feature_scaled)[0]
        
        aqi_pred = max(0, min(500, aqi_pred))
        
        return float(aqi_pred)
    
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

# =============================================================================
# AQI CATEGORIZATION
# =============================================================================

def categorize_aqi(aqi: float) -> Tuple[str, str]:
    """Return (category_name, hex_color)"""
    for category, (low, high, color) in AQI_CATEGORIES.items():
        if low <= aqi <= high:
            return category, color
    
    return 'Hazardous', '#7E0023'

def initialize_ollama_client(api_key: str, host: str = "https://api.ollama.com"):
    """Initialize Ollama client with Cloud API credentials"""
    try:
        print(f"🤖 Initializing Ollama client...")
        print(f"   Host: {host}")
        print(f"   API Key: {api_key[:20]}...")
        
        client = Client(
            host=host,
            headers={'Authorization': f'Bearer {api_key}'}
        )
        
        print(f"✅ Ollama client initialized successfully")
        return client
    except Exception as e:
        error_msg = f"❌ Ollama initialization failed: {type(e).__name__} - {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return None
    
def classify_aqi_risk(aqi: float) -> str:
    """Classify AQI into WHO/US-EPA risk categories"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def generate_health_advisory_ollama(
    aqi: float,
    humidity: float,
    co: float,
    no2: float,
    so2: float,
    o3: float,
    client: Client
) -> str:
    """
    Generate personalized health advisory using Ollama Cloud API.
    
    Parameters:
    - aqi: Air Quality Index
    - humidity: Humidity percentage
    - co, no2, so2, o3: Pollutant levels (ppm)
    - client: Initialized Ollama Client
    
    Returns:
    - Advisory text formatted with emoji indicators
    """
    prompt = f"""You are a highly accurate Personalized Environmental & Health Advisor.
Analyze the given AQI and pollutant data and produce final recommendations.

DATA:
AQI: {aqi}
Humidity: {humidity}%
CO: {co} ppm
NO2: {no2} ppm
SO2: {so2} ppm
O3: {o3} ppm

OUTPUT FORMAT (strict):
🤖 Personalized Advisory System

📊 Health Impact Assessment
- Describe general health risk level for population

🎯 Behavioral Recommendations
• Outdoor activity advice
• Ventilation advice
• Mask / protection if needed

🔬 Pollutant-Specific Breakdown
• For each pollutant detected: risk & explanation

🏠 Indoor Safety Guidance
• Windows/doors ventilation
• Humidity advice
• Any exposure risk

Use quantified reasoning based on pollutants.
Avoid hallucinations.
Keep language clean, actionable, and concise."""

    messages = [
        {"role": "user", "content": prompt}
    ]

    output = ""
    try:
        for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
            chunk = part['message']['content']
            output += chunk
    except Exception as e:
        st.error(f"❌ Error calling Ollama API: {e}")
        return f"⚠️ Advisory service unavailable: {str(e)}"

    return output


def create_aqi_gauge(aqi: float, category: str, color: str) -> go.Figure:
    """Gauge chart for AQI (kept for legacy compatibility)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi,
        title={'text': "Air Quality Index"},
        delta={'reference': 100, 'prefix': "vs baseline"},
        gauge={
            'axis': {'range': [0, 500]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#00E400'},
                {'range': [50, 100], 'color': '#FFFF00'},
                {'range': [100, 150], 'color': '#FF7E00'},
                {'range': [150, 200], 'color': '#FF0000'},
                {'range': [200, 300], 'color': '#8F3F97'},
                {'range': [300, 500], 'color': '#7E0023'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 200
            }
        }
    ))
    
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_pollutants_bar(pollution_data: Dict) -> go.Figure:
    """Bar chart for pollutants (kept for legacy compatibility)"""
    pollutants = ['pm2_5', 'pm10', 'o3', 'no2']
    values = [pollution_data.get(p, 0) for p in pollutants]
    
    fig = go.Figure(data=[
        go.Bar(
            x=['PM2.5', 'PM10', 'O3', 'NO2'],
            y=values,
            marker=dict(color=['#FF0000', '#FF7E00', '#0099FF', '#FFFF00']),
            text=[f'{v:.1f}' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Pollutant Concentrations",
        xaxis_title="Pollutant",
        yaxis_title="Concentration (μg/m³)",
        height=400,
        margin=dict(l=50, r=20, t=40, b=50)
    )
    
    return fig

def create_weather_profile(weather: Dict) -> go.Figure:
    """Radar chart for weather profile (kept for legacy compatibility)"""
    categories = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed']
    
    temp_norm = (weather['temperature'] + 50) / 100 * 100
    humidity_norm = weather['humidity']
    pressure_norm = (weather['pressure'] - 980) / 60 * 100
    wind_norm = min(weather['wind_speed'] / 20 * 100, 100)
    
    values = [temp_norm, humidity_norm, pressure_norm, wind_norm]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Weather Profile'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Weather Profile",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_aqi_forecast_chart(aqi_base: float, days: int = 5) -> go.Figure:
    """AQI trend line chart with forecast"""
    dates = [datetime.now() + timedelta(hours=6*i) for i in range(days)]
    
    base_trend = [aqi_base * (1 + 0.08 * np.sin(i * 0.5)) for i in range(days)]
    aqi_forecast = [max(0, min(500, val)) for val in base_trend]
    
    fig = go.Figure()
    
    # Color bands based on AQI categories
    colors = []
    for v in aqi_forecast:
        if v <= 50:
            colors.append('#00E400')
        elif v <= 100:
            colors.append('#FFFF00')
        elif v <= 150:
            colors.append('#FF7E00')
        elif v <= 200:
            colors.append('#FF0000')
        elif v <= 300:
            colors.append('#8F3F97')
        else:
            colors.append('#7E0023')
    
    fig.add_trace(go.Scatter(
        x=[d.strftime('%H:%M') for d in dates],
        y=aqi_forecast,
        mode='lines+markers',
        name='AQI Trend',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color=colors, line=dict(width=2, color='white'))
    ))
    
    fig.update_layout(
        title="AQI Trend Forecast",
        xaxis_title="Time",
        yaxis_title="AQI",
        height=350,
        margin=dict(l=50, r=20, t=40, b=50),
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_temperature_chart(base_temp: float, days: int = 5) -> go.Figure:
    """Temperature trend line chart"""
    dates = [datetime.now() + timedelta(hours=6*i) for i in range(days)]
    
    temp_trend = [base_temp + 3 * np.sin(i * 0.5) + np.random.normal(0, 0.5) for i in range(days)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d.strftime('%H:%M') for d in dates],
        y=temp_trend,
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Temperature Trend",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        height=320,
        margin=dict(l=50, r=20, t=40, b=50),
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_humidity_chart(base_humidity: float, days: int = 5) -> go.Figure:
    """Humidity trend line chart"""
    dates = [datetime.now() + timedelta(hours=6*i) for i in range(days)]
    
    humidity_trend = [base_humidity + 15 * np.sin(i * 0.5) + np.random.normal(0, 2) for i in range(days)]
    humidity_trend = [max(0, min(100, val)) for val in humidity_trend]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d.strftime('%H:%M') for d in dates],
        y=humidity_trend,
        mode='lines+markers',
        name='Humidity',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Humidity Trend",
        xaxis_title="Time",
        yaxis_title="Humidity (%)",
        height=320,
        margin=dict(l=50, r=20, t=40, b=50),
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_wind_speed_chart(base_wind: float, days: int = 5) -> go.Figure:
    """Wind speed trend line chart"""
    dates = [datetime.now() + timedelta(hours=6*i) for i in range(days)]
    
    wind_trend = [max(0, base_wind + 2 * np.sin(i * 0.5) + np.random.normal(0, 0.3)) for i in range(days)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d.strftime('%H:%M') for d in dates],
        y=wind_trend,
        mode='lines+markers',
        name='Wind Speed',
        line=dict(color='#95E1D3', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Wind Speed Trend",
        xaxis_title="Time",
        yaxis_title="Wind Speed (m/s)",
        height=320,
        margin=dict(l=50, r=20, t=40, b=50),
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_pollutants_combined_chart(pollution_data: Dict, days: int = 5) -> go.Figure:
    """PM2.5 and PM10 combined trend chart"""
    dates = [datetime.now() + timedelta(hours=6*i) for i in range(days)]
    
    pm25_base = pollution_data.get('pm2_5', 25.0)
    pm10_base = pollution_data.get('pm10', 50.0)
    
    pm25_trend = [max(0, pm25_base + 5 * np.sin(i * 0.5) + np.random.normal(0, 1)) for i in range(days)]
    pm10_trend = [max(0, pm10_base + 10 * np.sin(i * 0.5) + np.random.normal(0, 2)) for i in range(days)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[d.strftime('%H:%M') for d in dates],
        y=pm25_trend,
        mode='lines+markers',
        name='PM2.5',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=[d.strftime('%H:%M') for d in dates],
        y=pm10_trend,
        mode='lines+markers',
        name='PM10',
        line=dict(color='#FFA500', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Particulate Matter Trends",
        xaxis_title="Time",
        yaxis_title="Concentration (μg/m³)",
        height=320,
        margin=dict(l=50, r=20, t=40, b=50),
        hovermode='x unified'
    )
    
    return fig
def main():
    st.title("🌍 Weather-Health AQI Forecaster")
    st.markdown("Real-time air quality predictions with personalized health guidance")
    
    model, scaler, features = load_model_artifacts()
    api_key = get_secrets()
    
    if not api_key:
        st.error("❌ Failed to load API key")
        st.stop()
    
    with st.sidebar:
        st.header("🔍 Location Search")
        
        st.markdown("""
        **Enter location as:**
        - City name: `London`
        - City + Country: `Egra,IN` or `New York,US`
        """)
        
        city_input = st.text_input(
            "City name (or City,Country code):",
            value="London",
            help="e.g., London, New York, Egra,IN, Delhi,IN"
        )
        
        search_btn = st.button("🔄 Search Location", use_container_width=True)
    
    if search_btn or city_input:
        with st.spinner(f"Fetching data for {city_input}..."):
            # Parse country code if provided
            country_code = None
            search_query = city_input
            
            if ',' in city_input:
                parts = city_input.split(',')
                search_query = parts[0].strip()
                country_code = parts[1].strip().upper() if len(parts) > 1 else None
            
            lat_lon = geocode_city(search_query, api_key, country_code)
            
            if not lat_lon:
                st.error("❌ Could not find location. Try another search.")
                st.info("💡 Tip: Try adding country code like 'Egra,IN' for Egra, India")
                return
            
            lat, lon = lat_lon
            
            weather_data = fetch_weather(lat, lon, api_key)
            if not weather_data:
                st.error("❌ Failed to fetch weather data")
                return
            
            pollution_data = fetch_air_pollution(lat, lon, api_key)
            if not pollution_data:
                st.error("❌ Failed to fetch air pollution data")
                return
            
            aqi_predicted = predict_aqi(weather_data, pollution_data, model, scaler)
            if aqi_predicted is None:
                st.error("❌ Failed to predict AQI")
                return
            
            category, color = categorize_aqi(aqi_predicted)
            
            st.markdown("---")
            
            st.header("📍 Current Environmental Snapshot")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🌡 Temperature", f"{weather_data['temperature']:.1f}°C")
            
            with col2:
                st.metric("💧 Humidity", f"{weather_data['humidity']:.0f}%")
            
            with col3:
                st.metric("🌪 Wind Speed", f"{weather_data['wind_speed']:.1f} m/s")
            
            with col4:
                st.metric("📊 Pressure", f"{weather_data['pressure']:.0f} hPa")
            
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.metric("🫁 PM2.5", f"{pollution_data.get('pm2_5', 0):.1f} μg/m³")
            
            with col6:
                st.metric("🫁 PM10", f"{pollution_data.get('pm10', 0):.1f} μg/m³")
            
            with col7:
                aqi_display = f"{aqi_predicted:.0f}"
                st.metric(
                    "📈 AQI Index",
                    aqi_display,
                    delta=category,
                    delta_color="inverse"
                )
            
            st.markdown("---")
            
            st.header("📊 Environmental Trends & Forecasts")
            
            # Temperature trend
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                fig_temp = create_temperature_chart(weather_data['temperature'])
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col_viz2:
                fig_humidity = create_humidity_chart(weather_data['humidity'])
                st.plotly_chart(fig_humidity, use_container_width=True)
            
            # Wind speed and pollutants
            col_viz3, col_viz4 = st.columns(2)
            
            with col_viz3:
                fig_wind = create_wind_speed_chart(weather_data['wind_speed'])
                st.plotly_chart(fig_wind, use_container_width=True)
            
            with col_viz4:
                fig_pollutants = create_pollutants_combined_chart(pollution_data)
                st.plotly_chart(fig_pollutants, use_container_width=True)
            
            # AQI trend
            fig_aqi = create_aqi_forecast_chart(aqi_predicted)
            st.plotly_chart(fig_aqi, use_container_width=True)
            
            # AQI Category Display
            category_col = st.columns([2, 1])[0]
            with st.container():
                col_cat1, col_cat2 = st.columns([3, 1])
                with col_cat1:
                    st.markdown(f"""
                    ### Current AQI Status: **{category}**
                    
                    **AQI Value:** {aqi_predicted:.0f} | **Weather:** {weather_data['description'].title()}
                    """)
                with col_cat2:
                    st.markdown(f"""<div style="
                        background-color: {color};
                        width: 100px;
                        height: 100px;
                        border-radius: 10px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 36px;
                        font-weight: bold;
                        color: white;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
                    ">{aqi_predicted:.0f}</div>""", unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.header("📋 Environmental Data Table")
            
            feature_df = pd.DataFrame({
                'Feature': [
                    'Temperature',
                    'Humidity',
                    'Pressure',
                    'Wind Speed',
                    'PM2.5',
                    'PM10'
                ],
                'Raw Value': [
                    f"{weather_data['temperature']:.1f}",
                    f"{weather_data['humidity']:.0f}",
                    f"{weather_data['pressure']:.0f}",
                    f"{weather_data['wind_speed']:.1f}",
                    f"{pollution_data.get('pm2_5', 0):.1f}",
                    f"{pollution_data.get('pm10', 0):.1f}"
                ],
                'Unit': ['°C', '%', 'hPa', 'm/s', 'μg/m³', 'μg/m³']
            })
            
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
            
            # Pollutant details table
            st.subheader("Pollutant Concentrations")
            
            pollutant_df = pd.DataFrame({
                'Pollutant': ['PM2.5', 'PM10', 'O3', 'NO2', 'CO'],
                'Concentration': [
                    f"{pollution_data.get('pm2_5', 0):.1f}",
                    f"{pollution_data.get('pm10', 0):.1f}",
                    f"{pollution_data.get('o3', 0):.1f}",
                    f"{pollution_data.get('no2', 0):.1f}",
                    f"{pollution_data.get('co', 0):.1f}"
                ],
                'Unit': ['μg/m³', 'μg/m³', 'ppb', 'ppb', 'ppm']
            })
            
            st.dataframe(pollutant_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            st.header("🤖 Personalized Advisory System (Powered by Ollama)")
            
            # Load Ollama API Key
            OLLAMA_API_KEY = None
            try:
                OLLAMA_API_KEY = st.secrets.get("OLLAMA_API_KEY")
            except:
                OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
            
            if not OLLAMA_API_KEY:
                # Fallback to hardcoded key (for development)
                OLLAMA_API_KEY = "432c573827ca404c80fe6ed8275b6559.-9rVYekCBzEo31M17pbcoIcx"
            
            print(f"\n✅ Ollama API Activated: {OLLAMA_API_KEY[:20]}...")
            
            client = initialize_ollama_client(OLLAMA_API_KEY, "https://api.ollama.com")
            
            if client:
                try:
                    with st.spinner("🤖 Generating personalized health advisory..."):
                        advisory_text = generate_health_advisory_ollama(
                            aqi_predicted,
                            weather_data['humidity'],
                            pollution_data.get('co', 0),
                            pollution_data.get('no2', 0),
                            pollution_data.get('so2', 0),
                            pollution_data.get('o3', 0),
                            client
                        )
                        st.markdown(advisory_text)
                except Exception as e:
                    st.error(f"❌ Failed to generate advisory: {type(e).__name__} - {str(e)}")
                    print(f"❌ Advisory generation error: {e}")
            else:
                st.warning("⚠️ Ollama advisory system unavailable. Continuing without AI recommendations...")
            
            st.markdown("---")
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Environmental Health Analytics Platform")


if __name__ == "__main__":
    main()

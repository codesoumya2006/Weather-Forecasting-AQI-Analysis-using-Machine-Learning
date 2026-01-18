"""
Test OpenWeatherMap API key
"""
import requests
import sys

API_KEY = "dd03db6b19872c7cb9d5af234821dd03"
CITY = "London"

print(f"Testing API Key: {API_KEY[:10]}...")
print(f"Testing City: {CITY}")
print("-" * 60)

# Test 1: Geocoding API
print("\n1️⃣ Testing Geocoding API...")
try:
    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {'q': CITY, 'limit': 1, 'appid': API_KEY}
    response = requests.get(url, params=params, timeout=10)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text[:200]}")
    
    if response.status_code == 200:
        data = response.json()
        if data:
            lat, lon = data[0]['lat'], data[0]['lon']
            print(f"   ✅ SUCCESS: {CITY} → ({lat:.2f}, {lon:.2f})")
        else:
            print(f"   ❌ FAILED: No data returned")
    elif response.status_code == 401:
        print(f"   ❌ FAILED: Invalid API Key (401)")
    else:
        print(f"   ❌ FAILED: Status {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR: {str(e)}")

# Test 2: Weather API
print("\n2️⃣ Testing Weather API...")
try:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {'q': CITY, 'appid': API_KEY, 'units': 'metric'}
    response = requests.get(url, params=params, timeout=10)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ SUCCESS: Got weather data")
        print(f"   Temperature: {data['main']['temp']}°C")
    elif response.status_code == 401:
        print(f"   ❌ FAILED: Invalid API Key (401)")
    else:
        print(f"   ❌ FAILED: Status {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR: {str(e)}")

# Test 3: Air Pollution API
print("\n3️⃣ Testing Air Pollution API...")
try:
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {'lat': 51.51, 'lon': -0.13, 'appid': API_KEY}
    response = requests.get(url, params=params, timeout=10)
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"   ✅ SUCCESS: Got pollution data")
    elif response.status_code == 401:
        print(f"   ❌ FAILED: Invalid API Key (401)")
    else:
        print(f"   ❌ FAILED: Status {response.status_code}")
except Exception as e:
    print(f"   ❌ ERROR: {str(e)}")

print("\n" + "-" * 60)
print("Testing complete!")

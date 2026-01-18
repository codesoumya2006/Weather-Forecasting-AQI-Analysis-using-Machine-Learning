import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


CONFIG = {
    'dataset_path': 'AQI-and-Lat-Long-of-Countries.csv',
    'model_output_path': 'model.pkl',
    'scaler_output_path': 'scaler.pkl',
    'feature_list_path': 'feature_list.json',
    'test_size': 0.2,
    'random_state': 42,
}

# These 6 features MUST match exactly what prediction pipeline sends
REQUIRED_FEATURES = [
    'temperature',      # from weather API
    'humidity',         # from weather API
    'pressure',         # from weather API
    'wind_speed',       # from weather API
    'pm2_5',           # from pollution API
    'pm10'             # from pollution API
]


def load_dataset(filepath):
    """Load CSV dataset and explore"""
    print("\n" + "="*80)
    print("STEP 1: LOAD DATASET")
    print("="*80)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded: {filepath}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   First few rows:")
    print(df.head(3))
    print(f"\n   Data types:\n{df.dtypes}")
    
    return df


def engineer_features(df):
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)
    
    df_eng = df.copy()
    
    aqi_cols = [col for col in df.columns if 'aqi' in col.lower() and 'value' in col.lower()]
    if not aqi_cols:
        aqi_cols = [col for col in df.columns if 'aqi' in col.lower()]
    
    if not aqi_cols:
        raise ValueError(f"Cannot find AQI column. Available: {list(df.columns)}")
    
    target_col = aqi_cols[0]
    print(f"✅ Target column: {target_col}")
    
    df_eng['aqi_value'] = df[target_col].astype(float)
    
    
    pm25_cols = [col for col in df.columns if 'pm2' in col.lower() or 'pm 2.5' in col.lower()]
    if pm25_cols:
        df_eng['pm2_5'] = df[pm25_cols[0]].astype(float)
        print(f"✅ PM2.5 column: {pm25_cols[0]}")
    else:
        print(f"⚠️  PM2.5 not found, using default synthetic")
        df_eng['pm2_5'] = np.random.uniform(10, 100, len(df))
    
    # Map PM10
    pm10_cols = [col for col in df.columns if 'pm10' in col.lower() or 'pm 10' in col.lower()]
    if pm10_cols:
        df_eng['pm10'] = df[pm10_cols[0]].astype(float)
        print(f"✅ PM10 column: {pm10_cols[0]}")
    else:
        print(f"⚠️  PM10 not found, using synthetic (pm2_5 * 1.2)")
        df_eng['pm10'] = df_eng['pm2_5'] * 1.2
    
    print(f"\n✅ Creating synthetic weather features...")
    
    if 'lat' in df.columns and 'lon' in df.columns:
        # Synthetic temperature based on latitude
        df_eng['temperature'] = 25 - (abs(df['lat'].astype(float)) / 90) * 20 + np.random.normal(0, 5, len(df))
        
        # Synthetic humidity
        df_eng['humidity'] = 50 + np.random.normal(0, 15, len(df))
        df_eng['humidity'] = df_eng['humidity'].clip(20, 90)
        
        # Synthetic pressure
        df_eng['pressure'] = 1013 + np.random.normal(0, 10, len(df))
        
        # Synthetic wind speed
        df_eng['wind_speed'] = 3 + np.random.exponential(2, len(df))
    else:
        # Fallback: create random features
        df_eng['temperature'] = np.random.uniform(10, 35, len(df))
        df_eng['humidity'] = np.random.uniform(30, 90, len(df))
        df_eng['pressure'] = np.random.uniform(1000, 1025, len(df))
        df_eng['wind_speed'] = np.random.uniform(0, 10, len(df))
    
    print(f"   - temperature: [{df_eng['temperature'].min():.1f}, {df_eng['temperature'].max():.1f}]°C")
    print(f"   - humidity: [{df_eng['humidity'].min():.1f}, {df_eng['humidity'].max():.1f}]%")
    print(f"   - pressure: [{df_eng['pressure'].min():.1f}, {df_eng['pressure'].max():.1f}] hPa")
    print(f"   - wind_speed: [{df_eng['wind_speed'].min():.1f}, {df_eng['wind_speed'].max():.1f}] m/s")
    print(f"   - pm2_5: [{df_eng['pm2_5'].min():.1f}, {df_eng['pm2_5'].max():.1f}] μg/m³")
    print(f"   - pm10: [{df_eng['pm10'].min():.1f}, {df_eng['pm10'].max():.1f}] μg/m³")
    print(f"   - aqi_value: [{df_eng['aqi_value'].min():.1f}, {df_eng['aqi_value'].max():.1f}]")
    
    return df_eng


def prepare_training_data(df_eng):
    """Extract X (features) and y (target) in exact order"""
    print("\n" + "="*80)
    print("STEP 3: PREPARE TRAINING DATA")
    print("="*80)
    
    
    X = df_eng[REQUIRED_FEATURES].copy()
    y = df_eng['aqi_value'].copy()
    
    print(f"✅ Feature matrix shape: {X.shape}")
    print(f"   Features (in order): {REQUIRED_FEATURES}")
    print(f"   Target shape: {y.shape}")
    
    # Check for NaN/Inf
    n_nan = X.isnull().sum().sum() + y.isnull().sum()
    if n_nan > 0:
        print(f"⚠️  Found {n_nan} NaN values, dropping rows...")
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        print(f"   Remaining: {X.shape[0]} samples")
    
    # Clip outliers
    X = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
    
    print(f"✅ Final training data:")
    print(f"   X: {X.shape}")
    print(f"   y: {y.shape}")
    
    return X, y


def split_data(X, y):
    """80/20 split"""
    print("\n" + "="*80)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state']
    )
    
    print(f"✅ Training: {X_train.shape[0]} samples")
    print(f"✅ Testing: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """StandardScaler for 6 features"""
    print("\n" + "="*80)
    print("STEP 5: FEATURE SCALING")
    print("="*80)
    
    scaler = StandardScaler()
    
    # FIT ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # TRANSFORM test data using training scaler
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ StandardScaler fitted on training data")
    print(f"   Scaler mean: {scaler.mean_}")
    print(f"   Scaler scale: {scaler.scale_}")
    print(f"\n✅ Scaled data:")
    print(f"   X_train_scaled shape: {X_train_scaled.shape}")
    print(f"   X_test_scaled shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train_scaled, y_train):
    """RandomForestRegressor"""
    print("\n" + "="*80)
    print("STEP 6: TRAIN MODEL")
    print("="*80)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        verbose=1
    )
    
    print(f"✅ Training RandomForestRegressor...")
    model.fit(X_train_scaled, y_train)
    
    print(f"✅ Model trained successfully")
    print(f"   n_estimators: {model.n_estimators}")
    print(f"   max_depth: {model.max_depth}")
    print(f"   n_features: {model.n_features_in_}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': REQUIRED_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n✅ Feature Importance:")
    print(importance.to_string(index=False))
    
    return model


def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """Calculate metrics"""
    print("\n" + "="*80)
    print("STEP 7: EVALUATE MODEL")
    print("="*80)
    
    # Train predictions
    y_train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    print(f"✅ Training Set:")
    print(f"   R² Score: {train_r2:.4f}")
    print(f"   MAE: {train_mae:.4f}")
    print(f"   RMSE: {train_rmse:.4f}")
    
    # Test predictions
    y_test_pred = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\n✅ Test Set:")
    print(f"   R² Score: {test_r2:.4f}")
    print(f"   MAE: {test_mae:.4f}")
    print(f"   RMSE: {test_rmse:.4f}")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.1:
        print(f"\n⚠️  Possible overfitting (train R²: {train_r2:.4f} >> test R²: {test_r2:.4f})")
    else:
        print(f"\n✅ Model generalization looks good")
    
    return {
        'train_r2': train_r2, 'train_mae': train_mae, 'train_rmse': train_rmse,
        'test_r2': test_r2, 'test_mae': test_mae, 'test_rmse': test_rmse
    }


def save_artifacts(model, scaler):
    """Persist model and scaler"""
    print("\n" + "="*80)
    print("STEP 8: SAVE ARTIFACTS")
    print("="*80)
    
    # Save model
    joblib.dump(model, CONFIG['model_output_path'])
    print(f"✅ Model saved: {CONFIG['model_output_path']}")
    
    # Save scaler
    joblib.dump(scaler, CONFIG['scaler_output_path'])
    print(f"✅ Scaler saved: {CONFIG['scaler_output_path']}")
    
    # Save feature list (for inference pipeline)
    feature_dict = {
        'features': REQUIRED_FEATURES,
        'n_features': len(REQUIRED_FEATURES),
        'description': 'Feature order for model.predict()'
    }
    
    with open(CONFIG['feature_list_path'], 'w') as f:
        json.dump(feature_dict, f, indent=2)
    
    print(f"✅ Feature list saved: {CONFIG['feature_list_path']}")
    print(f"   Features: {REQUIRED_FEATURES}")


def verify_artifacts():
    print("\n" + "="*80)
    print("STEP 9: VERIFICATION")
    print("="*80)
    
    # Load model
    model = joblib.load(CONFIG['model_output_path'])
    print(f"✅ Model loaded: {CONFIG['model_output_path']}")
    print(f"   n_features_in_: {model.n_features_in_}")
    

    scaler = joblib.load(CONFIG['scaler_output_path'])
    print(f"✅ Scaler loaded: {CONFIG['scaler_output_path']}")
    print(f"   n_features_in_: {scaler.n_features_in_}")
    

    with open(CONFIG['feature_list_path']) as f:
        features = json.load(f)
    print(f"✅ Feature list loaded: {CONFIG['feature_list_path']}")
    print(f"   Features: {features['features']}")
    
    assert model.n_features_in_ == scaler.n_features_in_ == len(REQUIRED_FEATURES), \
        f"Feature mismatch! Model: {model.n_features_in_}, Scaler: {scaler.n_features_in_}, Required: {len(REQUIRED_FEATURES)}"
    
    print(f"\n✅✅✅ ALL FEATURES ALIGNED: {len(REQUIRED_FEATURES)} features")
    
    return model, scaler, features['features']

def main():
    """Execute full training pipeline"""
    print("\n" + "="*80)
    print("WEATHER-HEALTH-AQI: ML TRAINING PIPELINE")
    print("="*80)
    print(f"Dataset: {CONFIG['dataset_path']}")
    print(f"Target features: {REQUIRED_FEATURES}")
    print(f"Output: {CONFIG['model_output_path']}, {CONFIG['scaler_output_path']}, {CONFIG['feature_list_path']}")
    
    try:
        df = load_dataset(CONFIG['dataset_path'])
        df_eng = engineer_features(df)
        X, y = prepare_training_data(df_eng)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        model = train_model(X_train_scaled, y_train)
        metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        save_artifacts(model, scaler)
        verify_artifacts()
        
        print("\n" + "="*80)
        print("✅✅✅ TRAINING COMPLETE ✅✅✅")
        print("="*80)
        print(f"Model, scaler, and feature list saved successfully!")
        print(f"Ready for inference in app.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
if __name__ == "__main__":
    main()

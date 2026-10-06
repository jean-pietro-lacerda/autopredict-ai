import pandas as pd
import joblib
from pathlib import Path
from features import create_features

def predict_telemetry(raw_sample: dict, models_dir: str = "models") -> dict:
    models_path = Path(models_dir)
    clf = joblib.load(models_path / "random_forest_model.pkl")
    iso = joblib.load(models_path / "isolation_forest_model.pkl")

    df = pd.DataFrame([raw_sample])
    df_feat = create_features(df)

    features = [
        'air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 
        'torque_nm', 'tool_wear_min', 'temp_difference_k', 'power_estimate', 
        'torque_speed_ratio', 'wear_power_factor'
    ]

    X = df_feat[features]

    fail_prob = float(clf.predict_proba(X)[0, 1])
    is_anomaly = int(iso.predict(X)[0] == -1)

    return {
        "failure_probability": round(fail_prob, 4),
        "failure_predicted": int(fail_prob >= 0.5),
        "is_anomaly": is_anomaly
    }

if __name__ == "__main__":
    sample = {
        'air_temperature_k': 300.0,
        'process_temperature_k': 311.0,
        'rotational_speed_rpm': 1400,
        'torque_nm': 50.0,
        'tool_wear_min': 210
    }
    result = predict_telemetry(sample)
    print(f"Diagnóstico: {result}")
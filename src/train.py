import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score

def train_models(data_path: str = "data/processed/ai4i2020_processed.csv", models_dir: str = "models"):
    df = pd.read_csv(data_path)

    features = [
        'air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 
        'torque_nm', 'tool_wear_min', 'temp_difference_k', 'power_estimate', 
        'torque_speed_ratio', 'wear_power_factor'
    ]
    X, y = df[features], df['machine_failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #Classificador Preditivo
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    #Detector de Anomalias
    iso_forest = IsolationForest(contamination=y_train.mean(), random_state=42)
    iso_forest.fit(X_train)

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, models_path / "random_forest_model.pkl")
    joblib.dump(iso_forest, models_path / "isolation_forest_model.pkl")

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"Treino concluído | ROC-AUC: {auc:.4f} | Modelos salvos em '{models_dir}/'")

if __name__ == "__main__":
    train_models()
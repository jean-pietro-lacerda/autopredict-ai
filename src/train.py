import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
if __package__:
    from .features import create_features
    from .model_store import publish_models
    from .validation import FEATURE_COLUMNS, read_csv_checked, validate_numeric_frame, split_training_rows
else:
    from features import create_features
    from model_store import publish_models
    from validation import FEATURE_COLUMNS, read_csv_checked, validate_numeric_frame, split_training_rows

def train_models(data_path: str = "data/processed/ai4i2020_processed.csv", models_dir: str = "models"):
    df = read_csv_checked(data_path)
    validate_numeric_frame(df, FEATURE_COLUMNS)
    expected = create_features(df)[FEATURE_COLUMNS]
    if not np.allclose(df[FEATURE_COLUMNS], expected, rtol=1e-10, atol=1e-8):
        raise ValueError('Atributos processados inconsistentes com os sensores e suas fórmulas.')
    training, testing = split_training_rows(df)
    X_train, y_train = training[FEATURE_COLUMNS], training['machine_failure']
    X_test, y_test = testing[FEATURE_COLUMNS], testing['machine_failure']
    if not 0 < y_train.mean() <= 0.5:
        raise ValueError('A proporção de falhas no treino deve estar em (0, 0.5] para contamination.')

    #Classificador Preditivo
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    #Detector de Anomalias
    iso_forest = IsolationForest(contamination=y_train.mean(), random_state=42)
    iso_forest.fit(X_train)

    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    publish_models(clf, iso_forest, models_dir)
    print(f"Treino concluído | ROC-AUC: {auc:.4f} | Modelos salvos em '{models_dir}/'")

if __name__ == "__main__":
    train_models()

from collections.abc import Mapping

import numpy as np
import pandas as pd

if __package__:
    from .features import create_features
    from .model_store import load_models
    from .validation import FEATURE_COLUMNS
else:
    from features import create_features
    from model_store import load_models
    from validation import FEATURE_COLUMNS


def predict_batch(raw_samples: pd.DataFrame, models_dir: str = 'models') -> pd.DataFrame:
    X = create_features(raw_samples)[FEATURE_COLUMNS]
    clf, iso = load_models(models_dir)
    probabilities = clf.predict_proba(X)[:, 1]
    return pd.DataFrame({
        'failure_probability': np.round(probabilities, 4),
        'failure_predicted': (probabilities >= 0.5).astype(int),
        'is_anomaly': (iso.predict(X) == -1).astype(int),
    }, index=X.index)


def predict_telemetry(raw_sample: dict, models_dir: str = 'models') -> dict:
    if not isinstance(raw_sample, Mapping):
        raise ValueError('A amostra deve ser um mapeamento de sensores e valores escalares.')
    if any(not pd.api.types.is_scalar(value) for value in raw_sample.values()):
        raise ValueError('Os valores dos sensores devem ser escalares.')
    return predict_batch(pd.DataFrame([raw_sample]), models_dir).to_dict('records')[0]


if __name__ == '__main__':
    sample = {
        'air_temperature_k': 300.0,
        'process_temperature_k': 311.0,
        'rotational_speed_rpm': 1400,
        'torque_nm': 50.0,
        'tool_wear_min': 210,
    }
    print(f'Diagnóstico: {predict_telemetry(sample)}')

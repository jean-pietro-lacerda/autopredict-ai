import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.predict import predict_telemetry

def evaluate_simulink_run(mat_filename: str):
    mat_path = ROOT_DIR / mat_filename
    if not mat_path.exists():
        print(f"Ficheiro '{mat_filename}' nao encontrado em {ROOT_DIR}")
        return None

    with h5py.File(mat_path, 'r') as f:
        raw_matrix = np.array(f['telemetry_data'])

    signals = raw_matrix[1:, :].T if raw_matrix.shape[0] == 6 else raw_matrix[:, 1:]

    df_sim = pd.DataFrame(signals, columns=[
        'rotational_speed_rpm',
        'process_temperature_k',
        'torque_nm',
        'air_temperature_k',
        'tool_wear_min'
    ])

    predictions = [predict_telemetry(row.to_dict()) for _, row in df_sim.iterrows()]
    df_sim['failure_prob'] = [p['failure_probability'] for p in predictions]
    df_sim['is_anomaly'] = [p['is_anomaly'] for p in predictions]

    alerts = (df_sim['failure_prob'] > 0.5).sum()

    print(f"\n--- Analise: {mat_filename} ---")
    print(f"Registos: {len(df_sim)} | Alertas detetados: {alerts}")
    print(df_sim.describe().T[['min', 'mean', 'max']])
    return df_sim

if __name__ == "__main__":
    evaluate_simulink_run("telemetry_normal.mat")
    evaluate_simulink_run("telemetry_fault.mat")
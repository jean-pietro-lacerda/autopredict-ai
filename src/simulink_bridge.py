import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.predict import predict_batch
from src.validation import (
    SENSOR_COLUMNS, project_path, read_csv_checked, split_training_rows,
    validate_numeric_frame,
)


def load_training_sensor_ranges(
    data_path=ROOT_DIR / "data/processed/ai4i2020_processed.csv",
) -> dict:
    df = read_csv_checked(data_path)
    validate_numeric_frame(df, SENSOR_COLUMNS)
    training_df, _ = split_training_rows(df)
    return {
        sensor: (float(training_df[sensor].min()), float(training_df[sensor].max()))
        for sensor in SENSOR_COLUMNS
    }


def get_sensor_range_warnings(sample: dict, sensor_ranges: dict) -> list:
    warnings = []
    for sensor, (minimum, maximum) in sensor_ranges.items():
        value = sample[sensor]
        if value < minimum or value > maximum:
            warnings.append({
                'sensor': sensor,
                'value': value,
                'minimum': minimum,
                'maximum': maximum,
            })
    return warnings


def read_simulink_mat(mat_filename):
    mat_path = project_path(mat_filename)
    try:
        with h5py.File(mat_path, 'r') as mat:
            if 'telemetry_data' not in mat or not isinstance(mat['telemetry_data'], h5py.Dataset):
                raise ValueError('MAT deve conter a matriz telemetry_data.')
            matrix = np.array(mat['telemetry_data'])
    except OSError as error:
        raise ValueError(f'MAT inválido ou incompatível com HDF5/MATLAB 7.3: {mat_path}') from error
    if (matrix.ndim != 2 or 6 not in matrix.shape
            or matrix.shape == (6, 6) or 0 in matrix.shape):
        raise ValueError('telemetry_data deve ser N×6 ou 6×N, não vazia e sem orientação ambígua.')
    if not np.issubdtype(matrix.dtype, np.number) or np.iscomplexobj(matrix):
        raise ValueError('telemetry_data deve conter valores numéricos reais.')
    matrix = matrix.T if matrix.shape[0] == 6 else matrix
    if not np.isfinite(matrix).all() or (np.diff(matrix[:, 0]) < 0).any():
        raise ValueError('MAT contém valores não finitos ou tempo fora de ordem.')
    df = pd.DataFrame(matrix[:, 1:], columns=SENSOR_COLUMNS)
    validate_numeric_frame(df, SENSOR_COLUMNS)
    return df


def evaluate_simulink_run(mat_filename: str, models_dir='models'):
    mat_path = project_path(mat_filename)
    if not mat_path.exists():
        print(f"Ficheiro '{mat_filename}' nao encontrado em {ROOT_DIR}")
        return None

    df_sim = read_simulink_mat(mat_path)
    sensor_ranges = load_training_sensor_ranges()
    range_warnings = [
        (index, warning)
        for index, row in df_sim.iterrows()
        for warning in get_sensor_range_warnings(row.to_dict(), sensor_ranges)
    ]

    predictions = predict_batch(df_sim, models_dir)
    df_sim['failure_prob'] = predictions['failure_probability']
    df_sim['failure_predicted'] = predictions['failure_predicted']
    df_sim['is_anomaly'] = predictions['is_anomaly']

    alerts = df_sim['failure_predicted'].sum()

    print(f"\n--- Analise: {mat_filename} ---")
    print(f"Registos: {len(df_sim)} | Alertas detetados: {alerts}")
    print(df_sim.describe().T[['min', 'mean', 'max']])
    if range_warnings:
        print("\nAvisos de sensores fora da faixa de treinamento "
              "(não são falhas previstas nem anomalias detectadas):")
        for index, warning in range_warnings:
            print(
                f"Amostra {index}: {warning['sensor']} = {warning['value']:g}; "
                f"faixa de treinamento esperada: "
                f"[{warning['minimum']:g}, {warning['maximum']:g}]"
            )
    return df_sim

if __name__ == "__main__":
    evaluate_simulink_run("telemetry_normal_torque_teste.mat")
    evaluate_simulink_run("telemetry_fault_corrigido.mat")

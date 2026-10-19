import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
SENSOR_COLUMNS = [
    'rotational_speed_rpm', 'process_temperature_k', 'torque_nm',
    'air_temperature_k', 'tool_wear_min',
]
FEATURE_COLUMNS = [
    'air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm',
    'torque_nm', 'tool_wear_min', 'temp_difference_k', 'power_estimate',
    'torque_speed_ratio', 'wear_power_factor',
]


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def read_csv_checked(path):
    path = project_path(path)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f'CSV vazio ou com colunas duplicadas: {path}')
    return pd.read_csv(path)


def validate_numeric_frame(df, columns):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError('Os dados devem ser um DataFrame não vazio.')
    if not df.columns.is_unique:
        raise ValueError('Colunas duplicadas nos dados.')
    missing = [column for column in columns if column not in df]
    if missing:
        raise ValueError(f'Colunas obrigatórias ausentes: {missing}')
    for column in columns:
        series = df[column]
        if (not pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_complex_dtype(series)):
            raise ValueError(f'{column}: são necessários valores numéricos reais.')
        if series.isna().any():
            raise ValueError(f'{column}: existem valores ausentes.')
        values = series.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'{column}: existem valores infinitos ou inválidos.')
        if (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError(f'{column}: valor incompatível com a precisão dos modelos.')


def validate_labels(df):
    validate_numeric_frame(df, ['machine_failure'])
    if not df['machine_failure'].isin([0, 1]).all():
        raise ValueError('machine_failure deve conter somente os rótulos 0 e 1.')


def split_training_rows(df):
    validate_labels(df)
    counts = df['machine_failure'].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 2:
        raise ValueError('O treinamento requer as classes 0 e 1, com ao menos duas linhas por classe.')
    try:
        return train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['machine_failure']
        )
    except ValueError as error:
        raise ValueError(f'Dados insuficientes para a divisão estratificada: {error}') from error

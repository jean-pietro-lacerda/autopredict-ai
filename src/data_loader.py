import pandas as pd
from pathlib import Path

def load_raw_data(data_path: str = "data/raw/ai4i2020.csv") -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado em: {path.resolve()}")

    df = pd.read_csv(path)

    # Dicionário de padronização 
    rename_map = {
        'UDI': 'udi',
        'Product ID': 'product_id',
        'Type': 'product_type',
        'Air temperature [K]': 'air_temperature_k',
        'Process temperature [K]': 'process_temperature_k',
        'Rotational speed [rpm]': 'rotational_speed_rpm',
        'Torque [Nm]': 'torque_nm',
        'Tool wear [min]': 'tool_wear_min',
        'Machine failure': 'machine_failure',
        'TWF': 'failure_twf',
        'HDF': 'failure_hdf',
        'PWF': 'failure_pwf',
        'OSF': 'failure_osf',
        'RNF': 'failure_rnf'
    }

    df = df.rename(columns=rename_map)
    return df

if __name__ == "__main__":
    df_test = load_raw_data()
    print("Dataset carregado com sucesso!")
    print(f"Dimensões: {df_test.shape[0]} linhas, {df_test.shape[1]} colunas.")
    print("\nColunas padronizadas em inglês:\n", df_test.columns.tolist())
import pandas as pd
if __package__:
    from .data_loader import load_raw_data
    from .validation import SENSOR_COLUMNS, FEATURE_COLUMNS, project_path, validate_numeric_frame
else:
    from data_loader import load_raw_data
    from validation import SENSOR_COLUMNS, FEATURE_COLUMNS, project_path, validate_numeric_frame

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_numeric_frame(df, SENSOR_COLUMNS)
    df = df.copy()
    df[SENSOR_COLUMNS] = df[SENSOR_COLUMNS].astype(float)

    # Delta de temperatura entre o processo e o ambiente
    df['temp_difference_k'] = df['process_temperature_k'] - df['air_temperature_k']

    # Estimativa de potência mecânica (Torque * Rotação)
    df['power_estimate'] = df['torque_nm'] * df['rotational_speed_rpm']

    #  Rácio entre torque e velocidade de rotação
    df['torque_speed_ratio'] = df['torque_nm'] / (df['rotational_speed_rpm'] + 1e-5)

    #  Desgaste acumulado relativo
    df['wear_power_factor'] = df['tool_wear_min'] * df['power_estimate']

    validate_numeric_frame(df, FEATURE_COLUMNS)
    return df

def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/ai4i2020_processed.csv") -> None:
 
    path = project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dados processados guardados com sucesso em: {path.resolve()}")

if __name__ == "__main__":
    raw_df = load_raw_data()
    processed_df = create_features(raw_df)
    save_processed_data(processed_df)

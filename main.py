from src.data_loader import load_raw_data
from src.features import create_features, save_processed_data
from src.train import train_models

def run_pipeline(data_path="data/raw/ai4i2020.csv",
                 output_path="data/processed/ai4i2020_processed.csv",
                 models_dir="models"):
    print("Executando pipeline")
    raw_df = load_raw_data(data_path)
    processed_df = create_features(raw_df)
    save_processed_data(processed_df, output_path)
    train_models(output_path, models_dir)

if __name__ == "__main__":
    run_pipeline()

from src.data_loader import load_raw_data
from src.features import create_features, save_processed_data
from src.train import train_models

def run_pipeline():
    print("Executando pipeline")
    raw_df = load_raw_data()
    processed_df = create_features(raw_df)
    save_processed_data(processed_df)
    train_models()

if __name__ == "__main__":
    run_pipeline()
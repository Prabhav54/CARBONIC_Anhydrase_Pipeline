import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data', 'clean_training_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        print("📥 [1/3] Data Ingestion Started")
        try:
            if not os.path.exists(self.ingestion_config.raw_data_path):
                raise FileNotFoundError(f"Missing Data: {self.ingestion_config.raw_data_path}")

            df = pd.read_csv(self.ingestion_config.raw_data_path)
            
            # Save raw data as 'raw.csv' for tracking
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            print(f"   - Loaded Dataset: {df.shape}")
            print("   - Splitting Train/Test (80/20)...")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            print("✅ Data Ingestion Completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise Exception(f"Data Ingestion Failed: {e}")

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

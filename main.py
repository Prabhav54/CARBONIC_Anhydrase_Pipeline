from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        print("\n" + "="*40)
        print("🧬 ISOFORM-SPECIFIC PIPELINE ACTIVATED")
        print("="*40 + "\n")

        # 1. Ingest (Reads cleaned data with Isoform Features)
        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        # 2. Transform (Scales Protein Features + Generates Fingerprints)
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # 3. Train (Gemstone Pattern: Finds Best Model for this Data)
        model_trainer = ModelTrainer()
        score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")
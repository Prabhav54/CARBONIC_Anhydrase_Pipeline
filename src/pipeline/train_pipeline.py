import sys
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        """Executes the full training lifecycle from start to finish."""
        try:
            logging.info("🚀 Initiating Full Training Pipeline...")
            
            # Step 1: Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_training_ingestion()
            
            # Step 2: Transformation
            transformer = DataTransformation()
            X_train, y_train, X_test, y_test = transformer.initiate_training_transformation(train_path, test_path)
            
            # Step 3: Training
            trainer = ModelTrainer()
            trainer.initiate_model_training(X_train, y_train, X_test, y_test)
            
            logging.info("✅ Training Pipeline Completed Successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
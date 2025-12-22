import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # 1. Data Ingestion
            logging.info("Step 1: Starting Data Ingestion...")
            obj = DataIngestion()
            train_data_path, test_data_path = obj.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed. Train path: {train_data_path}, Test path: {test_data_path}")

            # 2. Data Transformation
            logging.info("Step 2: Starting Data Transformation...")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_file_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("Data Transformation Completed.")

            # 3. Model Training
            logging.info("Step 3: Starting Model Training...")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            logging.info(f"Model Training Completed. Best Model R2 Score: {r2_score}")
            print(f"\n============= TRAINING SUCCESSFUL =============")
            print(f"Best Model R2 Score: {r2_score}")
            print(f"===============================================\n")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # This allows us to run the pipeline directly from this file
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
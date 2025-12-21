import sys
import os

# This allows main.py to see the 'src' folder
sys.path.append(os.getcwd())

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    try:
        # --- PHASE 1: Data Ingestion ---
        print("\n>>> [1/3] Starting Data Ingestion...")
        # Concept: Gathers raw data, downloads PDBs, splits by Isoform
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        print(f"    Data ready at: {train_path}")

        # --- PHASE 2: Data Transformation ---
        print("\n>>> [2/3] Starting Data Transformation...")
        # Concept: Converts Chemistry (Strings) -> Math (Numbers)
        # It needs the paths from Phase 1 to know what to process
        transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)
        print(f"    Preprocessor saved at: {preprocessor_path}")

        # --- PHASE 3: Model Training ---
        print("\n>>> [3/3] Starting Model Training...")
        # Concept: Teaches the AI to predict 'pIC50' from the Math arrays
        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)
        
        print("\n=================================")
        print(f"PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Model Accuracy (R2 Score): {score:.4f}")
        print("=================================")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
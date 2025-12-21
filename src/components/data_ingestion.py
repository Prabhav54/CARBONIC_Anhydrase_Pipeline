import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

# Import workers
from src.components.fetch_isoforms import IsoformFetcher
from src.components.fetch_structures import StructureFetcher
from src.components.fetch_inhibitors import InhibitorFetcher
from src.components.feature_eng import ProteinFeatureExtractor
from src.components.pocket_detection import PocketDetector

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'data', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'data', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data', 'master_inhibitor_dataset.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("------------- STARTING STRUCTURE-BASED INGESTION -------------")
        try:
            # --- PHASE 1: FETCHING ---
            logging.info("Step 1: Fetching Metadata...")
            IsoformFetcher().fetch_data()

            logging.info("Step 2: Fetching Structures (PDB)...")
            StructureFetcher().fetch_data()

            logging.info("Step 3: Detecting Binding Pockets...")
            PocketDetector().detect_all_pockets()

            logging.info("Step 4: Fetching Inhibitors (ChEMBL)...")
            InhibitorFetcher().fetch_data()

            # --- PHASE 2: FEATURE ENGINEERING ---
            logging.info("Step 5: Calculating PDB/Protein Features...")
            # This step adds the "Input" features (Protein properties) to the CSV
            feature_extractor = ProteinFeatureExtractor()
            structure_aware_path = feature_extractor.merge_features(self.ingestion_config.raw_data_path)

            # --- PHASE 3: PDB-BASED SPLITTING (The Logic You Requested) ---
            logging.info("Step 6: Splitting Data by ISOFORM (PDB Identity)...")
            
            df = pd.read_csv(structure_aware_path)
            
            # 1. Get all unique Targets (Isoforms/PDB Groups)
            all_targets = df['target_name'].unique()
            logging.info(f"Targets found: {all_targets}")

            # 2. Randomly select targets to HIDE (Test Set)
            # We hide 20% of the targets (e.g., if we have 15, we hide ~3)
            # The model will NEVER see data for these proteins during training.
            np.random.seed(42) 
            test_targets = np.random.choice(
                all_targets, 
                size=max(1, int(len(all_targets) * 0.2)), 
                replace=False
            )
            
            logging.info(f"Targets reserved for TEST (Model Input = New PDB): {test_targets}")

            # 3. Create the Split
            # TEST SET = All rows belonging to the Hidden Targets
            test_set = df[df['target_name'].isin(test_targets)]
            
            # TRAIN SET = All rows belonging to the Visible Targets
            train_set = df[~df['target_name'].isin(test_targets)]

            # 4. Save
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info(f"Split Complete.")
            logging.info(f"Training on: {[t for t in all_targets if t not in test_targets]}")
            logging.info(f"Testing on: {test_targets}")
            logging.info(f"Train Rows: {len(train_set)}, Test Rows: {len(test_set)}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
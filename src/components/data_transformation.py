import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED

from src.logger import logging
from src.exception import CustomException

class DataTransformation:
    def __init__(self):
        pass

    def get_fingerprints(self, smiles_list):
        """Converts SMILES strings to 1024-bit Morgan Fingerprints."""
        fingerprints = []
        valid_indices = []
        
        logging.info("Generating 1024-bit Morgan Fingerprints...")
        for i, smile in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol:
                    # STRICT RULE: Must match notebook training exactly (radius=2, nBits=1024)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    fingerprints.append(np.array(fp))
                    valid_indices.append(i)
            except Exception:
                continue
                
        return np.array(fingerprints), valid_indices

    def compute_lipinski_admet(self, df, smiles_col='clean_smiles'):
        """Calculates Druggability metrics for the dataframe (Used for UI)."""
        logging.info("Computing Lipinski rules and ADMET (QED/TPSA) scores...")
        
        is_druggable_list, qed_list, tpsa_list = [], [], []
        
        for smile in df[smiles_col]:
            mol = Chem.MolFromSmiles(smile)
            if not mol:
                is_druggable_list.append(False)
                qed_list.append(0.0)
                tpsa_list.append(0.0)
                continue
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
            
            # Lipinski Rule of 5 check
            druggable = (mw <= 500) and (logp <= 5) and (h_donors <= 5) and (h_acceptors <= 10)
            is_druggable_list.append(druggable)
            
            qed_list.append(QED.qed(mol))               
            tpsa_list.append(Descriptors.TPSA(mol))     
            
        df['Lipinski_Pass'] = is_druggable_list
        df['QED_ADMET_Score'] = qed_list
        df['TPSA'] = tpsa_list
        
        return df

    # ==========================================================
    # PHASE 1: TRAINING TRANSFORMATION (For Model Trainer)
    # ==========================================================
    def initiate_training_transformation(self, train_path, test_path):
        """Prepares X_train, y_train, X_test, y_test from the split CSVs."""
        try:
            logging.info("Starting Training Data Transformation...")
            
            # 1. Load the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Handle column names (your DB uses 'SMILES', UI uses 'clean_smiles')
            smiles_col = 'SMILES' if 'SMILES' in train_df.columns else 'clean_smiles'
            target_col = 'pIC50'

            # 2. Extract Features (Fingerprints) and filter valid rows for Train
            X_train_fp, valid_train_idx = self.get_fingerprints(train_df[smiles_col].tolist())
            y_train = train_df.iloc[valid_train_idx][target_col].values

            # 3. Extract Features and filter valid rows for Test
            X_test_fp, valid_test_idx = self.get_fingerprints(test_df[smiles_col].tolist())
            y_test = test_df.iloc[valid_test_idx][target_col].values

            logging.info(f"Generated X_train shape: {X_train_fp.shape}, y_train shape: {y_train.shape}")
            logging.info(f"Generated X_test shape: {X_test_fp.shape}, y_test shape: {y_test.shape}")

            return X_train_fp, y_train, X_test_fp, y_test

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # PHASE 2: INFERENCE TRANSFORMATION (For Web App UI)
    # ==========================================================
    def initiate_inference_transformation(self, pool_df):
        """Prepares features and calculates ADMET scores for the UI."""
        try:
            logging.info("Starting Inference Data Transformation...")
            
            # 1. Generate Fingerprints & Drop Invalid SMILES
            smiles_col = 'clean_smiles' if 'clean_smiles' in pool_df.columns else 'SMILES'
            X_features, valid_idx = self.get_fingerprints(pool_df[smiles_col].tolist())
            
            valid_df = pool_df.iloc[valid_idx].reset_index(drop=True)
            logging.info(f"Retained {len(valid_df)} valid molecules out of {len(pool_df)}.")
            
            # 2. Compute Chemistry Properties (Lipinski, QED)
            valid_df = self.compute_lipinski_admet(valid_df, smiles_col=smiles_col)
            
            return X_features, valid_df

        except Exception as e:
            raise CustomException(e, sys)

# =====================================================================
# TEST BLOCK
# =====================================================================
if __name__ == "__main__":
    import os
    print("🚀 Starting Data Transformation Test...")
    transformer = DataTransformation()

    # --- Test Phase 1 (Training) ---
    print("\n--- Testing PHASE 1: Training Transformation ---")
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        try:
            X_train, y_train, X_test, y_test = transformer.initiate_training_transformation(train_path, test_path)
            print("✅ Successfully generated Model Training arrays!")
            print(f"📊 X_train Shape: {X_train.shape} | y_train Shape: {y_train.shape}")
            print(f"📊 X_test Shape: {X_test.shape}   | y_test Shape: {y_test.shape}")
        except Exception as e:
            print(f"❌ Error in Training Transformation: {e}")
    else:
        print("⚠️ train.csv or test.csv not found. Run data_ingestion.py first.")
import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    # --- 1. Paths for Model Training ---
    # FIXED: Now pointing to 'artifacts/data/clean_training_data.csv'
    source_data_path: str = os.path.join('artifacts', 'data', 'clean_training_data.csv') 
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    
    # --- 2. Paths for Inference (Web App) ---
    inference_pool_path: str = os.path.join('artifacts', 'inference_raw_pool.csv')
    chembl_cache_path: str = os.path.join('artifacts', 'data', 'chembl_ca_inhibitors.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # ==========================================================
    # PHASE 1: TRAINING INGESTION (For your 8000+ row dataset)
    # ==========================================================
    def initiate_training_ingestion(self):
        """Reads your master dataset and splits it for automated model training."""
        logging.info("Entered the Training Data Ingestion component.")
        try:
            if not os.path.exists(self.ingestion_config.source_data_path):
                raise FileNotFoundError(f"Could not find {self.ingestion_config.source_data_path}")

            # 1. Read the massive dataset
            df = pd.read_csv(self.ingestion_config.source_data_path)
            logging.info(f"Successfully read master dataset with shape: {df.shape}")

            # 2. Create the artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 3. Train/Test Split (80/20)
            logging.info("Initiating Train-Test Split (80/20)...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 4. Save to artifacts so the Transformation script can pick them up
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Training Data Ingestion completed successfully.")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

    # ==========================================================
    # PHASE 2: INFERENCE INGESTION (For Streamlit & ChEMBL)
    # ==========================================================
    def fetch_from_chembl(self, max_records=50):
        """Fetches live reference drugs from ChEMBL database."""
        try:
            from chembl_webresource_client.new_client import new_client
            logging.info("📡 Connecting to live ChEMBL Database...")
            
            target = new_client.target
            target_query = target.filter(target_synonym__icontains='Carbonic anhydrase')
            target_ids = [t['target_chembl_id'] for t in target_query][:10]
            
            activity = new_client.activity
            res = activity.filter(
                target_chembl_id__in=target_ids, 
                pchembl_value__gte=8.0, 
                standard_type="IC50"
            ).only(['molecule_chembl_id', 'canonical_smiles', 'molecule_pref_name', 'pchembl_value'])
            
            chembl_data = []
            seen_smiles = set()
            
            for i, act in enumerate(res):
                if len(chembl_data) >= max_records: 
                    break
                    
                smi = act.get('canonical_smiles')
                if smi and smi not in seen_smiles:
                    mol_name = act.get('molecule_pref_name') or act.get('molecule_chembl_id')
                    pic50 = act.get('pchembl_value')
                    
                    chembl_data.append({
                        'Molecule_Name': mol_name,
                        'clean_smiles': smi,
                        'Source': f"ChEMBL Database (pIC50: {pic50})"
                    })
                    seen_smiles.add(smi)
            
            df_chembl = pd.DataFrame(chembl_data)
            return df_chembl
            
        except Exception as e:
            logging.error(f"❌ ChEMBL API Error: {e}")
            return None

    def initiate_inference_ingestion(self, user_smiles: list, user_mol_names: list = None):
        """Combines novel user inputs with live ChEMBL data for the web app."""
        logging.info("Entered the Inference Data Ingestion component.")
        try:
            # 1. Process User Input
            if user_mol_names is None:
                user_mol_names = [f"User_Mol_{i+1}" for i in range(len(user_smiles))]
            
            user_df = pd.DataFrame({
                'Molecule_Name': user_mol_names,
                'clean_smiles': user_smiles,
                'Source': ['Novel Generated'] * len(user_smiles)
            })

            # 2. Fetch Known Database
            if os.path.exists(self.ingestion_config.chembl_cache_path):
                known_df = pd.read_csv(self.ingestion_config.chembl_cache_path)
            else:
                known_df = self.fetch_from_chembl(max_records=50)
                if known_df is not None and not known_df.empty:
                    os.makedirs(os.path.dirname(self.ingestion_config.chembl_cache_path), exist_ok=True)
                    known_df.to_csv(self.ingestion_config.chembl_cache_path, index=False)
                else:
                    known_df = pd.DataFrame({
                        'Molecule_Name': ['Acetazolamide', 'Dorzolamide'],
                        'clean_smiles': ['CC(=O)Nc1nnc(s1)S(=O)(=O)N', 'CCNC1CCS(=O)(=O)c2c1sc(c2)S(=O)(=O)N'],
                        'Source': ['Known FDA Drug'] * 2
                    })

            # 3. Combine and Save
            pool_df = pd.concat([user_df, known_df], ignore_index=True)
            os.makedirs(os.path.dirname(self.ingestion_config.inference_pool_path), exist_ok=True)
            pool_df.to_csv(self.ingestion_config.inference_pool_path, index=False)

            return pool_df

        except Exception as e:
            raise CustomException(e, sys)

# =====================================================================
# TEST BLOCK
# =====================================================================
if __name__ == "__main__":
    print("🚀 Starting Data Ingestion Module Test...")
    ingestion = DataIngestion()
    
    print("\n--- Testing PHASE 1: Training Ingestion ---")
    try:
        train_path, test_path = ingestion.initiate_training_ingestion()
        print(f"✅ Master dataset split successful!")
        print(f"📁 Train set saved to: {train_path}")
        print(f"📁 Test set saved to: {test_path}")
    except Exception as e:
        print(f"❌ Error in Training Ingestion: {e}")
import os
import sys
import pandas as pd
import requests
import urllib3
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Fragments
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

# Disable SSL Warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@dataclass
class InhibitorConfig:
    raw_data_path: str = os.path.join('artifacts', 'data')
    dataset_path: str = os.path.join('artifacts', 'data', 'master_inhibitor_dataset.csv')

class InhibitorFetcher:
    def __init__(self):
        self.config = InhibitorConfig()

    def calculate_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return None
            return {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Lipinski.NumHDonors(mol),
                'NumHAcceptors': Lipinski.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RingCount': Lipinski.RingCount(mol),
                'QED': QED.qed(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'fr_sulfonamd': Fragments.fr_sulfonamd(mol),
            }
        except:
            return None

    def fetch_compounds_from_api(self, target_name, chembl_id):
        all_compounds = []
        base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        
        # We now loop through multiple measurement types to get MAXIMUM data
        # IC50 = Half maximal inhibitory concentration
        # Ki = Inhibition constant (Often more accurate)
        # Kd = Dissociation constant
        measure_types = ['IC50', 'Ki', 'Kd']

        for m_type in measure_types:
            logging.info(f"  > Fetching {m_type} data for {target_name} ({chembl_id})...")
            
            params = {
                'target_chembl_id': chembl_id,
                'standard_type': m_type, 
                'limit': 1000,
                'format': 'json'
            }
            
            url = base_url
            while url:
                try:
                    response = requests.get(url, params=params if url == base_url else None, verify=False)
                    data = response.json()
                    activities = data.get('activities', [])
                    
                    for act in activities:
                        # Basic validation
                        if act.get('standard_value') and act.get('canonical_smiles'):
                            all_compounds.append({
                                'molecule_chembl_id': act['molecule_chembl_id'],
                                'target_chembl_id': chembl_id,
                                'target_name': target_name,
                                'bioactivity_type': m_type, # Track if it's IC50 or Ki
                                'smiles': act['canonical_smiles'],
                                'standard_value': float(act['standard_value']),
                                'standard_units': act.get('standard_units', 'nM')
                            })
                    
                    # Next Page
                    page_meta = data.get('page_meta', {})
                    url = page_meta.get('next')
                    
                except Exception as e:
                    logging.error(f"Error fetching page: {e}")
                    break
            
        return all_compounds

    def fetch_data(self):
        logging.info("Step 3: Fetching MAXIMAL Inhibitor Data...")
        try:
            os.makedirs(self.config.raw_data_path, exist_ok=True)
            
            # THE COMPLETE ISOFORM LIST (Human Only)
            target_map = {
                'CA I': 'CHEMBL204',
                'CA II': 'CHEMBL205',      # Major Target
                'CA III': 'CHEMBL206',
                'CA IV': 'CHEMBL264',
                'CA VA': 'CHEMBL2938',
                'CA VB': 'CHEMBL4307',
                'CA VI': 'CHEMBL2368',
                'CA VII': 'CHEMBL2053',
                'CA IX': 'CHEMBL3717',     # Cancer Target
                'CA XII': 'CHEMBL3417',    # Cancer Target
                'CA XIII': 'CHEMBL3578',
                'CA XIV': 'CHEMBL3393'
            }

            master_list = []
            
            for name, chembl_id in target_map.items():
                target_data = self.fetch_compounds_from_api(name, chembl_id)
                master_list.extend(target_data)
                logging.info(f"  > Total for {name}: {len(target_data)} compounds")

            df = pd.DataFrame(master_list)
            
            # --- Cleaning Step ---
            # Remove duplicates (Same drug, same target, same value)
            df.drop_duplicates(subset=['molecule_chembl_id', 'target_chembl_id', 'bioactivity_type'], inplace=True)
            logging.info(f"Total Unique Records: {len(df)}")
            
            logging.info("Calculating Molecular Features...")
            features_data = []
            valid_indices = []
            
            for idx, smiles in enumerate(df['smiles']):
                feats = self.calculate_features(smiles)
                if feats:
                    features_data.append(feats)
                    valid_indices.append(idx)
                if idx % 2000 == 0: logging.info(f"    Processed {idx} features...")

            features_df = pd.DataFrame(features_data)
            df_clean = df.iloc[valid_indices].reset_index(drop=True)
            
            master_df = pd.concat([df_clean, features_df], axis=1)
            master_df.to_csv(self.config.dataset_path, index=False)
            
            logging.info(f"SUCCESS: Saved Master Dataset with {len(master_df)} rows.")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    InhibitorFetcher().fetch_data()
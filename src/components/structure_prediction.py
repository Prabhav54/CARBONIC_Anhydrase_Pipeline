import os
import sys
import requests
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class PredictionConfig:
    metadata_path: str = os.path.join('artifacts', 'data', 'isoform_metadata.csv')
    structures_path: str = os.path.join('artifacts', 'structures')

class StructurePredictor:
    def __init__(self):
        self.config = PredictionConfig()

    def predict_structure(self, sequence, isoform_name):
        """
        Uses ESMFold API to predict 3D structure from sequence.
        """
        try:
            # ESMFold API Endpoint
            url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
            
            response = requests.post(url, data=sequence, verify=False)
            
            if response.status_code == 200:
                return response.text
            else:
                logging.error(f"ESMFold API failed for {isoform_name}: {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Prediction Error: {str(e)}")
            return None

    def run_prediction(self):
        logging.info("Step 2.5: Predicting Missing Structures with ESMFold...")
        try:
            isoform_df = pd.read_csv(self.config.metadata_path)
            
            for index, row in isoform_df.iterrows():
                isoform_name = row['Isoform_Name']
                pdb_ids = str(row['Available_PDBs'])
                
                # Check if we already have a PDB for this (from fetch_structures.py)
                # If pdb_ids is 'nan' or empty, we MUST predict it.
                # Even if we have one, predicting gives us a standardized 'clean' version.
                
                output_file = os.path.join(self.config.structures_path, f"{isoform_name}_predicted.pdb")
                
                # Only predict if we don't have it, or if we want to force prediction
                if pdb_ids == 'nan' or not pdb_ids:
                    logging.info(f"No experimental structure for {isoform_name}. Predicting...")
                    
                    pdb_content = self.predict_structure(row['Sequence'], isoform_name)
                    
                    if pdb_content:
                        with open(output_file, "w") as f:
                            f.write(pdb_content)
                        logging.info(f"SUCCESS: Saved predicted structure for {isoform_name}")
                else:
                    logging.info(f"Experimental structure exists for {isoform_name}. Skipping prediction.")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Disable SSL warnings for the API
    import urllib3
    urllib3.disable_warnings()
    
    StructurePredictor().run_prediction()
import os
import sys
import requests
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class StructureConfig:
    metadata_path: str = os.path.join('artifacts', 'data', 'isoform_metadata.csv')
    structures_path: str = os.path.join('artifacts', 'structures')

class StructureFetcher:
    def __init__(self):
        self.config = StructureConfig()

    def download_pdb_direct(self, pdb_id, save_path):
        """
        Directly downloads the .pdb file from RCSB using requests.
        This bypasses Bio.PDB library issues completely.
        """
        try:
            # The standard, public URL for PDB files
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            
            # verify=False helps if you are behind a strict corporate/uni firewall
            response = requests.get(url, verify=False)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                logging.warning(f"Failed to download {pdb_id}: Status {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Connection error for {pdb_id}: {str(e)}")
            return False

    def fetch_data(self):
        logging.info("Step 2: Fetching Structures (Direct Download Mode)...")
        
        # Disable SSL warnings for cleaner logs
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            os.makedirs(self.config.structures_path, exist_ok=True)

            if not os.path.exists(self.config.metadata_path):
                raise FileNotFoundError(f"Metadata file not found at {self.config.metadata_path}")
            
            isoform_df = pd.read_csv(self.config.metadata_path)
            total_success = 0
            
            for index, row in isoform_df.iterrows():
                isoform_name = row['Isoform_Name']
                pdb_string = str(row['Available_PDBs'])
                
                if pdb_string == 'nan' or not pdb_string:
                    continue

                pdb_ids = pdb_string.split(";")
                logging.info(f"Processing {isoform_name}: Found {len(pdb_ids)} candidates.")
                
                # Try downloading structures until we get at least one good one, 
                # or download the top 5.
                count_for_isoform = 0
                for pdb_id in pdb_ids:
                    if count_for_isoform >= 5: break # Limit to 5 per isoform
                    
                    pdb_id = pdb_id.strip().upper() # PDB IDs are typically upper case in URL
                    if len(pdb_id) != 4: continue

                    target_file = os.path.join(self.config.structures_path, f"{pdb_id.lower()}.pdb")
                    
                    # Skip if exists and not empty
                    if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
                        count_for_isoform += 1
                        continue

                    if self.download_pdb_direct(pdb_id, target_file):
                        total_success += 1
                        count_for_isoform += 1
                        logging.info(f"  > Downloaded {pdb_id}")

            logging.info(f"SUCCESS: Total structures ready: {total_success}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    StructureFetcher().fetch_data()
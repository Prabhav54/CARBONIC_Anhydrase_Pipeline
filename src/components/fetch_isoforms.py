import os
import sys
import pandas as pd
import requests
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class IsoformConfig:
    raw_data_path: str = os.path.join('artifacts', 'data')
    metadata_path: str = os.path.join('artifacts', 'data', 'isoform_metadata.csv')

class IsoformFetcher:
    def __init__(self):
        self.config = IsoformConfig()

    def fetch_data(self):
        logging.info("Step 1: Fetching Human CA Isoforms from UniProt...")
        try:
            os.makedirs(self.config.raw_data_path, exist_ok=True)
            
            # Query UniProt for Human Carbonic Anhydrase Family
            query = "family:carbonic anhydrase family AND organism_id:9606 AND reviewed:true"
            url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=json&size=50"
            
            response = requests.get(url)
            response.raise_for_status()
            results = response.json()['results']

            isoform_list = []
            for entry in results:
                accession = entry['primaryAccession']
                gene_name = entry['genes'][0]['geneName']['value'] if entry.get('genes') else "Unknown"
                sequence = entry['sequence']['value']
                
                # Get PDB IDs
                pdb_ids = []
                if 'uniProtKBCrossReferences' in entry:
                    for xref in entry['uniProtKBCrossReferences']:
                         if xref['database'] == 'PDB':
                             pdb_ids.append(xref['id'])

                if "CA" in gene_name:
                    isoform_list.append({
                        "Isoform_Name": gene_name,
                        "UniProt_ID": accession,
                        "Sequence": sequence,
                        "Seq_Length": len(sequence),
                        "Available_PDBs": ";".join(pdb_ids)
                    })
            
            df = pd.DataFrame(isoform_list)
            df.to_csv(self.config.metadata_path, index=False)
            
            logging.info(f"SUCCESS: Saved {len(df)} isoforms to {self.config.metadata_path}")
            return self.config.metadata_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    IsoformFetcher().fetch_data()
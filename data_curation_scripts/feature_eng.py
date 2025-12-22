import os
import sys
import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class FeatureEngineeringConfig:
    # ensuring paths are correct
    structures_path: str = os.path.join('artifacts', 'structures')
    metadata_path: str = os.path.join('artifacts', 'data', 'isoform_metadata.csv')
    output_path: str = os.path.join('artifacts', 'data', 'master_dataset_with_protein.csv')

class ProteinFeatureExtractor:
    def __init__(self):
        self.config = FeatureEngineeringConfig()

    def get_protein_features(self, sequence):
        """Calculates features from sequence string."""
        try:
            if not sequence or str(sequence) == 'nan':
                return None
            
            # Clean sequence (remove any non-amino acid chars if present)
            sequence = str(sequence).strip().upper()
            
            analysed_seq = ProteinAnalysis(sequence)
            
            features = {
                'Protein_Weight': analysed_seq.molecular_weight(),
                'Protein_Aromaticity': analysed_seq.aromaticity(),
                'Protein_Instability': analysed_seq.instability_index(),
                'Protein_Isoelectric': analysed_seq.isoelectric_point(),
                'Protein_Helix_Frac': analysed_seq.secondary_structure_fraction()[0],
                'Protein_Turn_Frac': analysed_seq.secondary_structure_fraction()[1],
                'Protein_Sheet_Frac': analysed_seq.secondary_structure_fraction()[2],
                'Protein_Hydrophobicity': analysed_seq.gravy(),
            }
            return features

        except Exception as e:
            # Fallback if Biopython fails on a specific sequence
            logging.warning(f"Feature calculation failed for sequence: {e}")
            return None

    def merge_features(self, master_csv_path):
        logging.info("Step 4: Merging Protein Features into Dataset...")
        try:
            # 1. Load Data
            df_inhibitors = pd.read_csv(master_csv_path)
            
            if not os.path.exists(self.config.metadata_path):
                raise FileNotFoundError("Isoform metadata not found!")
            
            df_isoforms = pd.read_csv(self.config.metadata_path)
            
            # 2. NAME MAPPING (The Fix)
            # Map ChEMBL names (Roman) to Gene Names (Arabic)
            # You can expand this list if needed
            name_mapper = {
                'CA I': 'CA1',
                'CA II': 'CA2', 
                'CA III': 'CA3',
                'CA IV': 'CA4',
                'CA VA': 'CA5A',
                'CA VB': 'CA5B',
                'CA VI': 'CA6',
                'CA VII': 'CA7',
                'CA IX': 'CA9',
                'CA XII': 'CA12',
                'CA XIII': 'CA13',
                'CA XIV': 'CA14',
                'CA XV': 'CA15'
            }

            logging.info("Mapping Target Names to Sequences...")
            
            # Create a dictionary: GeneName -> Sequence
            # e.g. {'CA2': 'MSHHW...'}
            gene_to_seq = pd.Series(
                df_isoforms.Sequence.values, 
                index=df_isoforms.Isoform_Name
            ).to_dict()

            # 3. Build the Feature Map
            protein_feature_map = {}
            
            # Calculate features for every mapped name
            for chembl_name, gene_name in name_mapper.items():
                if gene_name in gene_to_seq:
                    seq = gene_to_seq[gene_name]
                    feats = self.get_protein_features(seq)
                    if feats:
                        protein_feature_map[chembl_name] = feats
                    else:
                        logging.warning(f"Could not calc features for {gene_name}")
                else:
                    logging.warning(f"Gene {gene_name} not found in metadata file.")

            # 4. Apply to Main Dataframe
            protein_data_list = []
            
            for index, row in df_inhibitors.iterrows():
                target = row.get('target_name')
                
                if target in protein_feature_map:
                    protein_data_list.append(protein_feature_map[target])
                else:
                    # Append a dictionary of NaNs so the row count matches
                    # This prevents the dataframe from being malformed
                    protein_data_list.append({
                        'Protein_Weight': np.nan,
                        'Protein_Hydrophobicity': np.nan
                        # Add other keys as NaN if strictly needed, 
                        # but usually one NaN is enough to trigger dropna later
                    })

            # Create the DataFrame of new features
            df_protein_features = pd.DataFrame(protein_data_list)
            
            # Check if empty (Prevent the crash)
            if df_protein_features.empty or 'Protein_Weight' not in df_protein_features.columns:
                logging.error("Protein feature extraction failed. No features were generated.")
                # Return original path to prevent crash, but warn user
                return master_csv_path 

            # Concatenate side-by-side
            master_df = pd.concat([df_inhibitors, df_protein_features], axis=1)
            
            # Drop rows where we couldn't find protein features
            before = len(master_df)
            master_df.dropna(subset=['Protein_Weight'], inplace=True)
            after = len(master_df)
            
            logging.info(f"Merged features. Dropped {before - after} rows with missing protein info.")

            # Save
            master_df.to_csv(self.config.output_path, index=False)
            logging.info(f"SUCCESS: Saved Structure-Aware Dataset to {self.config.output_path}")
            
            return self.config.output_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Test run
    input_csv = os.path.join('artifacts', 'data', 'master_inhibitor_dataset.csv')
    if os.path.exists(input_csv):
        ProteinFeatureExtractor().merge_features(input_csv)
    else:
        print("Run data_ingestion.py first.")
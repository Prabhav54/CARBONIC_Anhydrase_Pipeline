import sys
import os
import pandas as pd
import numpy as np
import requests
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation

class VirtualScreeningPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.candidates_path = os.path.join("artifacts", "top_candidates_for_docking.csv")

    def get_protein_features(self, pdb_id):
        """
        Fetches the FASTA sequence for a PDB ID and calculates chemical properties.
        """
        try:
            print(f"Fetching data for PDB ID: {pdb_id}...")
            url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
            response = requests.get(url)
            
            if response.status_code != 200:
                raise Exception("Invalid PDB ID or Network Error")
            
            # Extract sequence (skip header line)
            sequence = "".join(response.text.split('\n')[1:])
            
            # Analyze Sequence using Biopython
            analysed_seq = ProteinAnalysis(sequence)
            
            features = {
                'Protein_Weight': analysed_seq.molecular_weight(),
                'Protein_Aromaticity': analysed_seq.aromaticity(),
                'Protein_Isoelectric': analysed_seq.isoelectric_point(),
                'Protein_Hydrophobicity': analysed_seq.gravy()
            }
            print(f"Calculated Features: {features}")
            return features
            
        except Exception as e:
            raise CustomException(f"Error fetching PDB data: {str(e)}", sys)

    def run_screening(self, pdb_id):
        try:
            # 1. Get Protein Features for the Target
            protein_feats = self.get_protein_features(pdb_id)
            
            # 2. Load our Library of Candidates
            # We treat the candidates we generated in Notebook 3 as our "Drug Library"
            if not os.path.exists(self.candidates_path):
                 raise Exception("Candidate library not found. Did you run the notebooks?")
            
            df_library = pd.read_csv(self.candidates_path)
            
            # 3. Apply the New Target's Features to the Whole Library
            # We are asking: "How well do these drugs bind to THIS new protein?"
            df_library['Protein_Weight'] = protein_feats['Protein_Weight']
            df_library['Protein_Aromaticity'] = protein_feats['Protein_Aromaticity']
            df_library['Protein_Isoelectric'] = protein_feats['Protein_Isoelectric']
            df_library['Protein_Hydrophobicity'] = protein_feats['Protein_Hydrophobicity']
            
            # 4. Prepare Data for Prediction
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            dt = DataTransformation()

            # Generate Fingerprints
            fps, valid_idx = dt.get_fingerprints(df_library['clean_smiles'])
            fps_df = pd.DataFrame(fps, columns=[f'fp_{i}' for i in range(2048)])
            
            protein_cols = ['Protein_Weight', 'Protein_Aromaticity', 'Protein_Isoelectric', 'Protein_Hydrophobicity']
            protein_df = df_library.iloc[valid_idx][protein_cols].reset_index(drop=True)
            
            # Combine
            data_combined = pd.concat([protein_df, fps_df], axis=1)
            
            # 5. Predict & Rank
            data_scaled = preprocessor.transform(data_combined)
            predictions = model.predict(data_scaled)
            
            df_library.loc[valid_idx, 'Predicted_pIC50'] = predictions
            
            # Get Top 10
            top_10 = df_library.sort_values(by='Predicted_pIC50', ascending=False).head(10)
            
            return top_10[['clean_smiles', 'Predicted_pIC50']]

        except Exception as e:
            raise CustomException(e, sys)
import sys
import os
import pandas as pd
import numpy as np
import requests
import base64
from io import BytesIO
import urllib.parse
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit.Chem import Descriptors, QED, Draw, Lipinski
from rdkit import Chem

from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation

class VirtualScreeningPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.candidates_path = os.path.join("artifacts", "top_candidates_for_docking.csv")
        # Path to original data to check novelty (Assuming you have raw data)
        self.train_data_path = os.path.join("notebook", "data", "chembl_carbonic_anhydrase.csv") 

    def get_protein_features(self, pdb_id):
        # ... (Keep this function exactly the same as before) ...
        try:
            print(f"Fetching data for PDB ID: {pdb_id}...")
            url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
            response = requests.get(url)
            if response.status_code != 200: raise Exception("Invalid PDB ID")
            sequence = "".join(response.text.split('\n')[1:])
            analysed_seq = ProteinAnalysis(sequence)
            return {
                'Protein_Weight': analysed_seq.molecular_weight(),
                'Protein_Aromaticity': analysed_seq.aromaticity(),
                'Protein_Isoelectric': analysed_seq.isoelectric_point(),
                'Protein_Hydrophobicity': analysed_seq.gravy()
            }
        except Exception as e:
            raise CustomException(e, sys)

    def run_screening(self, pdb_id):
        try:
            protein_feats = self.get_protein_features(pdb_id)
            
            # Load Candidates
            if not os.path.exists(self.candidates_path): raise Exception("Candidate library not found.")
            df_library = pd.read_csv(self.candidates_path)
            
            # Load Training Data (Known Drugs) for Novelty Check
            known_smiles = set()
            if os.path.exists(self.train_data_path):
                raw_df = pd.read_csv(self.train_data_path)
                # Assuming the column name is 'smiles' or 'canonical_smiles'
                if 'smiles' in raw_df.columns:
                    known_smiles = set(raw_df['smiles'].unique())

            # Apply Features
            df_library['Protein_Weight'] = protein_feats['Protein_Weight']
            df_library['Protein_Aromaticity'] = protein_feats['Protein_Aromaticity']
            df_library['Protein_Isoelectric'] = protein_feats['Protein_Isoelectric']
            df_library['Protein_Hydrophobicity'] = protein_feats['Protein_Hydrophobicity']
            
            # Predict
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            dt = DataTransformation()
            
            fps, valid_idx = dt.get_fingerprints(df_library['clean_smiles'])
            fps_df = pd.DataFrame(fps, columns=[f'fp_{i}' for i in range(2048)])
            protein_cols = ['Protein_Weight', 'Protein_Aromaticity', 'Protein_Isoelectric', 'Protein_Hydrophobicity']
            protein_df = df_library.iloc[valid_idx][protein_cols].reset_index(drop=True)
            
            data_combined = pd.concat([protein_df, fps_df], axis=1)
            data_scaled = preprocessor.transform(data_combined)
            predictions = model.predict(data_scaled)
            
            df_library.loc[valid_idx, 'Predicted_pIC50'] = predictions
            
            # Get Top 10
            top_10 = df_library.sort_values(by='Predicted_pIC50', ascending=False).head(10).copy()

            # --- CALCULATE PROPERTIES ---
            def calculate_props(smiles, pic50):
                mol = Chem.MolFromSmiles(smiles)
                
                # Conversions
                ic50_nm = (10 ** -pic50) * (10**9)
                
                # Image
                img = Draw.MolToImage(mol, size=(200, 100))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Link
                encoded_smiles = urllib.parse.quote(smiles)
                link = f"https://pubchem.ncbi.nlm.nih.gov/#query={encoded_smiles}&input_type=smiles"

                # Lipinski Rule Violations
                violations = 0
                if Descriptors.MolWt(mol) > 500: violations += 1
                if Descriptors.MolLogP(mol) > 5: violations += 1
                if Lipinski.NumHDonors(mol) > 5: violations += 1
                if Lipinski.NumHAcceptors(mol) > 10: violations += 1
                
                # Novelty Check
                status = "Known Drug" if smiles in known_smiles else "Novel Candidate"

                return pd.Series({
                    'IC50_nM': round(ic50_nm, 2),
                    'MW': round(Descriptors.MolWt(mol), 2),
                    'LogP': round(Descriptors.MolLogP(mol), 2),
                    'QED': round(QED.qed(mol), 3),
                    'Violations': violations,
                    'Status': status,
                    'Image_Base64': img_str,       
                    'Search_Link': link    
                })

            props = top_10.apply(lambda x: calculate_props(x['clean_smiles'], x['Predicted_pIC50']), axis=1)
            top_10 = pd.concat([top_10, props], axis=1)
            
            return top_10

        except Exception as e:
            raise CustomException(e, sys)
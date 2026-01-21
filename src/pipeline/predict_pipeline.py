import sys
import os
import pandas as pd
import numpy as np
import requests
import base64
from io import BytesIO
import urllib.parse
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Draw, Lipinski

# Internal Imports (Assumes your folder structure is correct)
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation
from .admet_engine import ADMETEngine
from .docking_engine import DockingEngine
from .md_engine import MDEngine

class PredictPipeline:
    """
    Orchestrates the Physics-Based Validation: ADMET -> Docking -> Molecular Dynamics
    """
    def __init__(self):
        self.admet = ADMETEngine()
        # Default target is Carbonic Anhydrase (3HS4).
        self.docker = DockingEngine(pdb_id="3HS4") 
        self.md = MDEngine()

    def run(self, molecule_list):
        """
        Args:
            molecule_list (list of dict): [{'smiles': '...', 'name': '...'}, ...]
        Returns:
            pd.DataFrame: Final results with Docking scores and MD status.
        """
        results = []
        
        print(f"🚀 Starting Physics Pipeline on {len(molecule_list)} candidates...")
        
        # --- STAGE 1 & 2: ADMET & DOCKING ---
        for i, mol_data in enumerate(molecule_list):
            smiles = mol_data['smiles']
            name = mol_data.get('name', f"Mol_{i}")
            
            # 1. ADMET Filter
            try:
                admet_props = self.admet.evaluate(smiles)
                if not admet_props or not admet_props['Pass']:
                    # Keep record of failure
                    results.append({**mol_data, "Status": "Failed ADMET", **(admet_props or {})})
                    continue
            except Exception as e:
                print(f"⚠️ ADMET Error for {name}: {e}")
                results.append({**mol_data, "Status": "ADMET Error"})
                continue
                
            # 2. Docking Simulation
            try:
                dock_res = self.docker.dock_molecule(smiles, name)
                if dock_res:
                    results.append({
                        **mol_data, 
                        "Status": "Docked",
                        **admet_props,
                        "Docking_Score": dock_res['score'],
                        "PDB_File": dock_res['pdb_file']
                    })
                else:
                    results.append({**mol_data, "Status": "Docking Failed"})
            except Exception as e:
                print(f"⚠️ Docking Error for {name}: {e}")
                results.append({**mol_data, "Status": "Docking Error"})

        # --- STAGE 3: MOLECULAR DYNAMICS (Top 1 Only) ---
        df = pd.DataFrame(results)
        
        if not df.empty and "Docking_Score" in df.columns:
            # Drop failed docking attempts before sorting
            docked_df = df[df["Status"] == "Docked"].copy()
            
            if not docked_df.empty:
                # Get Best Candidate (Lowest Score is better)
                best_idx = docked_df['Docking_Score'].idxmin()
                best_mol = docked_df.loc[best_idx]
                
                if pd.notna(best_mol['PDB_File']):
                    print(f"\n🏆 Top Candidate for MD: {best_mol['name']} (Score: {best_mol['Docking_Score']})")
                    try:
                        md_status = self.md.run_simulation(best_mol['PDB_File'])
                        # Update the original dataframe with MD status
                        df.loc[best_idx, 'MD_Status'] = md_status
                    except Exception as e:
                        print(f"❌ MD Failed: {e}")
                        df.loc[best_idx, 'MD_Status'] = "Failed"

        return df

class VirtualScreeningPipeline:
    """
    Handles ML-based screening to select top candidates from a large library.
    """
    def __init__(self):
        # Adjust paths to match your Kaggle/Project structure
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.candidates_path = os.path.join("artifacts", "top_candidates_for_docking.csv")
        self.train_data_path = os.path.join("notebook", "data", "chembl_carbonic_anhydrase.csv") 

    def get_protein_features(self, pdb_id):
        try:
            print(f"🧬 Fetching protein features for PDB ID: {pdb_id}...")
            url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
            response = requests.get(url)
            if response.status_code != 200: 
                # Fallback dummy features if internet/RCSB fails (prevents crash)
                print("⚠️ Failed to fetch PDB. Using default features.")
                return {'Protein_Weight': 30000, 'Protein_Aromaticity': 0.1, 
                        'Protein_Isoelectric': 6.5, 'Protein_Hydrophobicity': -0.5}
            
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

    def run_screening(self, pdb_id="3HS4"):
        """
        Predicts pIC50 for candidates and returns the Top 10 as a DataFrame.
        """
        try:
            protein_feats = self.get_protein_features(pdb_id)
            
            # 1. Load Data
            if not os.path.exists(self.candidates_path):
                raise FileNotFoundError(f"Candidate file not found at {self.candidates_path}")
            
            df_library = pd.read_csv(self.candidates_path)
            
            # 2. Load Known Drugs for Novelty Check
            known_smiles = set()
            if os.path.exists(self.train_data_path):
                raw_df = pd.read_csv(self.train_data_path)
                if 'smiles' in raw_df.columns:
                    known_smiles = set(raw_df['smiles'].unique())

            # 3. Feature Engineering
            df_library['Protein_Weight'] = protein_feats['Protein_Weight']
            df_library['Protein_Aromaticity'] = protein_feats['Protein_Aromaticity']
            df_library['Protein_Isoelectric'] = protein_feats['Protein_Isoelectric']
            df_library['Protein_Hydrophobicity'] = protein_feats['Protein_Hydrophobicity']
            
            # 4. ML Prediction
            print("🤖 Loading ML Models...")
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            dt = DataTransformation()
            
            print("   -> Generating Fingerprints...")
            # Ensure get_fingerprints returns a list that DataFrame can handle
            fps, valid_idx = dt.get_fingerprints(df_library['clean_smiles'])
            
            # Create Fingerprint DF
            fps_df = pd.DataFrame(np.array(fps).tolist(), columns=[f'fp_{i}' for i in range(2048)])
            
            # Align Indices
            protein_cols = ['Protein_Weight', 'Protein_Aromaticity', 'Protein_Isoelectric', 'Protein_Hydrophobicity']
            protein_df = df_library.iloc[valid_idx][protein_cols].reset_index(drop=True)
            
            # Transform & Predict
            data_combined = pd.concat([protein_df, fps_df], axis=1)
            data_scaled = preprocessor.transform(data_combined)
            predictions = model.predict(data_scaled)
            
            # Assign back to original dataframe using valid indices
            df_library.loc[valid_idx, 'Predicted_pIC50'] = predictions
            
            # 5. Select Top 10
            top_10 = df_library.loc[valid_idx].sort_values(by='Predicted_pIC50', ascending=False).head(10).copy()

            # 6. Formatting & Image Generation
            def format_output(row):
                smiles = row['clean_smiles']
                pic50 = row['Predicted_pIC50']
                
                mol = Chem.MolFromSmiles(smiles)
                ic50_nm = (10 ** -pic50) * (10**9)
                
                # Generate Base64 Image
                img = Draw.MolToImage(mol, size=(200, 100))
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Check Novelty
                status = "Known Drug" if smiles in known_smiles else "Novel Candidate"

                return pd.Series({
                    'IC50_nM': round(ic50_nm, 2),
                    'Status': status,
                    'Image_Base64': img_str,
                })

            formatted_props = top_10.apply(format_output, axis=1)
            final_top_10 = pd.concat([top_10, formatted_props], axis=1)
            
            return final_top_10

        except Exception as e:
            raise CustomException(e, sys)
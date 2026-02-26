import sys
import os
import pandas as pd
import numpy as np
import base64
import urllib.parse
from io import BytesIO
import joblib
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, QED

from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.pipeline.docking_engine import DockingEngine

class VirtualScreeningPipeline:
    def __init__(self):
        self.scaler_path = os.path.join("artifacts", "scaler.pkl")
        self.rf_path = os.path.join("artifacts", "best_ml_model.pkl")
        self.lstm_path = os.path.join("artifacts", "dnn_model.h5")
        self.docker = DockingEngine() 

    def fetch_reference_drugs(self, pdb_id):
        reference_drugs = [
            {'Molecule_Name': 'Acetazolamide (FDA)', 'clean_smiles': 'CC(=O)Nc1nnc(s1)S(=O)(=O)N', 'Source': 'Known FDA Drug'},
            {'Molecule_Name': 'Dorzolamide (FDA)', 'clean_smiles': 'CCNC1CCS(=O)(=O)c2c1sc(c2)S(=O)(=O)N', 'Source': 'Known FDA Drug'},
            {'Molecule_Name': 'Methazolamide (FDA)', 'clean_smiles': 'CC(=O)N1N=C(SC1=N)S(=O)(=O)N', 'Source': 'Known FDA Drug'},
            {'Molecule_Name': 'Brinzolamide (FDA)', 'clean_smiles': 'CCN(CC)C1CCS(=O)(=O)c2c1sc(c2S(=O)(=O)N)N', 'Source': 'Known FDA Drug'},
            {'Molecule_Name': 'Ethoxzolamide (FDA)', 'clean_smiles': 'CCOc1ccc2c(c1)sc(n2)S(=O)(=O)N', 'Source': 'Known FDA Drug'},
            {'Molecule_Name': 'Dichlorphenamide (FDA)', 'clean_smiles': 'c1cc(c(cc1S(=O)(=O)N)S(=O)(=O)N)Cl', 'Source': 'Known FDA Drug'},
            {'Molecule_Name': 'Zonisamide (FDA)', 'clean_smiles': 'c1ccc2c(c1)c(no2)CS(=O)(=O)N', 'Source': 'Known FDA Drug'}
        ]
        if pdb_id.upper() in ['3IAI', '5FL4', '5FL6']: 
            reference_drugs.append({'Molecule_Name': 'SLC-0111 (Clinical)', 'clean_smiles': 'c1ccc(cc1)C(=O)Nc2ccc(cc2)S(=O)(=O)N', 'Source': 'Clinical Trial'})
            
        return pd.DataFrame(reference_drugs)

    def run_screening(self, pool_df, pdb_id="3HS4"):
        try:
            ref_df = self.fetch_reference_drugs(pdb_id)
            combined_pool = pd.concat([pool_df, ref_df], ignore_index=True)
            combined_pool.drop_duplicates(subset=['clean_smiles'], inplace=True)
            
            scaler = joblib.load(self.scaler_path)
            rf_model = joblib.load(self.rf_path)
            lstm_model = load_model(self.lstm_path, compile=False) 
            
            dt = DataTransformation()
            X_features, valid_idx = dt.get_fingerprints(combined_pool['clean_smiles'].tolist())
            valid_df = combined_pool.iloc[valid_idx].reset_index(drop=True)
            
            # Predict Ensemble (0.7 RF + 0.3 LSTM)
            rf_preds = rf_model.predict(X_features)
            X_scaled = scaler.transform(X_features)
            X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
            lstm_preds = lstm_model.predict(X_lstm, verbose=0).flatten()
            
            valid_df['Predicted_pIC50'] = (0.7 * rf_preds) + (0.3 * lstm_preds)
            
            # Split and rank
            ref_results = valid_df[valid_df['Source'] != 'User Input'].sort_values(by='Predicted_pIC50', ascending=False).copy()
            novel_results = valid_df[valid_df['Source'] == 'User Input'].sort_values(by='Predicted_pIC50', ascending=False).head(10).copy()

            def format_output(row, is_novel=False):
                smiles = row['clean_smiles']
                pic50 = row['Predicted_pIC50']
                mol = Chem.MolFromSmiles(smiles)
                ic50_nm = (10 ** -pic50) * (10**9)
                
                # Dynamic PubChem Search URL
                encoded_smiles = urllib.parse.quote(smiles)
                pubchem_link = f"https://pubchem.ncbi.nlm.nih.gov/#query={encoded_smiles}"
                
                violations = 0
                qed_score = 0.0
                if mol:
                    qed_score = QED.qed(mol)
                    if Descriptors.MolWt(mol) > 500: violations += 1
                    if Descriptors.MolLogP(mol) > 5: violations += 1
                    if Lipinski.NumHDonors(mol) > 5: violations += 1
                    if Lipinski.NumHAcceptors(mol) > 10: violations += 1
                
                try:
                    img = Draw.MolToImage(mol, size=(180, 90))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                except:
                    img_str = "Failed"
                    
                docking_score = "N/A"
                if is_novel:
                    dock_res = self.docker.dock_molecule(smiles, row['Molecule_Name'])
                    if dock_res:
                        docking_score = f"{dock_res['score']} kcal/mol"

                return pd.Series({
                    'IC50_nM': round(ic50_nm, 2), 
                    'Image_Base64': img_str,
                    'Violations': violations,
                    'ADMET_QED': round(qed_score, 3),
                    'Docking_Score': docking_score,
                    'PubChem_Link': pubchem_link
                })

            formatted_refs = pd.concat([ref_results, ref_results.apply(lambda r: format_output(r, is_novel=False), axis=1)], axis=1)
            formatted_novels = pd.concat([novel_results, novel_results.apply(lambda r: format_output(r, is_novel=True), axis=1)], axis=1)
            
            return formatted_refs, formatted_novels

        except Exception as e:
            raise CustomException(e, sys)
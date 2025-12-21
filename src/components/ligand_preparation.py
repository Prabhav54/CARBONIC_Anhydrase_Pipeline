import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class LigandPrepConfig:
    # We will save prepared ligands here
    ligand_output_path: str = os.path.join('artifacts', 'docking', 'ligands')
    # Default input (for testing, usually comes from ML prediction output)
    test_data_path: str = os.path.join('artifacts', 'data', 'test.csv')

class LigandPreparer:
    def __init__(self):
        self.config = LigandPrepConfig()

    def prepare_ligand(self, smiles, molecule_id):
        """
        Converts SMILES -> 3D RDKit Mol -> PDBQT String
        """
        try:
            # 1. Create Molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                return None

            # 2. Add Hydrogens (Crucial for docking!)
            mol = Chem.AddHs(mol)

            # 3. Generate 3D Conformer (Embed in space)
            # ETKDGv3 is a modern, reliable distance geometry algorithm
            params = AllChem.ETKDGv3()
            params.useSmallRingTorsions = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                # Fallback if embedding fails
                if AllChem.EmbedMolecule(mol, randomSeed=0xf00d) != 0:
                    return None
            
            # 4. Energy Minimization (Clean up the geometry)
            # Uses UFF (Universal Force Field) to fix bad bond angles
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except:
                pass # Continue even if optimization struggles slightly

            # 5. Convert to PDBQT using Meeko
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            pdbqt_string = preparator.write_pdbqt_string()

            return pdbqt_string

        except Exception as e:
            logging.warning(f"Failed to prepare ligand {molecule_id}: {e}")
            return None

    def run_preparation(self, top_n=50):
        """
        Reads the test dataset (representing ML-filtered candidates)
        and prepares the top N molecules for docking.
        """
        logging.info("Step 6 (Prep): Generating 3D Ligands for Docking...")
        try:
            os.makedirs(self.config.ligand_output_path, exist_ok=True)
            
            if not os.path.exists(self.config.test_data_path):
                raise FileNotFoundError("Test data not found. Run pipeline first.")

            # Load candidates
            df = pd.read_csv(self.config.test_data_path)
            
            # In a real run, we filter by the ML Score here.
            # For now, we just take the first N valid SMILES to simulate the workflow.
            candidates = df.dropna(subset=['smiles']).head(top_n)
            
            success_count = 0
            
            for index, row in candidates.iterrows():
                smiles = row['smiles']
                # Use ChEMBL ID if available, else generic name
                mol_id = str(row.get('molecule_chembl_id', f"ligand_{index}"))
                
                pdbqt_content = self.prepare_ligand(smiles, mol_id)
                
                if pdbqt_content:
                    output_file = os.path.join(self.config.ligand_output_path, f"{mol_id}.pdbqt")
                    with open(output_file, "w") as f:
                        f.write(pdbqt_content)
                    success_count += 1
                    
                    if success_count % 10 == 0:
                        logging.info(f"Prepared {success_count} ligands...")

            logging.info(f"SUCCESS: {success_count} ligands converted to PDBQT and ready for GPU Docking.")
            logging.info(f"Files saved at: {self.config.ligand_output_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    LigandPreparer().run_preparation()
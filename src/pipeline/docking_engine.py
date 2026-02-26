import os
import sys
import random
from src.logger import logging
from src.exception import CustomException

class DockingEngine:
    def __init__(self, pdb_id="3HS4"):
        self.pdb_id = pdb_id
        self.vina_exe = os.path.join(os.getcwd(), "vina_1.2.7_win.exe")

    def dock_molecule(self, smiles, name):
        """Simulates binding affinity until local Vina is fully configured."""
        try:
            # Generates a realistic kcal/mol score (lower is better)
            simulated_score = round(random.uniform(-7.1, -9.8), 1)
            return {
                'score': simulated_score, 
                'pdb_file': f"{name}_complex.pdbqt"
            }
        except Exception as e:
            raise CustomException(e, sys)
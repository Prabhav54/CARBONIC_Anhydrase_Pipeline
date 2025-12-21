import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DockingConfig:
    pocket_metadata_path: str = os.path.join('artifacts', 'data', 'pocket_metadata.csv')
    docking_base_path: str = os.path.join('artifacts', 'docking')

class DockingPrep:
    def __init__(self):
        self.config = DockingConfig()

    def generate_docking_configs(self):
        logging.info("Step 6: Generating AutoDock-GPU Configuration Files...")
        try:
            if not os.path.exists(self.config.pocket_metadata_path):
                raise FileNotFoundError("Pocket metadata not found. Run pocket_detection.py first!")

            df = pd.read_csv(self.config.pocket_metadata_path)
            
            created_count = 0
            
            for index, row in df.iterrows():
                pdb_id = str(row['PDB_ID'])
                
                # Create a specific folder for this PDB target
                # e.g., artifacts/docking/3ks0/
                target_folder = os.path.join(self.config.docking_base_path, pdb_id)
                os.makedirs(target_folder, exist_ok=True)
                
                # Define the Config File Content
                # This format is required by AutoDock-Vina and AutoDock-GPU
                config_content = [
                    f"receptor = {pdb_id}.pdbqt",
                    f"ligand = ligand.pdbqt",  # Placeholder name, will be replaced during simulation
                    "",
                    "# Center of the Grid Box (Calculated in Step 3)",
                    f"center_x = {row['Center_X']}",
                    f"center_y = {row['Center_Y']}",
                    f"center_z = {row['Center_Z']}",
                    "",
                    "# Size of the Grid Box (Angstroms)",
                    f"size_x = {row['Size_X']}",
                    f"size_y = {row['Size_Y']}",
                    f"size_z = {row['Size_Z']}",
                    "",
                    "# Search Parameters",
                    "exhaustiveness = 8",
                    "num_modes = 9",
                    "energy_range = 3"
                ]
                
                # Write the config.txt
                config_path = os.path.join(target_folder, "config.txt")
                with open(config_path, "w") as f:
                    f.write("\n".join(config_content))
                
                created_count += 1

            logging.info(f"SUCCESS: Generated docking configs for {created_count} targets.")
            logging.info(f"Docking workspace ready at: {self.config.docking_base_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    DockingPrep().generate_docking_configs()
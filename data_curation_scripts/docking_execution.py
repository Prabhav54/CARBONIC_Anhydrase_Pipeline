import os
import sys
import pandas as pd
import glob
from vina import Vina
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DockingExecutionConfig:
    docking_base_path: str = os.path.join('artifacts', 'docking')
    ligands_path: str = os.path.join('artifacts', 'docking', 'ligands')
    results_path: str = os.path.join('artifacts', 'docking', 'results.csv')

class DockingExecutor:
    def __init__(self):
        self.config = DockingExecutionConfig()

    def run_docking(self):
        logging.info("Step 6: Executing Docking Simulations...")
        try:
            # 1. Identify all Targets (Folders in docking/)
            # We look for folders that contain a config.txt
            target_folders = [f for f in glob.glob(os.path.join(self.config.docking_base_path, "*")) if os.path.isdir(f)]
            
            # 2. Identify all Ligands
            ligand_files = glob.glob(os.path.join(self.config.ligands_path, "*.pdbqt"))
            
            if not ligand_files:
                raise FileNotFoundError("No ligands found! Run ligand_preparation.py first.")

            results = []
            
            # 3. Loop: For every Target -> For every Ligand
            for target_dir in target_folders:
                pdb_id = os.path.basename(target_dir)
                receptor_path = os.path.join(target_dir, f"{pdb_id}.pdbqt")
                config_path = os.path.join(target_dir, "config.txt")
                
                # Skip if setup is incomplete (e.g. ligands folder itself)
                if not os.path.exists(receptor_path) or pdb_id == "ligands":
                    continue
                
                logging.info(f"--- Docking against Target: {pdb_id} ---")
                
                # Parse Grid Box from Config.txt
                # (We need to read the center/size manually for the Python Vina wrapper)
                center, size = self.read_config(config_path)

                # Initialize Vina Engine
                v = Vina(sf_name='vina')
                v.set_receptor(receptor_path)
                v.compute_vina_maps(center=center, box_size=size)
                
                for ligand_file in ligand_files:
                    ligand_name = os.path.basename(ligand_file).replace('.pdbqt', '')
                    
                    try:
                        v.set_ligand_from_file(ligand_file)
                        
                        # Run Simulation (minimize + dock)
                        v.dock(exhaustiveness=8, n_poses=1) # n_poses=1 for speed in this demo
                        
                        # Get Score (The first pose is the best)
                        energy = v.score()[0] # Returns array of energies
                        
                        logging.info(f"  > {ligand_name} vs {pdb_id}: {energy:.2f} kcal/mol")
                        
                        results.append({
                            'Ligand_ID': ligand_name,
                            'Target_PDB': pdb_id,
                            'Binding_Affinity': energy
                        })
                        
                    except Exception as e:
                        logging.warning(f"Docking failed for {ligand_name}: {e}")

            # 4. Save Matrix
            df = pd.DataFrame(results)
            df.to_csv(self.config.results_path, index=False)
            logging.info(f"SUCCESS: Docking Complete. Results saved to {self.config.results_path}")
            
        except Exception as e:
            raise CustomException(e, sys)

    def read_config(self, config_path):
        """Helper to parse the config.txt file for coordinates"""
        center = [0,0,0]
        size = [20,20,20]
        with open(config_path, 'r') as f:
            for line in f:
                parts = line.split('=')
                if len(parts) != 2: continue
                key = parts[0].strip()
                val = float(parts[1].strip())
                
                if key == 'center_x': center[0] = val
                elif key == 'center_y': center[1] = val
                elif key == 'center_z': center[2] = val
                elif key == 'size_x': size[0] = val
                elif key == 'size_y': size[1] = val
                elif key == 'size_z': size[2] = val
        return center, size

if __name__ == "__main__":
    DockingExecutor().run_docking()
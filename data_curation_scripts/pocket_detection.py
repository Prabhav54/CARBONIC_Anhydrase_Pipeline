import os
import sys
import pandas as pd
import numpy as np
from Bio import PDB
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class PocketConfig:
    structures_path: str = os.path.join('artifacts', 'structures')
    output_path: str = os.path.join('artifacts', 'data', 'pocket_metadata.csv')
    # Standard grid box size for Carbonic Anhydrase (in Angstroms)
    box_size_x: float = 20.0
    box_size_y: float = 20.0
    box_size_z: float = 20.0

class PocketDetector:
    def __init__(self):
        self.config = PocketConfig()

    def find_pocket_center(self, pdb_path):
        """
        Strategy:
        1. Look for the Zinc atom (ZN) which anchors the active site in CA enzymes.
        2. If no Zinc, look for a bound ligand (HETATM).
        3. Fallback: Center of Mass of the protein.
        """
        try:
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure('target', pdb_path)
            
            zinc_coords = []
            ligand_coords = []
            all_atom_coords = []

            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Strategy 1: Find ZINC (The perfect anchor)
                        if residue.get_resname() == 'ZN':
                            for atom in residue:
                                zinc_coords.append(atom.get_coord())
                        
                        # Strategy 2: Collect other Heteroatoms (Ligands)
                        # Ignoring Water (HOH)
                        if residue.id[0].startswith('H_') and residue.get_resname() != 'HOH':
                            for atom in residue:
                                ligand_coords.append(atom.get_coord())
                                
                        # Strategy 3: Collect all protein atoms (Fallback)
                        if residue.id[0] == ' ': # Standard amino acid
                            for atom in residue:
                                all_atom_coords.append(atom.get_coord())

            # DECISION LOGIC
            if zinc_coords:
                # Best Case: Center on the Zinc
                center = np.mean(zinc_coords, axis=0)
                method = "Zinc_Anchor"
            elif ligand_coords:
                # Good Case: Center on the existing drug
                center = np.mean(ligand_coords, axis=0)
                method = "Ligand_Centroid"
            else:
                # Fallback: Center of the whole protein
                center = np.mean(all_atom_coords, axis=0)
                method = "Protein_Center_Mass"

            return center, method

        except Exception as e:
            logging.error(f"Error parsing {pdb_path}: {e}")
            return None, "Failed"

    def detect_all_pockets(self):
        logging.info("Step 3: Detecting Binding Pockets (Grid Box Coordinates)...")
        try:
            if not os.path.exists(self.config.structures_path):
                raise FileNotFoundError("Structures folder not found!")

            pocket_data = []
            
            # List all PDB files
            pdb_files = [f for f in os.listdir(self.config.structures_path) if f.endswith('.pdb')]
            
            for pdb_file in pdb_files:
                pdb_id = pdb_file.split('.')[0]
                full_path = os.path.join(self.config.structures_path, pdb_file)
                
                center, method = self.find_pocket_center(full_path)
                
                if center is not None:
                    pocket_data.append({
                        'PDB_ID': pdb_id,
                        'Center_X': round(float(center[0]), 3),
                        'Center_Y': round(float(center[1]), 3),
                        'Center_Z': round(float(center[2]), 3),
                        'Size_X': self.config.box_size_x,
                        'Size_Y': self.config.box_size_y,
                        'Size_Z': self.config.box_size_z,
                        'Method': method
                    })
                    logging.info(f"  > {pdb_id}: Found pocket using {method}")

            # Save to CSV
            df = pd.DataFrame(pocket_data)
            df.to_csv(self.config.output_path, index=False)
            logging.info(f"SUCCESS: Pocket metadata saved to {self.config.output_path}")
            
            return self.config.output_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    PocketDetector().detect_all_pockets()
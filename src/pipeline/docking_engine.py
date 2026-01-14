import os
import requests
import subprocess
from vina import Vina # type: ignore
from meeko import MoleculePreparation # type: ignore
from rdkit import Chem
from rdkit.Chem import AllChem

class DockingEngine:
    def __init__(self, pdb_id, output_dir="docking_results"):
        self.pdb_id = pdb_id
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def setup_receptor(self):
        """Downloads PDB, cleans it (keeps Protein + Zinc), and converts to PDBQT."""
        raw_pdb = f"{self.output_dir}/{self.pdb_id}.pdb"
        clean_pdb = f"{self.output_dir}/{self.pdb_id}_clean.pdb"
        receptor_pdbqt = f"{self.output_dir}/{self.pdb_id}.pdbqt"

        # 1. Download if missing
        if not os.path.exists(raw_pdb):
            print(f"⬇️ Downloading {self.pdb_id}...")
            url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
            response = requests.get(url)
            with open(raw_pdb, "w") as f:
                f.write(response.text)

        # 2. Clean (Keep ATOM and ZN only, remove Water)
        print("🧹 Cleaning Receptor...")
        with open(raw_pdb, "r") as infile, open(clean_pdb, "w") as outfile:
            for line in infile:
                if line.startswith("ATOM") or (line.startswith("HETATM") and "ZN" in line):
                    outfile.write(line)

        # 3. Convert to PDBQT (using OpenBabel system command)
        # Note: OpenBabel must be installed in the environment (apt-get install openbabel)
        if not os.path.exists(receptor_pdbqt):
            print("⚙️ Converting Receptor to PDBQT...")
            subprocess.run(f"obabel {clean_pdb} -O {receptor_pdbqt} -xr -xn -xp --partialcharge gasteiger", shell=True)
        
        return receptor_pdbqt, clean_pdb

    def find_zinc_center(self, pdb_file):
        """Finds the XYZ coordinates of the Zinc atom (Active Site)."""
        coords = [0.0, 0.0, 0.0]
        found = False
        with open(pdb_file, 'r') as f:
            for line in f:
                if "ZN" in line and line.startswith("HETATM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords = [x, y, z]
                    found = True
                    break
        if not found:
            print("⚠️ Warning: No Zinc found. Using default center (0,0,0).")
        return coords

    def _create_complex(self, protein_pdb, ligand_pdb, output_file):
        """Merges Protein + Ligand into one PDB file for MD Simulation."""
        with open(protein_pdb, 'r') as p:
            protein_lines = [line for line in p if line.startswith("ATOM") or (line.startswith("HETATM") and "ZN" in line)]
            
        with open(ligand_pdb, 'r') as l:
            ligand_lines = []
            for line in l:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # Change Chain ID to 'L' and Record Name to HETATM
                    new_line = line[:21] + "L" + line[22:].replace("ATOM  ", "HETATM")
                    ligand_lines.append(new_line)

        with open(output_file, 'w') as out:
            out.writelines(protein_lines)
            out.writelines(ligand_lines)
            out.write("END")

    def dock_molecule(self, smiles, mol_id):
        """Runs the full docking process for a single molecule."""
        print(f"🚀 Docking {mol_id}...")
        
        try:
            # 1. Setup Vina
            receptor_pdbqt, clean_pdb = self.setup_receptor()
            center = self.find_zinc_center(clean_pdb)
            
            v = Vina(sf_name='vina')
            v.set_receptor(receptor_pdbqt)

            # 2. Prepare Ligand (SMILES -> 3D PDBQT)
            mol = Chem.MolFromSmiles(smiles)
            if not mol: raise ValueError("Invalid SMILES")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            ligand_pdbqt = preparator.write_pdbqt_string()

            # 3. Run Docking
            v.set_ligand_from_string(ligand_pdbqt)
            v.compute_vina_maps(center=center, box_size=[20, 20, 20])
            v.dock(exhaustiveness=8, n_poses=1)
            score = v.energies(n_poses=1)[0][0]

            # 4. Save Outputs
            base_path = f"{self.output_dir}/{mol_id}"
            
            # A. Ligand PDBQT & PDB
            v.write_poses(f"{base_path}_ligand.pdbqt", n_poses=1, overwrite=True)
            subprocess.run(f"obabel {base_path}_ligand.pdbqt -O {base_path}_ligand.pdb", shell=True)

            # B. Complex PDB (For MD)
            self._create_complex(clean_pdb, f"{base_path}_ligand.pdb", f"{base_path}_complex.pdb")

            print(f"   ✅ Score: {score} kcal/mol")
            return {
                "score": score,
                "ligand_pdb": f"{base_path}_ligand.pdb",
                "complex_pdb": f"{base_path}_complex.pdb"
            }

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return None
import os
import requests
import subprocess
from vina import Vina # type: ignore
from meeko import MoleculePreparation, PDBQTWriterLegacy # type: ignore
from rdkit import Chem
from rdkit.Chem import AllChem

class DockingEngine:
    def __init__(self, pdb_id, output_dir="docking_output"):
        self.pdb_id = pdb_id
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def setup_receptor(self):
        """Downloads PDB and converts to PDBQT using safe OpenBabel settings."""
        raw_pdb = f"{self.output_dir}/{self.pdb_id}.pdb"
        receptor_pdbqt = f"{self.output_dir}/{self.pdb_id}.pdbqt"

        # 1. Download PDB
        if not os.path.exists(raw_pdb):
            print(f"⬇️ Downloading {self.pdb_id}...")
            url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
            response = requests.get(url)
            with open(raw_pdb, "w") as f:
                f.write(response.text)

        # 2. Convert directly using OpenBabel (skipping manual cleaning which causes errors)
        # -xr: Output rigid molecule (good for receptors)
        # -h: Add hydrogens
        # --partialcharge gasteiger: Calculate charges
        if not os.path.exists(receptor_pdbqt):
            print("⚙️ Converting Receptor to PDBQT...")
            # We filter HOH (water) inside the command using grep if on Linux, or let obabel handle it
            cmd = f"grep -v 'HOH' {raw_pdb} > {self.output_dir}/temp_clean.pdb && obabel {self.output_dir}/temp_clean.pdb -O {receptor_pdbqt} -xr -h --partialcharge gasteiger"
            
            # Run command safely
            try:
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError:
                print("❌ OpenBabel Failed! Using fallback conversion...")
                subprocess.run(f"obabel {raw_pdb} -O {receptor_pdbqt} -xr -h --partialcharge gasteiger", shell=True)

        return receptor_pdbqt

    def find_zinc_center(self, pdb_file):
        """Approximates the active site center based on Zinc location."""
        # If the PDBQT exists, we can read it directly or read original PDB
        search_file = pdb_file if os.path.exists(pdb_file) else f"{self.output_dir}/{self.pdb_id}.pdb"
        
        coords = [0.0, 0.0, 0.0]
        found = False
        with open(search_file, 'r') as f:
            for line in f:
                if "ZN" in line and ("HETATM" in line or "ATOM" in line):
                    # PDB column format ensures these positions are correct
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords = [x, y, z]
                        found = True
                        break
                    except ValueError:
                        continue
        
        if not found:
            print("⚠️ Warning: No Zinc found. Using (0,0,0).")
        return coords

    def dock_molecule(self, smiles, mol_id):
        print(f"🚀 Docking {mol_id}...")
        try:
            # 1. Setup Receptor
            receptor_pdbqt = self.setup_receptor()
            center = self.find_zinc_center(receptor_pdbqt)

            # 2. Setup Ligand (Meeko v0.5+ Syntax Fix)
            mol = Chem.MolFromSmiles(smiles)
            if not mol: raise ValueError("Invalid SMILES")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            # New Meeko API
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)
            ligand_pdbqt = PDBQTWriterLegacy.write_string(mol_setup[0])

            # 3. Run Vina
            v = Vina(sf_name='vina')
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_string(ligand_pdbqt)
            
            # Search Box (20x20x20 Angstroms around Zinc)
            v.compute_vina_maps(center=center, box_size=[20, 20, 20])
            v.dock(exhaustiveness=8, n_poses=1)
            
            # 4. Get Score
            score = v.energies(n_poses=1)[0][0]
            
            # 5. Save Output
            out_base = f"{self.output_dir}/{mol_id}"
            v.write_poses(f"{out_base}_ligand.pdbqt", n_poses=1, overwrite=True)
            
            print(f"   ✅ Score: {score} kcal/mol")
            return {"score": score, "ligand_pdbqt": f"{out_base}_ligand.pdbqt"}

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return None ````````````````
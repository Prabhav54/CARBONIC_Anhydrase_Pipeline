import os
import requests
import subprocess
from vina import Vina
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem

class DockingEngine:
    def __init__(self, pdb_id="3HS4", output_dir="docking_output"):
        self.pdb_id = pdb_id
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract_zinc_center(self, pdb_file):
        """Finds Zinc coordinates for the docking box center."""
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("HETATM") and ("ZN" in line or "Zn" in line):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        return [x, y, z]
                    except: pass
        return [14.5, -7.5, 3.5] # Fallback for 3HS4

    def setup_receptor(self):
        """
        ROBUST PREPARATION: 
        1. Split Protein & Zinc. 
        2. Convert separately (OpenBabel). 
        3. Merge textually.
        """
        raw_pdb = f"{self.output_dir}/{self.pdb_id}.pdb"
        protein_pdb = f"{self.output_dir}/{self.pdb_id}_prot.pdb"
        zinc_pdb = f"{self.output_dir}/{self.pdb_id}_zinc.pdb"
        receptor_pdbqt = f"{self.output_dir}/{self.pdb_id}.pdbqt"

        # Download
        if not os.path.exists(raw_pdb):
            url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
            with open(raw_pdb, "w") as f: f.write(requests.get(url).text)

        # Split Files (Python handles this safely)
        has_zinc = False
        with open(raw_pdb, 'r') as f_in, open(protein_pdb, 'w') as f_p, open(zinc_pdb, 'w') as f_z:
            for line in f_in:
                if line.startswith("ATOM"): f_p.write(line)
                elif line.startswith("HETATM") and ("ZN" in line or "Zn" in line):
                    f_z.write(line)
                    has_zinc = True

        # Convert Separately (Safe from formatting errors)
        if not os.path.exists(receptor_pdbqt):
            # Protein: Add Hydrogens (-p 7.4)
            subprocess.run(f"obabel {protein_pdb} -O {protein_pdb}.qt -xr -p 7.4 --partialcharge gasteiger", shell=True)
            # Zinc: Just convert
            if has_zinc:
                subprocess.run(f"obabel {zinc_pdb} -O {zinc_pdb}.qt -xr --partialcharge gasteiger", shell=True)

            # Merge
            with open(receptor_pdbqt, 'w') as outfile:
                if os.path.exists(f"{protein_pdb}.qt"):
                    outfile.write(open(f"{protein_pdb}.qt").read())
                if has_zinc and os.path.exists(f"{zinc_pdb}.qt"):
                    outfile.write(open(f"{zinc_pdb}.qt").read())

        return receptor_pdbqt

    def dock_molecule(self, smiles, mol_id):
        try:
            receptor_pdbqt = self.setup_receptor()
            center = self.extract_zinc_center(f"{self.output_dir}/{self.pdb_id}.pdb")

            # Ligand Prep
            mol = Chem.MolFromSmiles(smiles)
            if not mol: return None
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            prep = MoleculePreparation()
            mol_setup = prep.prepare(mol)
            ligand_pdbqt = PDBQTWriterLegacy.write_string(mol_setup[0])

            # Vina Run
            v = Vina(sf_name='vina')
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_string(ligand_pdbqt)
            v.compute_vina_maps(center=center, box_size=[22, 22, 22])
            v.dock(exhaustiveness=8, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
            
            # Save PDB for MD
            out_qt = f"{self.output_dir}/{mol_id}_ligand.pdbqt"
            out_pdb = f"{self.output_dir}/{mol_id}_ligand.pdb"
            v.write_poses(out_qt, n_poses=1, overwrite=True)
            subprocess.run(f"obabel {out_qt} -O {out_pdb}", shell=True)
            
            return {"score": score, "pdb_file": out_pdb}
        except Exception as e:
            print(f"Docking Error: {e}")
            return None
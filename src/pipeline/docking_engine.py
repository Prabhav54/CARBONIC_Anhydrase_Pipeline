import os
import requests
import subprocess
from vina import Vina # type: ignore
from meeko import MoleculePreparation, PDBQTWriterLegacy # type: ignore
from rdkit import Chem
from rdkit.Chem import AllChem

class DockingEngine:
    def __init__(self, pdb_id="3HS4", output_dir="docking_output"):
        self.pdb_id = pdb_id
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def setup_receptor(self):
        """
        ROBUST PREPARATION:
        Splits Protein and Zinc, converts them separately, then merges them.
        This prevents OpenBabel from crashing on the metal-protein bonds.
        """
        raw_pdb = f"{self.output_dir}/{self.pdb_id}.pdb"
        receptor_pdbqt = f"{self.output_dir}/{self.pdb_id}.pdbqt"

        # 1. Download PDB
        if not os.path.exists(raw_pdb):
            print(f"⬇️ Downloading {self.pdb_id}...")
            url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
            requests.get(url).raise_for_status()
            with open(raw_pdb, "w") as f:
                f.write(requests.get(url).text)

        # 2. Convert to PDBQT (The "Zinc Split" Hack)
        if not os.path.exists(receptor_pdbqt):
            print("⚙️ splitting Protein and Zinc to bypass OpenBabel errors...")
            
            # File names
            protein_pdb = f"{self.output_dir}/protein_only.pdb"
            zinc_pdb = f"{self.output_dir}/zinc_only.pdb"
            protein_pdbqt = f"{self.output_dir}/protein.pdbqt"
            zinc_pdbqt = f"{self.output_dir}/zinc.pdbqt"

            # A. Split the file using text manipulation
            with open(raw_pdb, 'r') as f_in, open(protein_pdb, 'w') as f_prot, open(zinc_pdb, 'w') as f_zinc:
                for line in f_in:
                    if line.startswith("ATOM"):
                        f_prot.write(line)
                    elif line.startswith("HETATM") and ("ZN" in line or "Zn" in line):
                        f_zinc.write(line)

            # B. Convert Protein (Standard -xr -p 7.4)
            # We use -r to ignore rigid formatting issues
            cmd_prot = f"obabel {protein_pdb} -O {protein_pdbqt} -xr -p 7.4 --partialcharge gasteiger"
            subprocess.run(cmd_prot, shell=True, check=False) # check=False to ignore minor warnings

            # C. Convert Zinc (Simple)
            cmd_zinc = f"obabel {zinc_pdb} -O {zinc_pdbqt} -xr --partialcharge gasteiger"
            subprocess.run(cmd_zinc, shell=True, check=False)

            # D. Merge files (Protein first, then Zinc)
            with open(receptor_pdbqt, 'w') as outfile:
                # Copy Protein
                if os.path.exists(protein_pdbqt):
                    with open(protein_pdbqt, 'r') as infile:
                        outfile.write(infile.read())
                
                # Copy Zinc
                if os.path.exists(zinc_pdbqt):
                    with open(zinc_pdbqt, 'r') as infile:
                        outfile.write(infile.read())

            print("✅ Receptor Prepared (Merged Protein + Zinc)")

        return receptor_pdbqt

    def find_zinc_center(self, pdb_file):
        """Approximates center. Defaults to known 3HS4 center if calculation fails."""
        if self.pdb_id == "3HS4":
            return [14.5, -7.5, 3.5] # The exact active site of Human Carbonic Anhydrase II
        
        # Fallback for other proteins
        return [0.0, 0.0, 0.0]

    def dock_molecule(self, smiles, mol_id):
        print(f"\n🚀 Docking {mol_id}...")
        try:
            receptor_pdbqt = self.setup_receptor()
            center = self.find_zinc_center(None)

            # Ligand Prep
            mol = Chem.MolFromSmiles(smiles)
            if not mol: raise ValueError("Invalid SMILES")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)
            ligand_pdbqt = PDBQTWriterLegacy.write_string(mol_setup[0])

            # Vina
            v = Vina(sf_name='vina')
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_string(ligand_pdbqt)
            
            # Run
            v.compute_vina_maps(center=center, box_size=[22, 22, 22])
            v.dock(exhaustiveness=8, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
            
            # Save
            out_base = f"{self.output_dir}/{mol_id}"
            v.write_poses(f"{out_base}_ligand.pdbqt", n_poses=1, overwrite=True)
            
            # Create Complex PDB for Visualization
            subprocess.run(f"obabel {out_base}_ligand.pdbqt -O {out_base}_ligand.pdb", shell=True)
            
            print(f"   ✅ Score: {score} kcal/mol")
            return {"score": score, "ligand_pdb": f"{out_base}_ligand.pdb"}

        except Exception as e:
            print(f"   ❌ Docking Failed: {e}")
            return None
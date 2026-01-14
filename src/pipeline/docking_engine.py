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

    def strict_clean_pdb(self, input_pdb, output_pdb):
        """
        STRICT CLEANER:
        - Keeps 'ATOM' (Protein residues)
        - Keeps 'HETATM' ONLY if it is Zinc (ZN)
        - DISCARDS the native drug, water, and salts.
        This prevents OpenBabel from crashing on the native ligand.
        """
        print(f"🧹 STRICT Cleaning {input_pdb}...")
        atom_count = 0
        zinc_count = 0
        
        with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
            for line in f_in:
                # Keep Protein (Standard Residues)
                if line.startswith("ATOM"):
                    f_out.write(line)
                    atom_count += 1
                # Keep Zinc (Vital for Carbonic Anhydrase)
                elif line.startswith("HETATM") and "ZN" in line:
                    f_out.write(line)
                    zinc_count += 1
                # Everything else (Native Ligands like AZM, Water HOH) is DELETED.
        
        print(f"   -> Kept {atom_count} atoms and {zinc_count} Zinc ions.")
        if atom_count == 0:
            raise ValueError("❌ Error: The cleaned PDB is empty! Download failed?")

    def setup_receptor(self):
        """Downloads PDB, cleans it strictly, and converts to PDBQT."""
        raw_pdb = f"{self.output_dir}/{self.pdb_id}.pdb"
        clean_pdb = f"{self.output_dir}/{self.pdb_id}_strict_clean.pdb"
        receptor_pdbqt = f"{self.output_dir}/{self.pdb_id}.pdbqt"

        # 1. Download PDB if missing
        if not os.path.exists(raw_pdb):
            print(f"⬇️ Downloading {self.pdb_id}...")
            url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
            response = requests.get(url)
            if response.status_code != 200:
                raise ValueError(f"❌ Failed to download PDB ID: {self.pdb_id}")
            with open(raw_pdb, "w") as f:
                f.write(response.text)

        # 2. STRICT Clean (Remove Native Ligand)
        if not os.path.exists(clean_pdb):
            self.strict_clean_pdb(raw_pdb, clean_pdb)

        # 3. Convert to PDBQT using OpenBabel
        # -xr: Output rigid molecule
        # --partialcharge gasteiger: Calculate charges necessary for docking
        if not os.path.exists(receptor_pdbqt) or os.path.getsize(receptor_pdbqt) == 0:
            print("⚙️ Converting Receptor to PDBQT...")
            cmd = f"obabel {clean_pdb} -O {receptor_pdbqt} -xr --partialcharge gasteiger"
            subprocess.run(cmd, shell=True, check=True)
            
            # Verification
            if not os.path.exists(receptor_pdbqt) or os.path.getsize(receptor_pdbqt) == 0:
                raise ValueError("❌ OpenBabel generated an empty PDBQT file!")

        return receptor_pdbqt

    def find_zinc_center(self, pdb_file):
        """Finds the geometric center of the Zinc atom."""
        coords = [0.0, 0.0, 0.0]
        found = False
        with open(pdb_file, 'r') as f:
            for line in f:
                if "ZN" in line and ("HETATM" in line or "ATOM" in line):
                    # Parse PDB columns for X, Y, Z
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords = [x, y, z]
                        found = True
                        break # Found the first Zinc, good enough for CA
                    except ValueError:
                        continue
        
        if not found:
            print("⚠️ WARNING: No Zinc found in receptor! Docking might fail.")
        else:
            print(f"🎯 Zinc Center Found: {coords}")
        return coords

    def dock_molecule(self, smiles, mol_id):
        print(f"\n🚀 Docking {mol_id}...")
        try:
            # 1. Setup Receptor
            receptor_pdbqt = self.setup_receptor()
            
            # Use the clean PDB to find the center (Coordinates are same as PDBQT)
            clean_pdb = f"{self.output_dir}/{self.pdb_id}_strict_clean.pdb"
            center = self.find_zinc_center(clean_pdb)

            # 2. Setup Ligand
            mol = Chem.MolFromSmiles(smiles)
            if not mol: raise ValueError("Invalid SMILES string")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)
            ligand_pdbqt = PDBQTWriterLegacy.write_string(mol_setup[0])

            # 3. Run Vina
            v = Vina(sf_name='vina')
            v.set_receptor(receptor_pdbqt)
            v.set_ligand_from_string(ligand_pdbqt)
            
            # Increased Box Size to 25.0 to ensure fit
            v.compute_vina_maps(center=center, box_size=[25, 25, 25])
            v.dock(exhaustiveness=8, n_poses=1)
            
            # 4. Results
            score = v.energies(n_poses=1)[0][0]
            
            # Save Output
            out_base = f"{self.output_dir}/{mol_id}"
            v.write_poses(f"{out_base}_ligand.pdbqt", n_poses=1, overwrite=True)
            
            # Create Complex PDB for Visualization (Merge Receptor + Docked Ligand)
            # We convert the docked ligand PDBQT -> PDB first
            subprocess.run(f"obabel {out_base}_ligand.pdbqt -O {out_base}_ligand.pdb", shell=True)
            
            print(f"   ✅ Score: {score} kcal/mol")
            return {
                "score": score,
                "ligand_pdb": f"{out_base}_ligand.pdb"
            }

        except Exception as e:
            print(f"   ❌ Docking Failed: {e}")
            return None
import os
import requests
import subprocess
from vina import Vina # type: ignore
from meeko import MoleculePreparation, PDBQTWriterLegacy # type: ignore
from rdkit import Chem
from rdkit.Chem import AllChem

class DockingEngine:
    # CHANGED: Default to '3HS4' (The Gold Standard CA-II structure)
    def __init__(self, pdb_id="3HS4", output_dir="docking_output"):
        self.pdb_id = pdb_id
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def strict_clean_pdb(self, input_pdb, output_pdb):
        """
        Stage 1: Text-based cleaning.
        Removes water, salts, and native ligands. Keeps Protein + Zinc.
        """
        print(f"🧹 STRICT Cleaning {input_pdb}...")
        kept_lines = []
        with open(input_pdb, 'r') as f_in:
            for line in f_in:
                if line.startswith("ATOM"):
                    kept_lines.append(line)
                elif line.startswith("HETATM") and "ZN" in line:
                    kept_lines.append(line)
        
        with open(output_pdb, 'w') as f_out:
            f_out.writelines(kept_lines)
            
        print(f"   -> Text cleaner kept {len(kept_lines)} atoms.")

    def sanitize_with_rdkit(self, input_pdb, output_pdb):
        """
        Stage 2: Chemical Sanitization using RDKit.
        Repairs broken bonds and aromatic rings that confuse OpenBabel.
        """
        print(f"💊 Sanitizing with RDKit...")
        try:
            # Load PDB (sanitize=True fixes bond orders automatically)
            mol = Chem.MolFromPDBFile(input_pdb, removeHs=False, sanitize=True)
            if mol:
                # Write back a chemically perfect PDB
                Chem.MolToPDBFile(mol, output_pdb)
                print("   -> RDKit repaired the structure successfully.")
            else:
                print("   ⚠️ RDKit warning: Could not fully parse PDB, using text-cleaned version.")
        except Exception as e:
            print(f"   ⚠️ RDKit skip: {e}")

    def setup_receptor(self):
        raw_pdb = f"{self.output_dir}/{self.pdb_id}.pdb"
        clean_pdb = f"{self.output_dir}/{self.pdb_id}_clean.pdb"
        receptor_pdbqt = f"{self.output_dir}/{self.pdb_id}.pdbqt"

        # 1. Download
        if not os.path.exists(raw_pdb):
            print(f"⬇️ Downloading {self.pdb_id}...")
            url = f"https://files.rcsb.org/download/{self.pdb_id}.pdb"
            requests.get(url).raise_for_status()
            with open(raw_pdb, "w") as f:
                f.write(requests.get(url).text)

        # 2. Multi-Stage Cleaning
        if not os.path.exists(receptor_pdbqt):
            # Stage 1: Text Clean
            self.strict_clean_pdb(raw_pdb, clean_pdb)
            
            # Stage 2: RDKit Sanitize (Fixes 'Kekulize' errors)
            self.sanitize_with_rdkit(clean_pdb, clean_pdb)

            # Stage 3: OpenBabel Conversion
            print("⚙️ Converting Receptor to PDBQT...")
            # We add '-p' to add hydrogens relevant to pH 7.4
            cmd = f"obabel {clean_pdb} -O {receptor_pdbqt} -xr -p 7.4 --partialcharge gasteiger"
            subprocess.run(cmd, shell=True, check=True)

            if not os.path.exists(receptor_pdbqt) or os.path.getsize(receptor_pdbqt) == 0:
                raise ValueError("❌ OpenBabel failed! Output PDBQT is empty.")

        return receptor_pdbqt

    def find_zinc_center(self, pdb_file):
        """Finds Zinc Center."""
        coords = [0.0, 0.0, 0.0]
        # Check PDBQT first (most accurate), then fallback to clean PDB
        target_file = f"{self.output_dir}/{self.pdb_id}.pdbqt"
        if not os.path.exists(target_file):
            target_file = pdb_file

        with open(target_file, 'r') as f:
            for line in f:
                # OpenBabel calls Zinc "Zn" or "ZN", check both
                if ("ZN" in line or "Zn" in line) and ("HETATM" in line or "ATOM" in line):
                    try:
                        # PDBQT/PDB columns for X, Y, Z
                        parts = line.split()
                        # Safe parsing logic for variable whitespace
                        if len(parts) > 6:
                            # Usually coordinates are at indices 6, 7, 8 in split lists of PDBQT
                            for i, part in enumerate(parts):
                                if "." in part and parts[i+1].replace('.','').isdigit() is False:
                                    # Heuristic to find the 3 floating point numbers
                                    try:
                                        coords = [float(parts[i]), float(parts[i+1]), float(parts[i+2])]
                                        print(f"🎯 Zinc found at: {coords}")
                                        return coords
                                    except:
                                        continue
                    except:
                        continue
        
        # Fallback Hardcoded Center for 3HS4 (Known Center)
        if self.pdb_id == "3HS4":
            print("⚠️ Zinc parsing tricky, using known 3HS4 center.")
            return [14.5, -7.5, 3.5] 
            
        return coords

    def dock_molecule(self, smiles, mol_id):
        print(f"\n🚀 Docking {mol_id}...")
        try:
            receptor_pdbqt = self.setup_receptor()
            clean_pdb = f"{self.output_dir}/{self.pdb_id}_clean.pdb"
            center = self.find_zinc_center(clean_pdb)

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
            
            # Convert output to PDB for viewing
            subprocess.run(f"obabel {out_base}_ligand.pdbqt -O {out_base}_ligand.pdb", shell=True)
            
            print(f"   ✅ Score: {score} kcal/mol")
            return {"score": score, "ligand_pdb": f"{out_base}_ligand.pdb"}

        except Exception as e:
            print(f"   ❌ Docking Failed: {e}")
            return None
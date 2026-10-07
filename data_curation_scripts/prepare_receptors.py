"""Builds the receptor PDBQT files the web app docks against.

For each supported target this keeps the protein chain whose catalytic zinc
lies closest to the docking box centre (pocket_metadata.csv), drops waters and
co-crystallised ligands, adds polar hydrogens at pH 7.4 and Gasteiger charges
with Open Babel, then re-attaches the zinc. The zinc must stay: sulfonamide
inhibitors bind CA by coordinating it. It is typed 'ZN' with zero charge (the
AutoDock4Zn convention), and added after charging because Open Babel has no
Gasteiger parameters for zinc and silently writes nothing if it is present.

Output: artifacts/receptors/<pdb_id>.pdbqt + boxes.json.
Run once, offline, from the project root (needs the `obabel` CLI, e.g. the
conda-forge `openbabel` package):
    python -m data_curation_scripts.prepare_receptors
"""
import csv
import json
import math
import os
import subprocess
import tempfile

from src.pipeline.targets import TARGETS

STRUCTURES_DIR = os.path.join("artifacts", "structures")
POCKETS_CSV = os.path.join("artifacts", "data", "pocket_metadata.csv")
OUT_DIR = os.path.join("artifacts", "receptors")


def _coords(line):
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def _is_zinc(line):
    return line.startswith("HETATM") and line[17:20] == " ZN"


def nearest_zinc(pdb_lines, center):
    zincs = [l for l in pdb_lines if _is_zinc(l)]
    if not zincs:
        raise ValueError("no catalytic zinc found")
    return min(zincs, key=lambda l: math.dist(_coords(l), center))


def protein_chain(pdb_lines, chain):
    """Protein atoms of one chain; no waters, ligands, ions or alt locations."""
    kept = [l for l in pdb_lines if l.startswith("ATOM  ") and l[21] == chain and l[16] in (" ", "A")]
    return "".join(kept) + "END\n"


def zinc_pdbqt_line(pdb_line):
    return f"{pdb_line[:54]}  1.00  0.00    {0.0:+6.3f} ZN\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(POCKETS_CSV, newline="") as f:
        pockets = {row["PDB_ID"]: row for row in csv.DictReader(f)}
    boxes = {}

    for pdb_id in TARGETS:
        row = pockets[pdb_id.lower()]
        center = [float(row[k]) for k in ("Center_X", "Center_Y", "Center_Z")]
        size = [float(row[k]) for k in ("Size_X", "Size_Y", "Size_Z")]

        with open(os.path.join(STRUCTURES_DIR, f"{pdb_id.lower()}.pdb")) as f:
            lines = f.readlines()
        zinc = nearest_zinc(lines, center)
        chain = zinc[21]

        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "protein.pdb"), os.path.join(tmp, "protein.pdbqt")
            with open(src, "w") as f:
                f.write(protein_chain(lines, chain))
            # -xr: rigid receptor (no ROOT/BRANCH torsion tree)
            subprocess.run(["obabel", src, "-O", dst, "-xr", "-p", "7.4", "--partialcharge", "gasteiger"],
                           check=True, capture_output=True, text=True)
            with open(dst) as f:
                atoms = [l for l in f if l.startswith(("ATOM", "HETATM"))]

        if not atoms or not any(float(l[70:76]) for l in atoms):
            raise RuntimeError(f"{pdb_id}: Open Babel produced no charged atoms")
        atoms.append(zinc_pdbqt_line(zinc))
        out_path = os.path.join(OUT_DIR, f"{pdb_id}.pdbqt")
        with open(out_path, "w") as f:
            f.writelines(atoms)

        boxes[pdb_id] = {"chain": chain, "center": center, "size": size}
        print(f"{pdb_id}: chain {chain}, {len(atoms)} atoms -> {out_path}")

    with open(os.path.join(OUT_DIR, "boxes.json"), "w") as f:
        json.dump(boxes, f, indent=2)


if __name__ == "__main__":
    main()

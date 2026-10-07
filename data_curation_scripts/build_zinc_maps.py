"""Precomputes AutoDock4Zn affinity maps for every supported target.

Vina's default scoring function has no metal-coordination term, so it cannot
reproduce how sulfonamides bind the CA active-site zinc (re-docking the crystal
acetazolamide into 3HS4 lands >5 A away). AutoDock4Zn fixes this with a
tetrahedral pseudo-atom (TZ) next to the zinc and zinc-specific pair potentials.

Maps can only be generated with AutoGrid 4.2.7 (not available on Vercel), so
they are built here, offline, and shipped gzipped in artifacts/receptors/zn_maps/.

Requires `autogrid4` 4.2.7 on PATH (conda-forge: `autogrid`). Run after
prepare_receptors.py, from the project root:
    python -m data_curation_scripts.build_zinc_maps
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

RECEPTOR_DIR = os.path.join("artifacts", "receptors")
MAPS_DIR = os.path.join(RECEPTOR_DIR, "zn_maps")
AD4ZN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ad4zn")

SPACING = 0.375
# Every type src/pipeline/ligand_prep.py can emit.
LIGAND_TYPES = ["A", "C", "N", "NA", "OA", "S", "SA", "HD", "F", "Cl", "Br", "I", "P"]
# AutoDock4Zn pair potentials (from the AutoDock-Vina zinc docking tutorial).
ZN_PAIR_POTENTIALS = """\
nbp_r_eps 0.25 23.2135 12 6 NA TZ
nbp_r_eps 2.1   3.8453 12 6 OA Zn
nbp_r_eps 2.25  7.5914 12 6 SA Zn
nbp_r_eps 1.0   0.0    12 6 HD Zn
nbp_r_eps 2.0   0.0060 12 6 NA Zn
nbp_r_eps 2.0   0.2966 12 6  N Zn
"""


def _grid_points(size):
    n = int(round(size / SPACING))
    return n + (n % 2)  # AutoGrid requires an even number of points


def _receptor_types(pdbqt_path):
    with open(pdbqt_path) as f:
        return sorted({l[77:79].strip() for l in f if l.startswith(("ATOM", "HETATM"))})


def build(pdb_id, box, workdir):
    receptor = os.path.join(workdir, f"{pdb_id}_tz.pdbqt")
    subprocess.run([sys.executable, os.path.join(AD4ZN_DIR, "zinc_pseudo.py"),
                    "-r", os.path.join(RECEPTOR_DIR, f"{pdb_id}.pdbqt"), "-o", receptor],
                   check=True, capture_output=True, text=True)
    if "TZ" not in _receptor_types(receptor):
        raise RuntimeError(f"{pdb_id}: no tetrahedral zinc site found")

    prefix = f"{pdb_id}_tz"
    npts = " ".join(str(_grid_points(s)) for s in box["size"])
    center = " ".join(f"{c:.3f}" for c in box["center"])
    maps = "".join(f"map {prefix}.{t}.map\n" for t in LIGAND_TYPES)
    gpf = (
        f"npts {npts}\n"
        f"parameter_file AD4Zn.dat\n"
        f"gridfld {prefix}.maps.fld\n"
        f"spacing {SPACING}\n"
        f"receptor_types {' '.join(_receptor_types(receptor))}\n"
        f"ligand_types {' '.join(LIGAND_TYPES)}\n"
        f"receptor {prefix}.pdbqt\n"
        f"gridcenter {center}\n"
        f"smooth 0.5\n"
        f"{maps}"
        f"elecmap {prefix}.e.map\n"
        f"dsolvmap {prefix}.d.map\n"
        f"dielectric -0.1465\n"
        f"{ZN_PAIR_POTENTIALS}"
    )
    with open(os.path.join(workdir, f"{prefix}.gpf"), "w") as f:
        f.write(gpf)
    shutil.copy(os.path.join(AD4ZN_DIR, "AD4Zn.dat"), workdir)
    subprocess.run(["autogrid4", "-p", f"{prefix}.gpf", "-l", f"{prefix}.glg"],
                   cwd=workdir, check=True, capture_output=True, text=True)

    out_dir = os.path.join(MAPS_DIR, pdb_id)
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(workdir):
        if name.endswith(".map") or name.endswith(".fld") or name.endswith(".xyz"):
            with open(os.path.join(workdir, name), "rb") as src, gzip.open(os.path.join(out_dir, name + ".gz"), "wb") as dst:
                shutil.copyfileobj(src, dst)
    return out_dir


def main():
    with open(os.path.join(RECEPTOR_DIR, "boxes.json")) as f:
        boxes = json.load(f)
    for pdb_id, box in boxes.items():
        with tempfile.TemporaryDirectory() as workdir:
            print(f"{pdb_id}: maps -> {build(pdb_id, box, workdir)}")


if __name__ == "__main__":
    main()

"""AutoDock Vina docking against the prepared carbonic anhydrase receptors.

Scoring uses the AutoDock4Zn force field via precomputed affinity maps
(data_curation_scripts/build_zinc_maps.py). Vina's default scoring function has
no metal-coordination term, so it cannot reproduce how sulfonamides bind the CA
active-site zinc; AD4Zn adds that term.

Known limitation: AD4Zn recovers the crystal pose of acetazolamide in 3HS4 to
1.6 A with the expected N-Zn contact, but ranks it second rather than first.
Callers are therefore given several poses and the zinc distance of each, instead
of a single number presented as the answer. See tests/test_docking.py.
"""
import gzip
import os
import shutil
import tempfile
import threading
from functools import lru_cache

import numpy as np
from vina import Vina

from src.pipeline.ligand_prep import LigandPrepError, prepare_ligand
from src.pipeline.pose_io import pdbqt_to_pdb_block

RECEPTOR_DIR = os.path.join("artifacts", "receptors")
MAPS_DIR = os.path.join(RECEPTOR_DIR, "zn_maps")
# Exhaustiveness 8 is Vina's default, but it often returns a single pose for these
# ligands, and for carbonic anhydrase that pose is frequently the non-coordinating
# one. Searching harder surfaces the zinc-binding poses the UI needs to show, and
# still finishes in roughly 50 s of the 300 s serverless budget.
EXHAUSTIVENESS = int(os.environ.get("VINA_EXHAUSTIVENESS", "16"))
N_POSES = int(os.environ.get("VINA_N_POSES", "5"))
# A sulfonamide N closer than this to the zinc is making the canonical contact.
ZINC_CONTACT_CUTOFF = 2.5


def _zinc_position(pdb_id):
    with open(os.path.join(RECEPTOR_DIR, f"{pdb_id}.pdbqt")) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line[77:79] == "ZN":
                return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    raise RuntimeError(f"{pdb_id}: receptor has no zinc")


def _unpack_maps(pdb_id):
    """Decompresses this target's maps into a temp dir and returns the map prefix."""
    src = os.path.join(MAPS_DIR, pdb_id)
    dst = os.path.join(tempfile.gettempdir(), "ca_maps", pdb_id)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        out = os.path.join(dst, name[: -len(".gz")])
        if os.path.exists(out):
            continue
        # Decompress to a unique name and rename, so a concurrent request can
        # never read a partially written map.
        with gzip.open(os.path.join(src, name)) as s, \
                tempfile.NamedTemporaryFile(dir=dst, delete=False) as d:
            shutil.copyfileobj(s, d)
            staged = d.name
        os.replace(staged, out)
    return os.path.join(dst, f"{pdb_id}_tz")


def _pose_coords(pose_block, serial_to_idx):
    return {
        serial_to_idx[int(l[6:11])]: np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        for l in pose_block.splitlines()
        if l.startswith(("ATOM", "HETATM"))
    }


class DockingEngine:
    """One warm instance per target; maps are loaded once and reused."""

    def __init__(self, pdb_id):
        self.pdb_id = pdb_id
        self.zinc = _zinc_position(pdb_id)
        self._lock = threading.Lock()  # a Vina object cannot be shared across threads
        # One CPU and a fixed seed keep scores reproducible, which the re-docking
        # benchmark in tests/test_docking.py depends on. Vina parallelises across
        # cores by splitting the search, so results vary slightly with cpu > 1.
        self._vina = Vina(sf_name="ad4", cpu=1, seed=42, verbosity=0)
        self._vina.load_maps(_unpack_maps(pdb_id))

    def dock(self, smiles, exhaustiveness=EXHAUSTIVENESS, n_poses=N_POSES):
        """Returns {'poses': [...]} ranked by score, or {'error': reason}.

        Each pose carries its AD4Zn score, the shortest distance from the zinc
        to a ligand sulfonamide nitrogen, and whether that is a real contact.
        """
        try:
            ligand, serial_to_idx, sulfonamide_n = prepare_ligand(smiles)
        except LigandPrepError as e:
            return {"error": str(e)}

        with self._lock:
            self._vina.set_ligand_from_string(ligand)
            self._vina.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
            energies = self._vina.energies(n_poses=n_poses)
            blocks = [b for b in self._vina.poses(n_poses=n_poses).split("ENDMDL") if "ATOM" in b]

        poses = []
        for rank, block in enumerate(blocks[: len(energies)]):
            coords = _pose_coords(block, serial_to_idx)
            distances = [float(np.linalg.norm(coords[i] - self.zinc)) for i in sulfonamide_n if i in coords]
            nearest = round(min(distances), 2) if distances else None
            poses.append({
                "rank": rank + 1,
                "score": round(float(energies[rank][0]), 2),
                "zinc_distance": nearest,
                "zinc_contact": nearest is not None and nearest <= ZINC_CONTACT_CUTOFF,
                "pose_pdbqt": block.strip() + "\n",
                "pose_pdb": pdbqt_to_pdb_block(block),  # for 3D viewers
            })
        if not poses:
            return {"error": "docking produced no poses"}
        return {"poses": poses, "has_sulfonamide": bool(sulfonamide_n)}


@lru_cache(maxsize=None)
def get_engine(pdb_id):
    return DockingEngine(pdb_id)

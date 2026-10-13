"""AutoDock Vina docking against any carbonic anhydrase structure.

Scoring uses the AutoDock4Zn force field through grid maps built for whatever
structure was asked for (src/pipeline/zn_maps.py). Vina's own scoring function
has no metal-coordination term, so it cannot reproduce how sulfonamides bind the
active-site zinc; AD4Zn adds it.

What the caller gets back is several poses, each with its score, the shortest
distance from the zinc to a ligand sulfonamide nitrogen, and the residues it
touches -- not a single headline number. A pose that does not coordinate the
zinc is probably not the real binding mode however well it scores, and that is
only visible if the distance is shown.

Measured on 3HS4 (re-docking the crystal acetazolamide): the top-ranked pose is
1.75 A from the crystal structure with a 1.94 A N-Zn contact, and the closest of
the four poses is 1.71 A. See tests/test_docking.py.
"""
import math
import os
import tempfile
import threading
from functools import lru_cache

import numpy as np
from vina import Vina

from src.logger import logging
from src.pipeline import receptor_prep, zn_maps
from src.pipeline.ligand_prep import LigandPrepError, prepare_ligand
from src.pipeline.pose_io import pdbqt_to_pdb_block

WORK_ROOT = os.path.join(tempfile.gettempdir(), "ca_receptors")

# Exhaustiveness 8 is Vina's default, but it often returns a single pose for
# these ligands, and for carbonic anhydrase that pose is frequently the
# non-coordinating one. Searching harder surfaces the zinc-binding poses the UI
# needs to show, and still leaves the request inside the serverless budget.
EXHAUSTIVENESS = int(os.environ.get("VINA_EXHAUSTIVENESS", "16"))
N_POSES = int(os.environ.get("VINA_N_POSES", "5"))

# A sulfonamide N closer than this to the zinc is making the canonical contact.
ZINC_CONTACT_CUTOFF = 2.5
# Residues with an atom this close to a pose are reported as contacts.
CONTACT_CUTOFF = 4.0

# One engine holds a full map set in memory, so only a few are kept warm. This
# matches zn_maps.MAX_CACHED_TARGETS, which bounds the same thing on disk.
MAX_WARM_ENGINES = zn_maps.MAX_CACHED_TARGETS


def _pose_coords(pose_block, serial_to_idx):
    return {
        serial_to_idx[int(l[6:11])]: np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        for l in pose_block.splitlines()
        if l.startswith(("ATOM", "HETATM"))
    }


class DockingEngine:
    """One warm instance per structure: receptor, maps and a Vina handle."""

    def __init__(self, pdb_id):
        self.receptor = receptor_prep.prepare(pdb_id, os.path.join(WORK_ROOT, pdb_id))
        self.pdb_id = self.receptor.pdb_id
        self.structure = self.receptor.structure
        self.zinc = np.array(self.receptor.zinc)
        self._residue_atoms = self._load_receptor_atoms()

        prefix = zn_maps.build(self.receptor)
        self._lock = threading.Lock()  # a Vina object cannot be shared across threads
        # One CPU and a fixed seed keep scores reproducible, which the re-docking
        # benchmark in tests/test_docking.py depends on. Vina parallelises across
        # cores by splitting the search, so results vary slightly with cpu > 1.
        self._vina = Vina(sf_name="ad4", cpu=1, seed=42, verbosity=0)
        self._vina.load_maps(prefix)

    def _load_receptor_atoms(self):
        """Heavy-atom coordinates per residue, for naming a pose's contacts."""
        residues = {}
        with open(self.receptor.path) as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")) or line[77:79].strip() == "HD":
                    continue
                key = (line[17:20].strip(), line[22:27].strip(), line[21])
                residues.setdefault(key, []).append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54])))
        return residues

    def _contacts(self, coords):
        """Residues within CONTACT_CUTOFF of any ligand heavy atom, nearest first."""
        positions = list(coords.values())
        found = []
        for (name, number, chain), atoms in self._residue_atoms.items():
            if name == "ZN":
                continue
            nearest = min(math.dist(a, p) for a in atoms for p in positions)
            if nearest <= CONTACT_CUTOFF:
                found.append({"name": name, "number": number, "chain": chain,
                              "distance": round(nearest, 2)})
        return sorted(found, key=lambda r: r["distance"])

    def site(self):
        """The docking site: box, zinc, the histidine triad and the pocket lining."""
        return self.receptor.site()

    def dock(self, smiles, exhaustiveness=EXHAUSTIVENESS, n_poses=N_POSES):
        """Returns {'poses': [...], 'site': {...}} ranked by score, or {'error': reason}.

        Each pose carries its AD4Zn score, the shortest distance from the zinc
        to a ligand sulfonamide nitrogen, whether that is a real contact, and
        the residues it touches.
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
            distances = [float(np.linalg.norm(coords[i] - self.zinc))
                         for i in sulfonamide_n if i in coords]
            nearest = round(min(distances), 2) if distances else None
            poses.append({
                "rank": rank + 1,
                "score": round(float(energies[rank][0]), 2),
                "zinc_distance": nearest,
                "zinc_contact": nearest is not None and nearest <= ZINC_CONTACT_CUTOFF,
                "contacts": self._contacts(coords)[:8],
                "pose_pdbqt": block.strip() + "\n",
                "pose_pdb": pdbqt_to_pdb_block(block),  # for 3D viewers
            })
        if not poses:
            return {"error": "docking produced no poses"}
        return {
            "poses": poses,
            "has_sulfonamide": bool(sulfonamide_n),
            "site": self.site(),
        }


@lru_cache(maxsize=MAX_WARM_ENGINES)
def get_engine(pdb_id):
    """The engine for a structure, built on first use and kept warm.

    Preparing a receptor and building its maps costs roughly 25 s; both are
    cached on disk as well, so a cold instance that has seen the structure
    before pays only the Vina map load.
    """
    engine = DockingEngine(pdb_id)
    logging.info(f"docking engine ready for {engine.pdb_id} "
                 f"({engine.structure['isoform'].label})")
    return engine

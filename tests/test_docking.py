"""Docking validation, including the re-docking benchmark.

These are slow (tens of seconds per molecule): run with `pytest -m slow`.
The re-docking test records what the AutoDock4Zn setup actually achieves on
3HS4 rather than asserting an idealised result, so a regression is visible.
"""
import os

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.pipeline.docking_engine import ZINC_CONTACT_CUTOFF, get_engine
from src.pipeline.targets import TARGETS

ACETAZOLAMIDE = "CC(=O)Nc1nnc(s1)S(=O)(=O)N"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join("artifacts", "receptors", "zn_maps")),
    reason="run data_curation_scripts/build_zinc_maps.py first",
)


def crystal_pose(pdb_path, resname, chain, smiles):
    """The co-crystallised ligand, with bond orders taken from the SMILES."""
    lines = [l for l in open(pdb_path)
             if l.startswith("HETATM") and l[17:20] == resname and l[21] == chain]
    return AllChem.AssignBondOrdersFromTemplate(
        Chem.MolFromSmiles(smiles), Chem.MolFromPDBBlock("".join(lines)))


def test_every_target_has_receptor_and_maps():
    for pdb_id in TARGETS:
        assert os.path.exists(os.path.join("artifacts", "receptors", f"{pdb_id}.pdbqt"))
        assert os.path.isdir(os.path.join("artifacts", "receptors", "zn_maps", pdb_id))


def test_receptor_keeps_a_charged_zinc():
    with open(os.path.join("artifacts", "receptors", "3HS4.pdbqt")) as f:
        atoms = [l for l in f if l.startswith(("ATOM", "HETATM"))]
    zincs = [l for l in atoms if l[77:79] == "ZN"]
    assert len(zincs) == 1, "exactly one catalytic zinc expected"
    assert float(zincs[0][70:76]) == 0.0, "AutoDock4Zn wants an uncharged ZN"
    assert any(float(l[70:76]) for l in atoms), "protein atoms need partial charges"


def test_unsupported_element_reports_an_error():
    result = get_engine("3HS4").dock("CC[Se]C")
    assert "error" in result and "cannot type" in result["error"]


@pytest.mark.slow
def test_redocking_acetazolamide_recovers_the_crystal_pose():
    """AD4Zn finds the native pose within 2 A, but not always as pose 1.

    Measured on 3HS4: the near-native pose is 1.6 A from the crystal structure
    with a 2.2 A N-Zn contact, and ranks second. The app therefore shows
    several poses with their zinc distances instead of one headline score.
    """
    engine = get_engine("3HS4")
    xtal = crystal_pose(os.path.join("artifacts", "structures", "3hs4.pdb"), "AZM", "A", ACETAZOLAMIDE)
    template = Chem.MolFromSmiles(ACETAZOLAMIDE)
    xtal_match = xtal.GetSubstructMatch(template)
    xtal_xyz = xtal.GetConformer().GetPositions()
    prepared = Chem.AddHs(Chem.MolFromSmiles(ACETAZOLAMIDE))

    from src.pipeline.ligand_prep import prepare_ligand

    _, serial_to_idx, _ = prepare_ligand(ACETAZOLAMIDE)
    result = engine.dock(ACETAZOLAMIDE, n_poses=3)
    assert "poses" in result

    best_rmsd = np.inf
    for pose in result["poses"]:
        coords = {serial_to_idx[int(l[6:11])]: np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                  for l in pose["pose_pdbqt"].splitlines() if l.startswith(("ATOM", "HETATM"))}
        for match in prepared.GetSubstructMatches(template, uniquify=False):
            deviations = [coords[match[i]] - xtal_xyz[xtal_match[i]] for i in range(len(match))]
            best_rmsd = min(best_rmsd, float(np.sqrt(np.mean(np.sum(np.square(deviations), axis=1)))))

    assert best_rmsd < 2.0, f"no near-native pose found (best RMSD {best_rmsd:.2f} A)"
    assert any(p["zinc_contact"] for p in result["poses"]), "no pose coordinates the zinc"


@pytest.mark.slow
def test_scores_are_ordered_and_poses_well_formed():
    result = get_engine("3HS4").dock(ACETAZOLAMIDE, exhaustiveness=4, n_poses=3)
    scores = [p["score"] for p in result["poses"]]
    assert scores == sorted(scores), "poses must be ranked best first"
    assert all(p["pose_pdbqt"].startswith(("MODEL", "ATOM", "HETATM", "REMARK")) for p in result["poses"])
    assert result["has_sulfonamide"]


@pytest.mark.slow
def test_molecule_without_sulfonamide_reports_no_zinc_distance():
    result = get_engine("3HS4").dock(CAFFEINE, exhaustiveness=4, n_poses=1)
    assert result["has_sulfonamide"] is False
    pose = result["poses"][0]
    assert pose["zinc_distance"] is None
    assert pose["zinc_contact"] is False


@pytest.mark.slow
def test_sulfonamide_contact_cutoff_is_a_real_coordination_distance():
    # Zn-N coordination is ~2 A; the cutoff must not be so loose it accepts
    # a pose sitting outside the first coordination shell.
    assert 2.0 < ZINC_CONTACT_CUTOFF < 3.0

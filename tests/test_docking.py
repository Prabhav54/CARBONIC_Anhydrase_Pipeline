"""Docking validation, including the re-docking benchmark.

These are slow: preparing a receptor and building its AutoDock4Zn maps takes
about 25 s per structure and each docking run another 30 to 50 s. Run with
`pytest -m slow`.

The re-docking tests record what this setup actually achieves rather than
asserting an idealised result, so a regression is visible.
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.pipeline import rcsb, zn_maps
from src.pipeline.docking_engine import ZINC_CONTACT_CUTOFF, get_engine
from src.pipeline.ligand_prep import prepare_ligand

ACETAZOLAMIDE = "CC(=O)Nc1nnc(s1)S(=O)(=O)N"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(zn_maps.autogrid_binary() is None,
                       reason="no AutoGrid binary for this platform"),
]


def crystal_pose(pdb_id, resname, chain, smiles):
    """The co-crystallised ligand, with bond orders taken from the SMILES."""
    lines = [l for l in rcsb.coordinates(pdb_id).splitlines(keepends=True)
             if l.startswith("HETATM") and l[17:20].strip() == resname
             and l[21] == chain and l[16] in (" ", "A")]
    assert lines, f"{pdb_id} has no {resname} in chain {chain}"
    return AllChem.AssignBondOrdersFromTemplate(
        Chem.MolFromSmiles(smiles), Chem.MolFromPDBBlock("".join(lines)))


def best_rmsd_to_crystal(result, smiles, xtal):
    """Symmetry-aware RMSD of the closest pose, and the rank it came in at."""
    template = Chem.MolFromSmiles(smiles)
    xtal_match = xtal.GetSubstructMatch(template)
    xtal_xyz = xtal.GetConformer().GetPositions()
    prepared = Chem.AddHs(Chem.MolFromSmiles(smiles))
    _, serial_to_idx, _ = prepare_ligand(smiles)

    best, best_rank = np.inf, None
    for pose in result["poses"]:
        coords = {serial_to_idx[int(l[6:11])]:
                  np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                  for l in pose["pose_pdbqt"].splitlines()
                  if l.startswith(("ATOM", "HETATM"))}
        for match in prepared.GetSubstructMatches(template, uniquify=False):
            deviations = [coords[match[i]] - xtal_xyz[xtal_match[i]] for i in range(len(match))]
            rmsd = float(np.sqrt(np.mean(np.sum(np.square(deviations), axis=1))))
            if rmsd < best:
                best, best_rank = rmsd, pose["rank"]
    return best, best_rank


def test_redocking_acetazolamide_recovers_the_crystal_pose_in_3hs4():
    """Measured: the closest pose is 1.7 A from the crystal structure and the
    top-ranked pose coordinates the zinc at 1.9 A.

    The previous pipeline recovered 1.6 A but ranked the near-native pose
    second, behind a pose 4.2 A away.
    """
    engine = get_engine("3HS4")
    result = engine.dock(ACETAZOLAMIDE)
    assert "poses" in result

    xtal = crystal_pose("3HS4", "AZM", engine.receptor.chain, ACETAZOLAMIDE)
    rmsd, rank = best_rmsd_to_crystal(result, ACETAZOLAMIDE, xtal)
    assert rmsd < 2.0, f"no near-native pose found (best RMSD {rmsd:.2f} A at rank {rank})"
    assert result["poses"][0]["zinc_contact"], "the top-ranked pose should coordinate the zinc"


def test_redocking_the_native_ligand_of_a_ca7_structure():
    """3MDZ holds ethoxzolamide. Measured: recovered to about 1.0 A.

    CA VII could not be screened at all before: it was not one of the eight
    prepared structures, and it has no bioactivity data, so only docking can
    say anything about it.
    """
    engine = get_engine("3MDZ")
    ligand = next(l for l in engine.structure["ligands"] if l["comp_id"] == "EZL")
    result = engine.dock(ligand["smiles"])
    assert "poses" in result

    xtal = crystal_pose("3MDZ", "EZL", engine.receptor.chain, ligand["smiles"])
    rmsd, _ = best_rmsd_to_crystal(result, ligand["smiles"], xtal)
    assert rmsd < 2.0, f"best RMSD {rmsd:.2f} A"


@pytest.mark.parametrize("pdb_id", ["3HS4", "1JCZ", "3D0N", "1ZNC", "2CAB"])
def test_a_sulfonamide_coordinates_the_zinc_in_every_isoform(pdb_id):
    """At least one pose must make the canonical contact.

    1JCZ, 3D0N and 1ZNC are the cases that used to fail outright: the first two
    because their docking box did not contain the zinc, all three because their
    insertion codes broke the zinc pseudo-atom step.
    """
    result = get_engine(pdb_id).dock(ACETAZOLAMIDE)
    assert "poses" in result, result
    distances = [p["zinc_distance"] for p in result["poses"] if p["zinc_distance"]]
    assert any(p["zinc_contact"] for p in result["poses"]), distances


def test_poses_are_ranked_and_carry_their_site_and_contacts():
    result = get_engine("3HS4").dock(ACETAZOLAMIDE, exhaustiveness=4, n_poses=3)
    scores = [p["score"] for p in result["poses"]]
    assert scores == sorted(scores), "poses must be ranked best first"
    assert result["has_sulfonamide"]

    triad = {r["name"] + r["number"] for r in result["site"]["coordinating_residues"]}
    assert triad == {"HIS94", "HIS96", "HIS119"}
    # A zinc-binding pose sits against Thr199, which orients every sulfonamide.
    contacts = {c["name"] + c["number"] for c in result["poses"][0]["contacts"]}
    assert "THR199" in contacts, contacts


def test_molecule_without_sulfonamide_reports_no_zinc_distance():
    result = get_engine("3HS4").dock(CAFFEINE, exhaustiveness=4, n_poses=1)
    assert result["has_sulfonamide"] is False
    pose = result["poses"][0]
    assert pose["zinc_distance"] is None
    assert pose["zinc_contact"] is False


def test_unsupported_element_reports_an_error():
    result = get_engine("3HS4").dock("CC[Se]C")
    assert "error" in result and "cannot type" in result["error"]


def test_sulfonamide_contact_cutoff_is_a_real_coordination_distance():
    # Zn-N coordination is ~2 A; the cutoff must not be so loose it accepts
    # a pose sitting outside the first coordination shell.
    assert 2.0 < ZINC_CONTACT_CUTOFF < 3.0


def test_maps_are_reused_rather_than_rebuilt():
    """Building a map set costs ~13 s, so a second request must hit the cache."""
    import time

    engine = get_engine("3HS4")
    # Build first, so the test does not depend on whether an earlier test left
    # 3HS4's maps in the cache or evicted them.
    zn_maps.build(engine.receptor)

    started = time.time()
    prefix = zn_maps.build(engine.receptor)
    assert time.time() - started < 2.0, "cached maps should return immediately"
    assert prefix.endswith("3HS4_tz")

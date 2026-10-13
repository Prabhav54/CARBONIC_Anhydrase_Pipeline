"""Preparing a receptor at request time, for structures the old pipeline could not.

These need the network (RCSB) and pdb2pqr, and each takes several seconds, so
they are marked `network`. The map-building and docking tests live in
test_docking.py.
"""
import math

import pytest

from src.pipeline import receptor_prep
from src.pipeline.rcsb import StructureError

pytestmark = pytest.mark.network

# The catalytic zinc of every carbonic anhydrase is held by three histidines.
# Author numbering differs between isoforms (CA VII is offset by two), so the
# expectation is the count and the atoms, not the residue numbers.
CA_STRUCTURES = ["3HS4", "2CAB", "5FL6", "1JCZ", "3D0N", "1ZNC", "3MDZ"]


@pytest.fixture(scope="module")
def site_3hs4():
    return receptor_prep.inspect_site("3HS4")


def test_the_site_is_the_textbook_ca2_active_site(site_3hs4):
    triad = {r["name"] + r["number"] for r in site_3hs4["coordinating_residues"]}
    assert triad == {"HIS94", "HIS96", "HIS119"}
    pocket = {r["name"] + r["number"] for r in site_3hs4["pocket_residues"]}
    # Thr199 orients the sulfonamide and Thr200/Glu106/Trp209 line the cone.
    assert {"THR199", "THR200", "GLU106", "TRP209"} <= pocket


def test_the_box_is_centred_on_the_zinc(site_3hs4):
    """The committed boxes were not: for five of eight targets the zinc sat
    16-24 A outside the box, so no pose could ever coordinate it."""
    assert site_3hs4["box_centre"] == site_3hs4["zinc"]
    assert site_3hs4["box_size"] == [20.0, 20.0, 20.0]


@pytest.mark.parametrize("pdb_id", CA_STRUCTURES)
def test_every_isoform_yields_three_coordinating_histidines(pdb_id):
    site = receptor_prep.inspect_site(pdb_id)
    coordinating = site["coordinating_residues"]
    assert len(coordinating) == 3, coordinating
    assert all(r["name"].startswith("HIS") for r in coordinating), coordinating
    assert all(1.8 <= r["distance"] <= 2.5 for r in coordinating), coordinating


def test_a_non_carbonic_anhydrase_has_no_site():
    with pytest.raises(StructureError, match="not a human carbonic anhydrase"):
        receptor_prep.inspect_site("1HHO")


@pytest.fixture(scope="module")
def receptor_3hs4(tmp_path_factory):
    return receptor_prep.prepare("3HS4", str(tmp_path_factory.mktemp("rec")))


def _atoms(path):
    with open(path) as f:
        return [l for l in f if l.startswith(("ATOM", "HETATM"))]


def test_the_receptor_is_typed_and_charged_for_autodock(receptor_3hs4):
    atoms = _atoms(receptor_3hs4.path)
    types = {l[77:79].strip() for l in atoms}
    assert types <= receptor_prep.RECEPTOR_TYPES, types - receptor_prep.RECEPTOR_TYPES
    # Aromatic carbons, acceptor nitrogens and polar hydrogens must all appear,
    # because the AD4 hydrogen-bond and aromatic terms key off them.
    assert {"A", "NA", "HD", "OA", "C", "N"} <= types
    assert any(float(l[70:76]) for l in atoms)


def test_the_receptor_keeps_exactly_one_uncharged_zinc(receptor_3hs4):
    zincs = [l for l in _atoms(receptor_3hs4.path) if l[77:79].strip() == "ZN"]
    assert len(zincs) == 1, "exactly one catalytic zinc expected"
    assert float(zincs[0][70:76]) == 0.0, "AutoDock4Zn wants an uncharged ZN"


def test_the_net_charge_is_physically_plausible(receptor_3hs4):
    """A misaligned charge column reads garbage and produces a huge total."""
    total = sum(float(l[70:76]) for l in _atoms(receptor_3hs4.path))
    assert abs(total) < 25, total


def test_no_hydrogen_is_left_sitting_on_the_zinc(receptor_3hs4):
    """pdb2pqr protonates the histidine triad as if the metal were absent."""
    zinc = receptor_3hs4.zinc
    for line in _atoms(receptor_3hs4.path):
        if line[77:79].strip() != "HD":
            continue
        position = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        assert math.dist(position, zinc) >= receptor_prep.ZINC_HYDROGEN_CLASH, line


@pytest.mark.parametrize("pdb_id", ["1ZNC", "1JCZ", "3D0N"])
def test_residues_with_insertion_codes_keep_a_numeric_sequence_column(pdb_id):
    """These three entries have insertion codes. Writing "50A" into the numeric
    residue column crashes AutoDock4Zn's zinc_pseudo.py, which parses it with
    int(), and no zinc pseudo-atom gets placed."""
    receptor = receptor_prep.prepare_in_temp(pdb_id)
    for line in _atoms(receptor.path):
        int(line[22:26])  # must not raise

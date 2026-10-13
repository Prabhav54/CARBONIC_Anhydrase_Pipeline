"""Resolving a typed PDB ID to a carbonic anhydrase structure.

The tests marked `network` hit the live RCSB and ChEMBL APIs, which is the point:
the app's claim is that it reads real structures and real bioactivity rather than
a hardcoded table, and that claim is only tested against the real services.
"""
import pytest

from src.pipeline import chembl, isoforms, rcsb
from src.pipeline.rcsb import StructureError

pytestmark = pytest.mark.network


@pytest.mark.parametrize("raw,expected", [
    ("3hs4", "3HS4"), (" 5fl6 ", "5FL6"), ("1CA2", "1CA2"),
])
def test_normalise_accepts_valid_ids(raw, expected):
    assert rcsb.normalise_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "ABC", "ABCD", "12345", "3HS!", None])
def test_normalise_rejects_malformed_ids(raw):
    with pytest.raises(StructureError, match="not a PDB ID"):
        rcsb.normalise_id(raw)


def test_isoform_table_covers_the_twelve_catalytic_isoforms():
    labels = sorted(i.label for i in isoforms.all_isoforms())
    assert labels == sorted(["CA I", "CA II", "CA III", "CA IV", "CA VA", "CA VB",
                             "CA VI", "CA VII", "CA IX", "CA XII", "CA XIII", "CA XIV"])


def test_acatalytic_isoforms_are_not_offered():
    """CA VIII, X and XI lack the zinc triad, so they cannot be docked."""
    for accession in ("P35219", "Q9NS85", "O75493"):
        assert isoforms.by_accession(accession) is None


@pytest.mark.parametrize("pdb_id,label", [
    ("3HS4", "CA II"),
    ("2CAB", "CA I"),
    ("5FL6", "CA IX"),
    ("1JCZ", "CA XII"),
    ("3D0N", "CA XIII"),
    ("1ZNC", "CA IV"),
    ("3MDZ", "CA VII"),
])
def test_structures_resolve_to_the_right_isoform(pdb_id, label):
    """Seven isoforms, five of which the previous eight-target table could not reach."""
    structure = rcsb.describe(pdb_id)
    assert structure["isoform"].label == label
    assert structure["pdb_id"] == pdb_id
    assert structure["title"]


def test_a_non_carbonic_anhydrase_is_rejected_with_its_real_identity():
    with pytest.raises(StructureError) as excinfo:
        rcsb.describe("1HHO")
    message = str(excinfo.value)
    assert "not a human carbonic anhydrase" in message
    assert "HAEMOGLOBIN" in message.upper()


def test_a_nonexistent_entry_is_rejected():
    # 0ZZZ is well-formed but unused: deposited IDs start with 1-9. (1ZZZ and
    # 8ZZZ, the obvious guesses, are both real structures.)
    with pytest.raises(StructureError, match="no entry"):
        rcsb.describe("0ZZZ")


def test_the_cocrystal_ligand_is_the_real_one_not_a_buffer():
    """3HS4 holds acetazolamide plus glycerol; only the inhibitor is a ligand."""
    ligands = {l["comp_id"]: l for l in rcsb.describe("3HS4")["ligands"]}
    assert "AZM" in ligands
    assert "GOL" not in ligands and "ZN" not in ligands
    assert ligands["AZM"]["smiles"] == "CC(=O)Nc1nnc(s1)S(=O)(=O)N"


def test_coordinates_are_cached_after_the_first_download():
    first = rcsb.coordinates("3HS4")
    assert first.lstrip().startswith(("HEADER", "ATOM", "REMARK", "TITLE"))
    assert rcsb.coordinates("3hs4") == first


def test_chembl_targets_differ_between_isoforms():
    """The old table held three ChEMBL ids; these are resolved per isoform."""
    resolved = {}
    for label in ("CA I", "CA II", "CA IX", "CA XII"):
        isoform = isoforms.by_label(label)
        target = chembl.target_for(isoform.accession)
        assert target is not None, label
        resolved[label] = target["chembl_id"]
    assert len(set(resolved.values())) == len(resolved), resolved


def test_known_inhibitors_are_isoform_specific_and_carry_provenance():
    ca2, _ = chembl.known_inhibitors(isoforms.by_label("CA II").accession)
    ca9, _ = chembl.known_inhibitors(isoforms.by_label("CA IX").accession)
    assert ca2 and ca9
    for row in ca2 + ca9:
        assert row["smiles"] and row["chembl_id"]
        assert row["source"] in ("mechanism", "activity")
        assert row["provenance"]
    # Two different isoforms should not return an identical list.
    assert {r["chembl_id"] for r in ca2} != {r["chembl_id"] for r in ca9}

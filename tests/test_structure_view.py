"""The structure panel: what step one of the interface shows about a target.

The three kinds of claim it carries are not equally strong, and the tests check
they stay distinguishable: curated context about the isoform, drugs ChEMBL
annotates against the target, and the genetic disease UniProt annotates for the
gene.
"""
import pytest

from src.pipeline import isoforms, structure_view, uniprot
from src.pipeline.admet_engine import evaluate
from src.pipeline.rcsb import StructureError

ACETAZOLAMIDE = "CC(=O)Nc1nnc(s1)S(=O)(=O)N"


def test_every_isoform_has_therapeutic_context():
    """A user asked to pick a target needs to know why it is a target."""
    for isoform in isoforms.all_isoforms():
        assert isoform.therapeutic, isoform.label
        assert isoform.role, isoform.label


def test_lipinski_detail_reports_each_rule_with_its_limit():
    detail = evaluate(ACETAZOLAMIDE)["Lipinski"]
    assert [c["short"] for c in detail] == ["MW", "LogP", "HBD", "HBA"]
    assert all(c["limit"] for c in detail)
    assert all(isinstance(c["passed"], bool) for c in detail)
    # Acetazolamide is small and polar: it passes all four.
    assert all(c["passed"] for c in detail)


def test_lipinski_flags_the_rule_that_failed():
    """A large greasy molecule should fail on weight and lipophilicity."""
    big = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCc1ccc(-c2ccc(-c3ccccc3)cc2)cc1"
    result = evaluate(big)
    failed = {c["short"] for c in result["Lipinski"] if not c["passed"]}
    assert "MW" in failed and "LogP" in failed
    assert result["Violations"] == len(failed)
    assert result["Pass"] is False
    assert "MW" in result["Reasons"]


@pytest.mark.network
def test_build_assembles_the_panel_for_a_ca2_structure():
    view = structure_view.build("3HS4")
    assert view["isoform"].label == "CA II"
    assert view["structure"]["image_url"].endswith("3hs4_assembly-1.jpeg")

    # The docking site, found from the catalytic zinc.
    triad = {r["name"] + r["number"] for r in view["site"]["coordinating_residues"]}
    assert triad == {"HIS94", "HIS96", "HIS119"}

    # Drugs come from ChEMBL's mechanism annotations, so they are real.
    assert view["drugs"], "CA II should have annotated drugs"
    assert all(d["chembl_id"] and d["provenance"] for d in view["drugs"])


@pytest.mark.network
def test_uniprot_disease_annotations_are_only_claimed_where_they_exist():
    """Three of the twelve isoforms carry one; the rest must come back empty.

    `diseases()` returns [] when the lookup fails, by design, so an unreachable
    UniProt would otherwise look exactly like "no annotation" and make this test
    pass for the wrong reason. The direct fetch below distinguishes the two.
    """
    try:
        uniprot.fetch("P00918")
    except Exception as e:
        pytest.skip(f"UniProt unreachable: {e}")

    assert any(d["acronym"] == "OPTB3" for d in uniprot.diseases("P00918"))   # CA II
    assert any(d["acronym"] == "RP17" for d in uniprot.diseases("P22748"))    # CA IV
    assert uniprot.diseases("P00915") == []                                   # CA I
    assert uniprot.diseases("Q16790") == []                                   # CA IX


@pytest.mark.network
def test_a_non_carbonic_anhydrase_is_rejected():
    with pytest.raises(StructureError, match="not a human carbonic anhydrase"):
        structure_view.build("1HHO")


@pytest.mark.network
def test_json_form_is_serialisable_and_carries_the_site():
    import json

    payload = structure_view.as_json(structure_view.build("5FL6"))
    text = json.dumps(payload)              # must not raise
    assert payload["isoform"]["label"] == "CA IX"
    assert payload["site"]["box_size"] == [20.0, 20.0, 20.0]
    assert payload["site"]["box_centre"] == payload["site"]["zinc"]
    assert "pocket_residues" in payload["site"]
    assert len(text) > 500

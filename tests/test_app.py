"""Route-level tests for the Flask app.

Screening and docking both reach the network and docking takes a minute, so the
tests that cover the page's own logic stub them out. The end-to-end behaviour is
covered in test_rcsb.py, test_receptor_prep.py and test_docking.py.
"""
import pytest

import app as flask_app
from src.pipeline import isoforms
from src.pipeline.rcsb import StructureError


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


LIPINSKI = [
    {"name": "Molecular weight", "short": "MW", "value": 222.0, "limit": 500, "passed": True},
    {"name": "Lipophilicity", "short": "LogP", "value": -0.3, "limit": 5, "passed": True},
    {"name": "H-bond donors", "short": "HBD", "value": 2, "limit": 5, "passed": True},
    {"name": "H-bond acceptors", "short": "HBA", "value": 7, "limit": 10, "passed": True},
]


def _row(name, smiles, score=7.0, percentile=70):
    return {
        "name": name, "smiles": smiles, "Predicted_pIC50": 7.5, "IC50_nM": 31.6,
        "ADMET_QED": 0.6, "MW": 222.0, "LogP": -0.3, "TPSA": 115.0,
        "Lipinski": LIPINSKI, "Violations": 0, "Lipinski_Pass": True,
        "Lipinski_Reasons": "None",
        "PubChem_Link": "#", "score": score, "percentile": percentile,
        "provenance": "co-crystallised in 3HS4", "chembl_id": None,
    }


def _structure(scored=True):
    isoform = isoforms.by_label("CA II")
    return {
        "pdb_id": "3HS4", "title": "Human carbonic anhydrase II", "method": "X-RAY",
        "resolution": 1.1, "deposited": "2009-06-10", "uniprot": isoform.accession,
        "isoform": isoform, "ligands": [], "rcsb_url": "https://www.rcsb.org/structure/3HS4",
        "image_url": "https://cdn.rcsb.org/images/structures/3hs4_assembly-1.jpeg",
    }


def _view(scored=True, site=True):
    isoform = isoforms.by_label("CA II")
    return {
        "structure": _structure(),
        "isoform": isoform,
        "site": {
            "pdb_id": "3HS4", "chain": "A", "zinc": [-6.72, -1.68, 15.29],
            "box_centre": [-6.72, -1.68, 15.29], "box_size": [20.0, 20.0, 20.0],
            "coordinating_residues": [{"name": "HIS", "number": "94", "chain": "A",
                                       "atom": "NE2", "distance": 1.99}],
            "pocket_residues": [{"name": "THR", "number": "199", "chain": "A",
                                 "distance": 3.5}],
            "zinc_count": 1,
        } if site else None,
        "site_error": None if site else "no zinc here",
        "diseases": [{"name": "Osteopetrosis, autosomal recessive 3", "acronym": "OPTB3",
                      "description": "A rare genetic disease.", "mim": "259730"}],
        "drugs": [{"name": "Acetazolamide", "chembl_id": "CHEMBL20",
                   "provenance": "approved drug acting on this isoform (ChEMBL mechanism)",
                   "smiles": "CC(=O)Nc1nnc(s1)S(=O)(=O)N", "source": "mechanism",
                   "measured": None}],
        "chembl_target": {"chembl_id": "CHEMBL205", "pref_name": "Carbonic anhydrase 2"},
        "scored": scored,
        "withheld_reason": None if scored else "no data",
    }


def _result(scored=True):
    isoform = isoforms.by_label("CA II")
    return {
        "structure": _structure(),
        "isoform": isoform,
        "novels": [_row("Mine", "CCO", score=None if not scored else 4.2, percentile=42)],
        "refs": [_row("Acetazolamide", "CC(=O)Nc1nnc(s1)S(=O)(=O)N")],
        "reference_note": "Known CA II inhibitors from ChEMBL target CHEMBL205.",
        "skipped": ["bogus"],
        "scored": scored,
        "scale": "Out of 10 by percentile against 1538 molecules.",
    }


def test_parse_candidates_handles_names_and_blank_lines():
    candidates = flask_app.parse_candidates("Aspirin, CC(=O)Oc1ccccc1C(=O)O\n\nCCO\n")
    assert candidates == [
        {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"name": "Candidate 2", "smiles": "CCO"},
    ]


def test_parse_candidates_keeps_commas_in_the_name_only():
    # Only the last comma separates name from SMILES.
    assert flask_app.parse_candidates("Drug A, batch 2, CCO") == [
        {"name": "Drug A, batch 2", "smiles": "CCO"}
    ]


def test_ordinal_filter():
    assert [flask_app.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 62, 100)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "62nd", "100th"]


def test_home_page_does_not_preselect_a_target(client):
    """The user has to type the structure: defaulting it hid that it matters.

    The suggestion buttons underneath do carry PDB IDs, which is fine: nothing
    is submitted until the user picks one. What must not happen is the text
    field arriving with a structure already in it.
    """
    import re

    page = client.get("/").data.decode()
    field = re.search(r'<input[^>]*name="target_pdb"[^>]*>', page)
    assert field, "the PDB ID field should be on the page"
    assert re.search(r'value="\s*"', field.group(0)), field.group(0)


def test_home_page_lists_every_isoform_and_says_whether_it_is_scored(client):
    page = client.get("/").data.decode()
    for isoform in isoforms.all_isoforms():
        assert isoform.label in page
        # Each one is labelled either with its accuracy or as not scored, so a
        # user can see what the model can and cannot do before they search.
        if isoform.is_scored:
            assert f"{isoform.test_r2:.2f}" in page, isoform.label
        else:
            assert isoform.withheld_reason in page, isoform.label


def test_step_two_is_locked_until_a_structure_is_loaded(client):
    page = client.get("/").data.decode()
    assert "Load a structure first" in page
    assert 'name="smiles_input"' not in page


def test_structure_step_renders_the_target_and_its_context(client, monkeypatch):
    monkeypatch.setattr(flask_app.structure_view, "build", lambda pdb_id: _view())
    page = client.post("/structure", data={"target_pdb": "3HS4"}).data.decode()
    assert "Human carbonic anhydrase II" in page
    assert "cdn.rcsb.org/images/structures/3hs4" in page       # structure image
    assert "glaucoma" in page                                  # therapeutic context
    assert "Acetazolamide" in page                             # ChEMBL-annotated drug
    assert "OPTB3" in page                                     # UniProt disease
    assert "not the same claim" in page                        # the two are kept apart
    assert "HIS94" in page and "THR199" in page                # docking site
    assert 'id="viewer3d"' in page                             # 3D viewer mounted
    assert 'name="smiles_input"' in page                       # step two now open


def test_structure_step_requires_an_id(client):
    response = client.post("/structure", data={"target_pdb": ""})
    assert b"Enter a PDB ID" in response.data


def test_structure_step_shows_why_an_entry_was_rejected(client, monkeypatch):
    monkeypatch.setattr(flask_app.structure_view, "build",
                        lambda pdb_id: (_ for _ in ()).throw(
                            StructureError("1HHO is not a human carbonic anhydrase.")))
    page = client.post("/structure", data={"target_pdb": "1HHO"}).data.decode()
    assert "not a human carbonic anhydrase" in page


def test_predict_requires_a_target(client):
    response = client.post("/predict", data={"target_pdb": "", "smiles_input": "CCO"})
    assert response.status_code == 200
    assert b"Choose a structure first" in response.data


def test_predict_requires_at_least_one_candidate(client, monkeypatch):
    monkeypatch.setattr(flask_app.structure_view, "build", lambda pdb_id: _view())
    response = client.post("/predict", data={"target_pdb": "3HS4", "smiles_input": "   \n"})
    assert b"at least one SMILES" in response.data


def test_predict_renders_scores_in_plain_language(client, monkeypatch):
    monkeypatch.setattr(flask_app.structure_view, "build", lambda pdb_id: _view())
    monkeypatch.setattr(flask_app.predict_pipeline, "screen",
                        lambda candidates, pdb_id: _result())
    page = client.post("/predict", data={"target_pdb": "3HS4",
                                         "smiles_input": "CCO"}).data.decode()
    assert "4.2" in page                                  # the score out of 10
    assert "stronger than" in page and "42%" in page      # said in words, not "42nd pct"
    assert "MW 222.0" in page                             # Lipinski values are shown
    assert "all 4 rules pass" in page
    assert "bogus" in page                                # unparseable input reported


def test_predict_withholds_the_score_when_the_isoform_is_not_scored(client, monkeypatch):
    monkeypatch.setattr(flask_app.structure_view, "build",
                        lambda pdb_id: _view(scored=False, site=False))
    monkeypatch.setattr(flask_app.predict_pipeline, "screen",
                        lambda candidates, pdb_id: _result(scored=False))
    page = client.post("/predict", data={"target_pdb": "1JCZ",
                                         "smiles_input": "CCO"}).data.decode()
    assert "potency withheld" in page
    assert "not scored" in page
    assert "no zinc here" in page


def test_dock_requires_smiles(client):
    response = client.post("/dock", json={"target_pdb": "3HS4", "smiles": ""})
    assert response.status_code == 400
    assert b"No SMILES" in response.data


def test_dock_reports_an_unusable_structure(client, monkeypatch):
    def reject(pdb_id):
        raise StructureError("9ZZZ is not a human carbonic anhydrase.")

    monkeypatch.setattr(flask_app, "get_engine", reject)
    response = client.post("/dock", json={"target_pdb": "9ZZZ", "smiles": "CCO"})
    assert response.status_code == 400
    assert b"not a human carbonic anhydrase" in response.json["error"].encode()


def test_site_route_reports_a_rejection(client, monkeypatch):
    monkeypatch.setattr(flask_app.receptor_prep, "inspect_site",
                        lambda pdb_id: (_ for _ in ()).throw(StructureError("nope")))
    response = client.get("/site/1HHO")
    assert response.status_code == 400
    assert response.json["error"] == "nope"

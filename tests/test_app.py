"""Route-level tests for the Flask app, with docking stubbed out (it is slow)."""
import pytest

import app as flask_app
from src.pipeline.targets import DEFAULT_TARGET


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    return flask_app.app.test_client()


def test_parse_candidates_handles_names_and_blank_lines():
    candidates = flask_app.parse_candidates("Aspirin, CC(=O)Oc1ccccc1C(=O)O\n\nCCO\n")
    assert candidates == [
        {"Molecule_Name": "Aspirin", "clean_smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"Molecule_Name": "Candidate 2", "clean_smiles": "CCO"},
    ]


def test_parse_candidates_keeps_commas_in_the_name_only():
    # Only the last comma separates name from SMILES.
    assert flask_app.parse_candidates("Drug A, batch 2, CCO") == [
        {"Molecule_Name": "Drug A, batch 2", "clean_smiles": "CCO"}
    ]


def test_home_page_lists_the_targets(client):
    response = client.get("/")
    assert response.status_code == 200
    assert DEFAULT_TARGET.encode() in response.data
    assert b"5FL6" in response.data


def test_predict_rejects_an_unsupported_target(client):
    response = client.post("/predict", data={"target_pdb": "1ABC", "smiles_input": "CCO"})
    assert response.status_code == 200
    assert b"Unsupported target" in response.data


def test_predict_requires_at_least_one_candidate(client):
    response = client.post("/predict", data={"target_pdb": DEFAULT_TARGET, "smiles_input": "   \n"})
    assert b"at least one SMILES" in response.data


def test_predict_renders_results(client, monkeypatch):
    monkeypatch.setattr(flask_app.predict_pipeline, "screen", lambda candidates, pdb_id: (
        [{"Molecule_Name": "Acetazolamide", "clean_smiles": "CC(=O)Nc1nnc(s1)S(=O)(=O)N",
          "Predicted_pIC50": 7.5, "IC50_nM": 31.6, "ADMET_QED": 0.6, "Violations": 0,
          "Lipinski_Reasons": "None", "PubChem_Link": "#"}],
        [{"Molecule_Name": "Mine", "clean_smiles": "CCO", "Predicted_pIC50": 5.0,
          "IC50_nM": 10000.0, "ADMET_QED": 0.4, "Violations": 1,
          "Lipinski_Reasons": "LogP > 5", "PubChem_Link": "#"}],
        "ChEMBL API", ["bogus"],
    ))
    response = client.post("/predict", data={"target_pdb": DEFAULT_TARGET, "smiles_input": "CCO"})
    assert response.status_code == 200
    assert b"Acetazolamide" in response.data
    assert b"Could not parse 1 input" in response.data


def test_predict_reports_a_failure_without_leaking_internals(client, monkeypatch):
    def boom(candidates, pdb_id):
        raise RuntimeError("model file is corrupt")

    monkeypatch.setattr(flask_app.predict_pipeline, "screen", boom)
    response = client.post("/predict", data={"target_pdb": DEFAULT_TARGET, "smiles_input": "CCO"})
    assert response.status_code == 200
    assert b"Screening failed" in response.data
    assert b"corrupt" not in response.data


def test_dock_validates_its_input(client):
    assert client.post("/dock", json={"target_pdb": "nope", "smiles": "CCO"}).status_code == 400
    assert client.post("/dock", json={"target_pdb": DEFAULT_TARGET, "smiles": ""}).status_code == 400


def test_dock_returns_poses(client, monkeypatch):
    monkeypatch.setattr(flask_app, "get_engine", lambda pdb_id: type("E", (), {
        "dock": staticmethod(lambda smiles: {
            "poses": [{"rank": 1, "score": -9.7, "zinc_distance": 2.1,
                       "zinc_contact": True, "pose_pdbqt": "ATOM\n"}],
            "has_sulfonamide": True,
        })
    })())
    response = client.post("/dock", json={"target_pdb": DEFAULT_TARGET, "smiles": "CCO"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["scoring"] == "AutoDock4Zn"
    assert body["poses"][0]["zinc_contact"] is True


def test_dock_surfaces_a_ligand_preparation_error(client, monkeypatch):
    monkeypatch.setattr(flask_app, "get_engine", lambda pdb_id: type("E", (), {
        "dock": staticmethod(lambda smiles: {"error": "Vina cannot type element(s): Se"})
    })())
    response = client.post("/dock", json={"target_pdb": DEFAULT_TARGET, "smiles": "CC[Se]C"})
    assert response.status_code == 422
    assert "Se" in response.get_json()["error"]

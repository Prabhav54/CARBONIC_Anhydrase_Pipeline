"""Tests for the MD helpers that need neither OpenMM nor a trajectory.

src.analysis.md imports OpenMM, OpenFF and mdtraj inside the functions that use
them, so receptor extraction and the summary statistics are testable on their own.
"""
import numpy as np
import pytest

from src.analysis.md import extract_receptor, summarise


def test_extract_receptor_keeps_one_chain_and_its_zinc(tmp_path):
    out = tmp_path / "receptor.pdb"
    path, chain = extract_receptor("artifacts/structures/3hs4.pdb", out_path=str(out))
    lines = out.read_text().splitlines()

    assert path == str(out) and chain == "A"
    assert all(l[21] == chain for l in lines if l.startswith(("ATOM", "HETATM")))
    assert sum(1 for l in lines if l.startswith("HETATM") and l[17:20] == " ZN") == 1
    # Waters and the co-crystallised ligand must be gone.
    assert not any(l[17:20] in ("HOH", "AZM", "GOL") for l in lines if l.startswith(("ATOM", "HETATM")))
    assert lines[-1] == "END"


def test_extract_receptor_picks_a_chain_for_a_multimer(tmp_path):
    out = tmp_path / "receptor.pdb"
    _, chain = extract_receptor("artifacts/structures/5fl6.pdb", out_path=str(out))
    assert chain in "ABCD"
    text = out.read_text()
    assert " ZN " in text


def test_extract_receptor_honours_an_explicit_chain(tmp_path):
    out = tmp_path / "receptor.pdb"
    _, chain = extract_receptor("artifacts/structures/5fl6.pdb", chain="B", out_path=str(out))
    assert chain == "B"


def test_extract_receptor_rejects_a_structure_without_zinc(tmp_path):
    source = tmp_path / "apo.pdb"
    source.write_text("ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00  0.00           N\n")
    with pytest.raises(ValueError, match="no catalytic zinc"):
        extract_receptor(str(source), out_path=str(tmp_path / "out.pdb"))


def synthetic_metrics(n=100, drift=0.0, zinc_start=2.0, zinc_drift=0.0, bonds=3):
    ramp = np.linspace(0, 1, n)
    return {
        "ligand_rmsd": 0.5 + drift * ramp,
        "backbone_rmsd": 1.0 + 0.2 * ramp,
        "zinc_distance": zinc_start + zinc_drift * ramp,
        "hydrogen_bonds": np.full(n, bonds),
    }


def test_summarise_ignores_the_equilibration_window():
    # A spike inside the first 20% of frames must not reach the summary.
    metrics = synthetic_metrics()
    metrics["ligand_rmsd"] = metrics["ligand_rmsd"].copy()
    metrics["ligand_rmsd"][:10] = 99.0

    summary = summarise(metrics, equilibration_fraction=0.2)
    assert summary["frames_analysed"] == 80
    assert summary["ligand_rmsd_max"] < 1.0, "equilibration frames leaked into the summary"


def test_summarise_reports_a_retained_zinc_contact():
    summary = summarise(synthetic_metrics(zinc_start=2.0, zinc_drift=0.0))
    assert summary["zinc_contact_fraction"] == 1.0
    assert summary["zinc_distance_mean"] == 2.0
    assert summary["zinc_distance_final"] == 2.0


def test_summarise_reports_a_lost_zinc_contact():
    # Starts coordinated, ends 6 A away: the contact should read as mostly lost.
    summary = summarise(synthetic_metrics(zinc_start=2.0, zinc_drift=4.0))
    assert summary["zinc_distance_final"] == 6.0
    assert summary["zinc_contact_fraction"] < 0.2


def test_summarise_without_a_zinc_distance():
    metrics = synthetic_metrics()
    metrics["zinc_distance"] = None
    summary = summarise(metrics)
    assert "zinc_distance_mean" not in summary
    assert summary["hydrogen_bonds_mean"] == 3.0


def test_summarise_tracks_ligand_drift():
    stable = summarise(synthetic_metrics(drift=0.0))
    drifting = summarise(synthetic_metrics(drift=5.0))
    assert stable["ligand_rmsd_mean"] < 2.0 < drifting["ligand_rmsd_mean"]

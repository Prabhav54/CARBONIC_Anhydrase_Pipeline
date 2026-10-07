"""Checks that a docked pose survives the round trip back to a molecule.

The conversion relies on RDKit's RemoveHs preserving heavy-atom order, so the
coordinates are verified independently against the PDBQT text rather than
against another call to the same code.
"""
import numpy as np
import pytest
from rdkit import Chem

from src.pipeline.ligand_prep import prepare_ligand
from src.pipeline.pose_io import heavy_atom_rmsd, pose_to_mol, write_pose_sdf

ACETAZOLAMIDE = "CC(=O)Nc1nnc(s1)S(=O)(=O)N"
DORZOLAMIDE = "CCN[C@H]1C[C@@H](C)S(=O)(=O)c2sc(cc12)S(N)(=O)=O"


def pdbqt_atoms(pdbqt):
    """(element, coordinate) for every atom line, in file order."""
    atoms = []
    for line in pdbqt.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            element = line[77:79].strip()
            element = {"A": "C", "NA": "N", "OA": "O", "SA": "S", "HD": "H"}.get(element, element)
            atoms.append((element, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    return atoms


@pytest.mark.parametrize("smiles", [ACETAZOLAMIDE, DORZOLAMIDE])
def test_every_heavy_atom_keeps_its_coordinates_and_element(smiles):
    pdbqt, serial_to_idx, _ = prepare_ligand(smiles)
    mol = pose_to_mol(smiles, pdbqt, serial_to_idx)
    positions = mol.GetConformer().GetPositions()

    pose = [(element, xyz) for element, xyz in pdbqt_atoms(pdbqt) if element != "H"]
    for element, xyz in pose:
        distances = np.linalg.norm(positions - xyz, axis=1)
        nearest = int(np.argmin(distances))
        assert distances[nearest] < 1e-3, f"no atom at the pose position of {element}"
        assert mol.GetAtomWithIdx(nearest).GetSymbol() == element, "element mismatch at that position"


def test_hydrogens_are_rebuilt():
    pdbqt, serial_to_idx, _ = prepare_ligand(ACETAZOLAMIDE)
    mol = pose_to_mol(ACETAZOLAMIDE, pdbqt, serial_to_idx)
    # C4H6N4O3S2: 13 heavy atoms, 6 hydrogens. The PDBQT kept only the polar ones.
    assert mol.GetNumAtoms() == 19
    assert sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1) == 6
    assert mol.GetNumConformers() == 1


def test_bond_orders_and_formula_are_preserved():
    pdbqt, serial_to_idx, _ = prepare_ligand(ACETAZOLAMIDE)
    mol = pose_to_mol(ACETAZOLAMIDE, pdbqt, serial_to_idx)
    reference = Chem.AddHs(Chem.MolFromSmiles(ACETAZOLAMIDE))
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(Chem.RemoveHs(reference))


def test_sdf_round_trip_keeps_the_pose(tmp_path):
    pdbqt, serial_to_idx, _ = prepare_ligand(DORZOLAMIDE)
    path = write_pose_sdf(str(tmp_path / "pose.sdf"), DORZOLAMIDE, pdbqt, serial_to_idx,
                          name="pose", properties={"score": -9.5})
    loaded = Chem.SDMolSupplier(path, removeHs=False)[0]
    assert loaded is not None, "RDKit could not read the SDF back"
    assert loaded.GetProp("score") == "-9.5"
    assert loaded.GetProp("_Name") == "pose"
    assert heavy_atom_rmsd(loaded, pose_to_mol(DORZOLAMIDE, pdbqt, serial_to_idx)) < 1e-3


def test_missing_atoms_are_rejected():
    pdbqt, serial_to_idx, _ = prepare_ligand(ACETAZOLAMIDE)
    truncated = "\n".join(pdbqt.splitlines()[:5])
    with pytest.raises(ValueError, match="missing"):
        pose_to_mol(ACETAZOLAMIDE, truncated, serial_to_idx)


def test_pdbqt_to_pdb_fixes_the_element_column():
    from src.pipeline.pose_io import pdbqt_to_pdb_block

    pdbqt, _, _ = prepare_ligand(ACETAZOLAMIDE)
    block = pdbqt_to_pdb_block(pdbqt)

    elements = [l[76:78].strip() for l in block.splitlines() if l.startswith(("ATOM", "HETATM"))]
    assert set(elements) <= {"C", "N", "O", "S", "H"}, "AutoDock types left in the element column"
    assert "A" not in elements and "HD" not in elements
    assert block.rstrip().endswith("END")

    # RDKit is a stand-in for any PDB reader: it should now perceive the elements.
    mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    assert mol is not None
    counts = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    assert counts.get("C") == 4 and counts.get("N") == 4 and counts.get("S") == 2, counts

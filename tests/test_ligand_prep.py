"""Checks the RDKit-only PDBQT writer against Meeko, the reference implementation.

Meeko is a dev-only dependency (it needs SciPy and gemmi, which do not fit in
the serverless bundle), so the Meeko comparisons skip if it is not installed.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.pipeline.ligand_prep import LigandPrepError, autodock_type, prepare_ligand

DRUGS = {
    "acetazolamide": "CC(=O)Nc1nnc(s1)S(=O)(=O)N",
    "methazolamide": "CC(=O)/N=c1/sc(S(N)(=O)=O)nn1C",
    "dorzolamide": "CCN[C@H]1C[C@@H](C)S(=O)(=O)c2sc(cc12)S(N)(=O)=O",
    "ethoxzolamide": "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    "celecoxib": "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


def atom_types(pdbqt):
    return [l[77:79].strip() for l in pdbqt.splitlines() if l.startswith(("ATOM", "HETATM"))]


def torsdof(pdbqt):
    return int(next(l for l in pdbqt.splitlines() if l.startswith("TORSDOF")).split()[1])


@pytest.mark.parametrize("smiles", DRUGS.values(), ids=list(DRUGS))
def test_pdbqt_is_well_formed(smiles):
    pdbqt, serial_to_idx, _ = prepare_ligand(smiles)

    assert pdbqt.count("ROOT") == 2, "expected exactly one ROOT/ENDROOT pair"
    assert pdbqt.count("BRANCH") == 2 * pdbqt.count("ENDBRANCH")
    assert len(atom_types(pdbqt)) == len(serial_to_idx)
    # Serials must be contiguous from 1, which Vina relies on.
    assert sorted(serial_to_idx) == list(range(1, len(serial_to_idx) + 1))
    # Non-polar hydrogens are merged, so no plain H type survives.
    assert "H" not in atom_types(pdbqt)


@pytest.mark.parametrize("smiles", DRUGS.values(), ids=list(DRUGS))
def test_atom_types_and_torsions_match_meeko(smiles):
    meeko = pytest.importorskip("meeko", reason="dev-only dependency")

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    setup = meeko.MoleculePreparation().prepare(mol)[0]
    reference = meeko.PDBQTWriterLegacy.write_string(setup)[0]

    ours, _, _ = prepare_ligand(smiles)
    assert sorted(atom_types(ours)) == sorted(atom_types(reference))
    assert torsdof(ours) == torsdof(reference)


def test_sulfonamide_nitrogen_is_located():
    _, _, nitrogens = prepare_ligand(DRUGS["acetazolamide"])
    assert len(nitrogens) == 1

    _, _, nitrogens = prepare_ligand(DRUGS["caffeine"])
    assert nitrogens == [], "caffeine has no sulfonamide"


def test_amide_bond_is_rigid():
    # Acetazolamide: one amide (rigid) leaves 3 rotatable bonds, matching Meeko.
    pdbqt, _, _ = prepare_ligand(DRUGS["acetazolamide"])
    assert torsdof(pdbqt) == 3


def test_aromatic_and_polar_typing():
    mol = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1"))
    types = {a.GetIdx(): autodock_type(a) for a in mol.GetAtoms()}
    assert types[0] == "OA"
    assert types[1] == "A", "aromatic carbon should be type A"
    assert any(t == "HD" for t in types.values()), "phenol H is polar"


def test_salt_is_stripped_to_parent():
    # The sodium salt and the free acid must give the same heavy-atom count.
    salt, _, _ = prepare_ligand("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
    free, _, _ = prepare_ligand("CC(=O)Oc1ccccc1C(=O)[O-]")
    assert len(atom_types(salt)) == len(atom_types(free))


def test_unsupported_element_is_rejected():
    with pytest.raises(LigandPrepError, match="cannot type"):
        prepare_ligand("CC[Se]C")  # selenium has no AutoDock type


def test_invalid_smiles_is_rejected():
    with pytest.raises(LigandPrepError, match="invalid SMILES"):
        prepare_ligand("this is not a molecule")


def test_preparation_is_deterministic():
    first, _, _ = prepare_ligand(DRUGS["dorzolamide"])
    second, _, _ = prepare_ligand(DRUGS["dorzolamide"])
    assert first == second

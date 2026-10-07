"""Converts a docked PDBQT pose back into a chemically complete molecule.

A PDBQT pose carries only heavy atoms and polar hydrogens, and no bond orders.
Downstream tools (force-field parameterisation, PyMOL, SDF viewers) need both,
so the pose coordinates are mapped back onto the RDKit molecule the pose came
from and the non-polar hydrogens are rebuilt geometrically.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def pose_to_mol(smiles, pose_pdbqt, serial_to_idx, name=""):
    """Returns an RDKit molecule with explicit hydrogens at the docked pose.

    `serial_to_idx` is the mapping returned by ligand_prep.prepare_ligand, which
    relates PDBQT atom serials to atom indices of `Chem.AddHs(Chem.MolFromSmiles(smiles))`.
    """
    reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
    coords = {}
    for line in pose_pdbqt.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serial = int(line[6:11])
            coords[serial_to_idx[serial]] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))

    heavy = Chem.RemoveHs(reference)
    # RemoveHs keeps heavy atoms in their original relative order, so the
    # non-hydrogen indices of `reference` map onto `heavy` in sequence.
    heavy_indices = [a.GetIdx() for a in reference.GetAtoms() if a.GetAtomicNum() > 1]
    missing = [i for i in heavy_indices if i not in coords]
    if missing:
        raise ValueError(f"pose is missing {len(missing)} heavy atom(s)")

    conformer = Chem.Conformer(heavy.GetNumAtoms())
    for position, index in enumerate(heavy_indices):
        conformer.SetAtomPosition(position, Point3D(*coords[index]))
    heavy.RemoveAllConformers()
    heavy.AddConformer(conformer, assignId=True)

    Chem.AssignStereochemistryFrom3D(heavy)
    mol = Chem.AddHs(heavy, addCoords=True)
    if name:
        mol.SetProp("_Name", name)
    return mol


def write_pose_sdf(path, smiles, pose_pdbqt, serial_to_idx, name="", properties=None):
    """Writes the pose to an SDF file, optionally with scalar properties."""
    mol = pose_to_mol(smiles, pose_pdbqt, serial_to_idx, name=name)
    for key, value in (properties or {}).items():
        mol.SetProp(str(key), str(value))
    with Chem.SDWriter(path) as writer:
        writer.write(mol)
    return path


def heavy_atom_rmsd(mol_a, mol_b):
    """Symmetry-aware heavy-atom RMSD between two conformers of one molecule."""
    a, b = Chem.RemoveHs(mol_a), Chem.RemoveHs(mol_b)
    best = np.inf
    for match in a.GetSubstructMatches(b, uniquify=False, useChirality=False):
        pos_a = a.GetConformer().GetPositions()
        pos_b = b.GetConformer().GetPositions()
        deviations = pos_a[list(match)] - pos_b
        best = min(best, float(np.sqrt(np.mean(np.sum(deviations ** 2, axis=1)))))
    if not np.isfinite(best):
        raise ValueError("molecules do not share a substructure match")
    return best

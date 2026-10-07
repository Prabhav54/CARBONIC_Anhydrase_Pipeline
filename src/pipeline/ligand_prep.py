"""SMILES -> 3D ligand PDBQT for AutoDock Vina, using RDKit only.

Follows Meeko's default AutoDock typing and torsion-tree conventions (validated
against Meeko in tests/test_ligand_prep.py). Meeko itself is not used at
runtime because it pulls in SciPy and gemmi, which do not fit in the Vercel
function bundle alongside RDKit and Vina.
"""
from rdkit import Chem
from rdkit.Chem import AllChem

SUPPORTED_ELEMENTS = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}
POLAR_PARENTS = {"N", "O", "S"}


class LigandPrepError(ValueError):
    pass


def _is_polar_h(atom):
    return atom.GetSymbol() == "H" and atom.GetNeighbors()[0].GetSymbol() in POLAR_PARENTS


def _is_carbonyl_like(carbon):
    """Carbon double-bonded to O, S or N (amide / thioamide / amidine carbon)."""
    return carbon.GetSymbol() == "C" and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(carbon).GetSymbol() in ("O", "S", "N")
        for b in carbon.GetBonds()
    )


def _nitrogen_type(atom):
    """'NA' when the lone pair is available to accept an H-bond, else 'N'."""
    if atom.GetFormalCharge() > 0:
        return "N"
    if atom.GetIsAromatic():
        # Pyridine-like N accepts; pyrrole-like N donates its pair to the ring.
        return "NA" if atom.GetDegree() == 2 else "N"
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds()):
        return "NA"  # imine or nitrile: the pair is in-plane and available
    for nbr in atom.GetNeighbors():
        if nbr.GetIsAromatic() or _is_carbonyl_like(nbr):
            return "N"  # lone pair delocalised: aniline, amide
    return "NA"


def autodock_type(atom):
    symbol = atom.GetSymbol()
    if symbol == "C":
        return "A" if atom.GetIsAromatic() else "C"
    if symbol == "N":
        return _nitrogen_type(atom)
    if symbol == "O":
        return "OA"
    if symbol == "S":
        return "S" if atom.GetIsAromatic() or atom.GetTotalValence() > 2 else "SA"
    if symbol == "H":
        return "HD"
    return symbol  # P, F, Cl, Br, I


def _is_rotatable(bond, kept):
    if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
        return False
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    for atom in (a, b):
        if atom.GetHybridization() == Chem.HybridizationType.SP:
            return False
        # Rotation must move something besides merged (non-polar) hydrogens.
        if sum(1 for n in atom.GetNeighbors() if n.GetIdx() in kept) < 2:
            return False
    for c, n in ((a, b), (b, a)):
        # A true amide is planar because the nitrogen donates its lone pair to
        # the carbonyl, which needs that pair to be free: an imine nitrogen
        # (as in methazolamide) is already using it, so that bond does rotate.
        if (n.GetSymbol() == "N" and _is_carbonyl_like(c)
                and all(nb.GetBondType() == Chem.BondType.SINGLE for nb in n.GetBonds())):
            return False
    return True


def _embed(smiles, seed):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise LigandPrepError("invalid SMILES")
    # Dock only the parent compound of salts / mixtures.
    mol = max(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms())
    unsupported = {a.GetSymbol() for a in mol.GetAtoms()} - SUPPORTED_ELEMENTS
    if unsupported:
        raise LigandPrepError(f"Vina cannot type element(s): {', '.join(sorted(unsupported))}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise LigandPrepError("could not generate 3D coordinates")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    # Force-field setup kekulizes the molecule and leaves some aromatic flags
    # cleared (caffeine's carbonyl carbons, for one), which would mistype them.
    Chem.SanitizeMol(mol)
    AllChem.ComputeGasteigerCharges(mol)
    return mol


def _charges(mol, kept):
    """Gasteiger charges with non-polar hydrogens merged into their carbon."""
    charge = {i: mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge") for i in kept}
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in kept:
            parent = atom.GetNeighbors()[0].GetIdx()
            charge[parent] += atom.GetDoubleProp("_GasteigerCharge")
    return {i: (0.0 if c != c else c) for i, c in charge.items()}  # NaN -> 0


def _fragments(mol, kept, rotatable):
    """Rigid fragments (atom sets) and the fragment adjacency through rotatable bonds."""
    frag_of, frags = {}, []
    for start in sorted(kept):
        if start in frag_of:
            continue
        stack, members = [start], []
        frag_of[start] = len(frags)
        while stack:
            idx = stack.pop()
            members.append(idx)
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                j = nbr.GetIdx()
                if j in kept and j not in frag_of and frozenset((idx, j)) not in rotatable:
                    frag_of[j] = len(frags)
                    stack.append(j)
        frags.append(sorted(members))
    links = {f: [] for f in range(len(frags))}
    for bond in rotatable:
        i, j = tuple(bond)
        links[frag_of[i]].append((frag_of[j], i, j))
        links[frag_of[j]].append((frag_of[i], j, i))
    return frags, links


def _pick_root(frags, links):
    """Fragment that minimises torsion-tree depth (ties: the largest fragment)."""
    def depth(root):
        seen, frontier, d = {root}, [root], 0
        while frontier:
            frontier = [n for f in frontier for n, _, _ in links[f] if n not in seen]
            seen.update(frontier)
            d += bool(frontier)
        return d
    return min(range(len(frags)), key=lambda f: (depth(f), -len(frags[f])))


SULFONAMIDE_N = Chem.MolFromSmarts("[NX3,NX2-;$([N]S(=O)=O)]")


def prepare_ligand(smiles, seed=42):
    """Returns (pdbqt_string, serial_to_atom_idx, sulfonamide_nitrogen_indices).

    The nitrogen indices let the caller measure the zinc contact that defines
    carbonic anhydrase binding.
    """
    mol = _embed(smiles, seed)
    kept = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H" or _is_polar_h(a)}
    rotatable = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds() if _is_rotatable(b, kept)}
    charge = _charges(mol, kept)
    frags, links = _fragments(mol, kept, rotatable)
    pos = mol.GetConformer().GetPositions()

    lines, serial_of = [], {}

    def write_atoms(frag, first=None):
        # Vina expects a branch's axis atom to be listed first in the branch.
        order = sorted(frags[frag], key=lambda i: i != first)
        for idx in order:
            atom = mol.GetAtomWithIdx(idx)
            serial = len(serial_of) + 1
            serial_of[idx] = serial
            name = f"{atom.GetSymbol()}{serial}"[:4]
            x, y, z = pos[idx]
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} UNL     1    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00    {charge[idx]:6.3f} {autodock_type(atom):<2s}"
            )

    def write_branches(frag, parent):
        for child, here, there in links[frag]:
            if child == parent:
                continue
            lines.append(f"BRANCH {serial_of[here]:3d} {len(serial_of) + 1:3d}")
            write_atoms(child, first=there)
            write_branches(child, frag)
            lines.append(f"ENDBRANCH {serial_of[here]:3d} {serial_of[there]:3d}")

    root = _pick_root(frags, links)
    lines.append("ROOT")
    write_atoms(root)
    lines.append("ENDROOT")
    write_branches(root, None)
    lines.append(f"TORSDOF {len(rotatable)}")

    sulfonamide_n = [i for (i,) in mol.GetSubstructMatches(SULFONAMIDE_N) if i in serial_of]
    return "\n".join(lines) + "\n", {s: i for i, s in serial_of.items()}, sulfonamide_n

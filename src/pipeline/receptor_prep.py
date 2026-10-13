"""Turns any carbonic anhydrase PDB entry into a docking-ready receptor.

This used to be an offline step: `obabel` protonated and charged eight hand
picked structures and the results were committed. Open Babel is a conda package,
so that approach caps the app at whatever was prepared in advance. Here the same
work is done at request time with pip-installable tools only -- pdb2pqr for
protonation and charges, and plain parsing for AutoDock typing -- so any of the
twelve catalytic isoforms can be screened.

Two things the old offline pipeline got wrong are fixed by construction:

  * The docking box is centred on the catalytic zinc. The committed boxes came
    from a separate pocket search, and for five of the eight targets the box
    centre sat 16-24 A from the zinc -- outside the box entirely -- so no pose
    could ever have made the sulfonamide-zinc contact the scoring depends on.
  * The chain is chosen by which zinc is actually held by three histidines,
    rather than by proximity to that box centre, which is what multimeric
    entries need.

The zinc is stripped before protonation and re-attached afterwards: pdb2pqr has
no parameters for it, and AutoDock4Zn wants it typed ZN with zero charge anyway
(the zinc chemistry lives in the pair potentials, see zn_maps.py).
"""
import math
import os
import tempfile

from src.logger import logging
from src.pipeline.rcsb import StructureError, coordinates, describe

# A 20 A cube centred on the zinc. Wide enough for the active-site cone and the
# hydrophobic wall that selectivity depends on, small enough that AutoGrid stays
# inside a few seconds. The three CA II boxes of the previous pipeline, which is
# where the 1.6 A re-docking benchmark comes from, were exactly this.
BOX_SIZE = 20.0

# Zn-N(His) coordination is ~2.0-2.2 A; 2.8 A admits a slightly loose crystal
# structure without reaching the second shell.
COORDINATION_CUTOFF = 2.8
# Residues with an atom this close to the zinc are reported as lining the site.
POCKET_RADIUS = 8.0

# Ring atoms that AutoDock types as aromatic carbon (A) rather than aliphatic (C).
AROMATIC_RING_ATOMS = {
    "HIS": {"CG", "CD2", "CE1"},
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
}

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
    "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # pdb2pqr's protonation-state variants of histidine and cysteine.
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN", "TYM",
}

# Residue names that are histidine under any protonation state, for recognising
# the three-histidine zinc site.
HISTIDINES = {"HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"}


def _coords(line):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _element(line):
    """The element, from the PDB element column, falling back to the atom name."""
    element = line[76:78].strip().upper()
    if element:
        return element
    name = line[12:16].strip()
    return (name[1:2] if name[:1].isdigit() else name[:1]).upper()


def _residue_key(line):
    return (line[21], line[22:27].strip(), line[17:20].strip())


def _is_protein_atom(line):
    return line.startswith("ATOM  ") and line[17:20].strip() in AMINO_ACIDS


def _usable_altloc(line):
    return line[16] in (" ", "A")


def _zinc_atoms(lines):
    return [l for l in lines
            if l.startswith(("ATOM", "HETATM")) and _element(l) == "ZN" and _usable_altloc(l)]


def find_catalytic_zinc(lines):
    """Picks the zinc held by the three active-site histidines.

    Returns (zinc_line, chain_id, coordinating_residues). Multimeric entries
    carry one per chain; the first by chain order is used, and which one it was
    is reported so the app can say so.
    """
    zincs = _zinc_atoms(lines)
    if not zincs:
        raise StructureError(
            "This structure has no zinc. Carbonic anhydrase inhibitors bind by "
            "coordinating the catalytic zinc, so an apo structure cannot be docked.")

    protein = [l for l in lines if _is_protein_atom(l) and _usable_altloc(l)]
    candidates = []
    for zinc in zincs:
        centre = _coords(zinc)
        contacts = []
        for atom in protein:
            if _element(atom) not in ("N", "O"):
                continue
            distance = math.dist(_coords(atom), centre)
            if distance <= COORDINATION_CUTOFF:
                contacts.append((distance, atom))
        histidines = {_residue_key(a) for _, a in contacts if a[17:20].strip() in HISTIDINES}
        candidates.append((len(histidines), len(contacts), zinc, contacts))

    # Most coordinating histidines wins; ties break on chain order, which keeps
    # the choice deterministic across requests.
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2][21]))
    histidine_count, _, zinc, contacts = candidates[0]
    if histidine_count < 2:
        raise StructureError(
            "No zinc in this structure is held by the histidine triad of a "
            "carbonic anhydrase active site, so there is no site to dock into.")

    residues = []
    seen = set()
    for distance, atom in sorted(contacts, key=lambda c: c[0]):
        key = _residue_key(atom)
        if key in seen:
            continue
        seen.add(key)
        residues.append({
            "chain": key[0],
            "number": key[1],
            "name": key[2],
            "atom": atom[12:16].strip(),
            "distance": round(distance, 2),
        })

    # The zinc's own chain label is unreliable -- in some entries it is blank or
    # belongs to a solvent chain -- so the chain comes from what holds it.
    chain = max({r["chain"] for r in residues},
                key=lambda c: sum(1 for r in residues if r["chain"] == c))
    return zinc, chain, residues


def pocket_residues(lines, chain, zinc_centre):
    """Residues of the docked chain lining the active site, nearest first."""
    nearest = {}
    for atom in lines:
        if not _is_protein_atom(atom) or not _usable_altloc(atom) or atom[21] != chain:
            continue
        distance = math.dist(_coords(atom), zinc_centre)
        if distance > POCKET_RADIUS:
            continue
        key = _residue_key(atom)
        if key not in nearest or distance < nearest[key]:
            nearest[key] = distance
    return [{"chain": k[0], "number": k[1], "name": k[2], "distance": round(d, 2)}
            for k, d in sorted(nearest.items(), key=lambda kv: kv[1])]


def _protein_block(lines, chain):
    """The one chain pdb2pqr will protonate: no waters, ligands, ions or hydrogens."""
    kept = [l for l in lines
            if _is_protein_atom(l) and _usable_altloc(l) and l[21] == chain
            and _element(l) != "H"]
    if len(kept) < 100:
        raise StructureError(
            f"Chain {chain} has only {len(kept)} protein atoms, too few to "
            f"prepare a receptor from.")
    return "".join(kept) + "END\n"


def _pdb2pqr_flags():
    """The arguments pdb2pqr is run with, as a list.

    --noopt matters, and not for speed. pdb2pqr's hydrogen-bond optimiser flips
    the nearly symmetric histidine ring when it thinks that makes a better
    hydrogen bond, and it knows nothing about the zinc because the zinc has been
    stripped. On 3HS4 it rotates His94 so that ring carbon CE1 lands 2.0 A from
    the zinc in place of NE2, which destroys the histidine triad: AutoDock4Zn
    then finds no tetrahedral site and refuses to place its pseudo-atom. Docking
    into a crystal structure should not rearrange it, so the optimiser stays off.

    --titration-state-method=propka is left off deliberately: predicting
    titration states costs more than the rest of the request put together, and
    standard states are what the offline Open Babel step assumed too.
    """
    return ["--ff=AMBER", "--keep-chain", "--noopt"]


def _run_pdb2pqr(pdb_path, pqr_path):
    """Adds hydrogens and AMBER partial charges, in this process.

    pdb2pqr is called through its Python API rather than as a subprocess. The
    obvious `sys.executable -m pdb2pqr` does work locally but fails immediately
    in a serverless function, where sys.executable is not a Python the function
    can re-launch; docking then failed with nothing but "could not be
    protonated". Running in-process has no interpreter to find.
    """
    from pdb2pqr.main import build_main_parser, main_driver

    arguments = build_main_parser().parse_args(_pdb2pqr_flags() + [pdb_path, pqr_path])
    try:
        main_driver(arguments)
    except SystemExit as e:              # argparse/pdb2pqr error paths exit
        raise StructureError(
            "This structure could not be protonated for docking "
            f"(pdb2pqr exited with {e.code}).")
    except Exception as e:
        logging.warning(f"pdb2pqr failed on {os.path.basename(pdb_path)}: {e!r}")
        raise StructureError(
            "This structure could not be protonated for docking. It may have "
            "missing or non-standard residues in the active-site chain.")

    if not os.path.exists(pqr_path):
        raise StructureError("Protonating this structure produced no output.")
    with open(pqr_path) as f:
        return f.read()


def _parse_pqr(pqr_text):
    """PQR keeps PDB columns and appends charge and radius."""
    atoms = []
    for line in pqr_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        tail = line[54:].split()
        if len(tail) < 2:
            continue
        atoms.append({
            "name": line[12:16].strip(),
            "residue": line[17:20].strip(),
            "chain": line[21],
            # The residue sequence number and its insertion code are separate
            # PDB fields and have to stay that way. Merging them puts a letter
            # in the numeric column, and AutoDock4Zn's zinc_pseudo.py parses
            # that column with int() -- an entry with an insertion code (CA IV
            # 1ZNC, CA XII 1JCZ, CA XIII 3D0N all have one) then crashes it and
            # no zinc pseudo-atom is placed.
            "number": line[22:26].strip(),
            "icode": line[26],
            "xyz": _coords(line),
            "charge": float(tail[-2]),
            "element": _element(line) or line[12:16].strip()[:1].upper(),
        })
    if not atoms:
        raise StructureError("The protonated receptor came back empty.")
    return atoms


def _autodock_type(atom, has_polar_hydrogen):
    """AutoDock4 atom type for a protein atom.

    N vs NA and the aromatic-carbon A type are what the AD4 hydrogen-bond and
    aromatic terms key off, so they are assigned from the protonated structure
    rather than guessed from the residue name alone.
    """
    element, name, residue = atom["element"], atom["name"], atom["residue"]
    if element == "C":
        return "A" if name in AROMATIC_RING_ATOMS.get(residue[:3], ()) else "C"
    if element == "N":
        # A nitrogen carrying a hydrogen is a donor (N); one without a hydrogen
        # -- an unprotonated histidine ring nitrogen -- is an acceptor (NA).
        return "N" if has_polar_hydrogen else "NA"
    if element == "O":
        return "OA"
    if element == "S":
        return "SA"
    if element == "H":
        return "HD"
    return element


def _pdbqt_line(serial, atom, autodock_type, charge):
    """One receptor atom in PDBQT columns.

    PDBQT is PDB with the partial charge in columns 71-76 and the AutoDock type
    in 78-79, which is where Vina and AutoGrid read them from, so the layout has
    to be exact: atom name 13-16, altLoc 17, residue 18-20, chain 22,
    residue number 23-26, insertion code 27.
    """
    name = atom["name"][:4].ljust(4) if len(atom["name"]) >= 4 else f" {atom['name']:<3}"
    x, y, z = atom["xyz"]
    return (f"ATOM  {serial:5d} {name} {atom['residue'][:3]:>3} {atom['chain']}"
            f"{atom['number']:>4}{atom.get('icode') or ' '}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    "
            f"{charge:+6.3f} {autodock_type:<2}\n")


# No hydrogen sits this close to a zinc. One that does is an artefact of
# protonating a metal-coordinating residue with the metal stripped out.
ZINC_HYDROGEN_CLASH = 1.6


def _to_pdbqt(atoms, zinc_centre):
    """Writes the AutoDock receptor: polar hydrogens kept, non-polar merged.

    AutoDock4 is a united-atom force field for non-polar hydrogens: their charge
    is folded into the carbon they sit on and the hydrogen itself is dropped.
    Hydrogens on N, O and S stay, typed HD, because the hydrogen-bond term needs
    them.

    The exception is the zinc's own coordination shell. pdb2pqr protonates the
    histidine triad as if it were free, because the zinc was removed before
    protonation, and leaves a hydrogen about 1 A from where the metal sits. In
    the enzyme those nitrogens are deprotonated and donate their lone pair to
    the zinc, so such hydrogens are dropped and their charge folded back into
    the nitrogen, which then types as an acceptor.
    """
    heavy = [a for a in atoms if a["element"] != "H"]
    hydrogens = [a for a in atoms if a["element"] == "H"]

    # Bond each hydrogen to the nearest heavy atom of its own residue.
    by_residue = {}
    for index, atom in enumerate(heavy):
        by_residue.setdefault((atom["chain"], atom["number"]), []).append(index)

    merged_charge = [a["charge"] for a in heavy]
    polar = {}
    deprotonated = 0
    for hydrogen in hydrogens:
        siblings = by_residue.get((hydrogen["chain"], hydrogen["number"]), [])
        if not siblings:
            continue
        parent = min(siblings, key=lambda i: math.dist(heavy[i]["xyz"], hydrogen["xyz"]))
        in_zinc_shell = math.dist(hydrogen["xyz"], zinc_centre) < ZINC_HYDROGEN_CLASH
        if heavy[parent]["element"] in ("N", "O", "S") and not in_zinc_shell:
            polar.setdefault(parent, []).append(hydrogen)
        else:
            # Non-polar hydrogens and the zinc-shell artefacts both fold into
            # their parent, which leaves the latter typed as an acceptor.
            merged_charge[parent] += hydrogen["charge"]
            deprotonated += in_zinc_shell
    if deprotonated:
        logging.info(f"dropped {deprotonated} hydrogen(s) from the zinc coordination shell")

    lines, serial = [], 1
    for index, atom in enumerate(heavy):
        autodock_type = _autodock_type(atom, has_polar_hydrogen=index in polar)
        lines.append(_pdbqt_line(serial, atom, autodock_type, merged_charge[index]))
        serial += 1
        for hydrogen in polar.get(index, []):
            lines.append(_pdbqt_line(serial, hydrogen, "HD", hydrogen["charge"]))
            serial += 1
    return lines


def _zinc_pdbqt_line(serial, zinc_line):
    """The zinc, typed for AutoDock4Zn: element ZN, zero charge."""
    x, y, z = _coords(zinc_line)
    return (f"ATOM  {serial:5d} ZN    ZN {zinc_line[21]}"
            f"{zinc_line[22:26].strip():>4}{zinc_line[26]}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000 ZN\n")


# Every AutoDock type this module emits. AutoGrid reads the type from columns
# 78-79 and the charge from 71-76; a one-column slip there is silent and poisons
# every map, so the written file is checked against this set before it is used.
RECEPTOR_TYPES = {"C", "A", "N", "NA", "OA", "SA", "HD", "ZN"}


def _validate_pdbqt(lines):
    """Rejects a receptor whose type or charge columns did not land where AutoGrid looks."""
    types = {l[77:79].strip() for l in lines}
    unknown = types - RECEPTOR_TYPES
    if unknown:
        raise StructureError(f"Receptor typing produced unknown AutoDock types: "
                             f"{sorted(unknown)}")
    total = sum(float(l[70:76]) for l in lines)
    if not any(float(l[70:76]) for l in lines):
        raise StructureError("The prepared receptor carries no partial charges.")
    # A protein chain's formal charge is a small integer; anything far outside
    # that means the charge column is being read from the wrong place.
    if abs(total) > 25:
        raise StructureError(f"The prepared receptor has an implausible net charge "
                             f"({total:+.1f}); the PDBQT columns are misaligned.")
    return total


class Receptor:
    """A prepared receptor: the PDBQT, the box it is docked in, and its site."""

    def __init__(self, pdb_id, path, zinc, chain, coordinating, pocket, structure):
        self.pdb_id = pdb_id
        self.path = path
        self.zinc = zinc
        self.chain = chain
        self.coordinating_residues = coordinating
        self.pocket_residues = pocket
        self.structure = structure
        self.box_size = (BOX_SIZE, BOX_SIZE, BOX_SIZE)

    @property
    def box_centre(self):
        """The box is centred on the catalytic zinc."""
        return self.zinc

    def site(self):
        """What the app shows as the docking site for this structure."""
        return {
            "pdb_id": self.pdb_id,
            "chain": self.chain,
            "zinc": [round(c, 3) for c in self.zinc],
            "box_centre": [round(c, 3) for c in self.box_centre],
            "box_size": list(self.box_size),
            "coordinating_residues": self.coordinating_residues,
            "pocket_residues": self.pocket_residues,
        }


def prepare(pdb_id, workdir):
    """Builds the receptor for `pdb_id` inside `workdir`.

    Raises StructureError, with a message meant for the user, for anything that
    makes the structure undockable.
    """
    structure = describe(pdb_id)
    pdb_id = structure["pdb_id"]
    lines = coordinates(pdb_id).splitlines(keepends=True)

    zinc, chain, coordinating = find_catalytic_zinc(lines)
    centre = _coords(zinc)
    pocket = pocket_residues(lines, chain, centre)

    os.makedirs(workdir, exist_ok=True)
    protein_path = os.path.join(workdir, f"{pdb_id}_chain{chain}.pdb")
    pqr_path = os.path.join(workdir, f"{pdb_id}.pqr")
    with open(protein_path, "w") as f:
        f.write(_protein_block(lines, chain))

    atoms = _parse_pqr(_run_pdb2pqr(protein_path, pqr_path))
    pdbqt_lines = _to_pdbqt(atoms, centre)
    net_charge = _validate_pdbqt(pdbqt_lines)
    pdbqt_lines.append(_zinc_pdbqt_line(len(pdbqt_lines) + 1, zinc))

    receptor_path = os.path.join(workdir, f"{pdb_id}.pdbqt")
    staged = receptor_path + ".partial"
    with open(staged, "w") as f:
        f.writelines(pdbqt_lines)
    os.replace(staged, receptor_path)

    logging.info(f"{pdb_id}: prepared chain {chain}, {len(pdbqt_lines)} atoms, "
                 f"net charge {net_charge:+.1f}, {len(coordinating)} zinc contacts")
    return Receptor(pdb_id, receptor_path, centre, chain, coordinating, pocket, structure)


def prepare_in_temp(pdb_id):
    """Convenience wrapper for scripts and tests: prepares into a temp directory."""
    return prepare(pdb_id, tempfile.mkdtemp(prefix=f"ca_{pdb_id}_"))


def inspect_site(pdb_id):
    """The docking site of a structure, without preparing it for docking.

    Finding the zinc and the residues around it needs only the coordinates, so
    screening can show where docking would happen in about a second, instead of
    paying the ~25 s of protonation and map building for a molecule the user may
    never dock.
    """
    structure = describe(pdb_id)
    lines = coordinates(structure["pdb_id"]).splitlines(keepends=True)
    zinc, chain, coordinating = find_catalytic_zinc(lines)
    centre = _coords(zinc)
    return {
        "pdb_id": structure["pdb_id"],
        "chain": chain,
        "zinc": [round(c, 3) for c in centre],
        "box_centre": [round(c, 3) for c in centre],
        "box_size": [BOX_SIZE, BOX_SIZE, BOX_SIZE],
        "coordinating_residues": coordinating,
        "pocket_residues": pocket_residues(lines, chain, centre),
        "zinc_count": len(_zinc_atoms(lines)),
    }

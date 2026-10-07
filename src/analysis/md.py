"""Molecular dynamics of a docked carbonic anhydrase complex (OpenMM).

Docking scores a single rigid pose. MD asks a different question: once the
protein and ligand can move and water is present, does the pose survive?

Offline / notebook use only (see notebook/05_MD_stability.ipynb). Needs the
conda-forge stack `openmm openmmforcefields openff-toolkit pdbfixer mdtraj`,
none of which is installed in the deployed app.

Zinc is modelled as Amber's non-bonded Zn(2+) ion: a point charge plus a
Lennard-Jones term. Sulfonamide-zinc binding has real covalent character, so
this force field systematically under-binds that contact. Treat a lengthening
N-Zn distance as a limitation of the model, not as proof the pose is wrong.
"""
import os

import numpy as np

HYDROGEN_MASS = 3.0          # amu; heavy hydrogens allow the 4 fs timestep
TIMESTEP_PS = 0.004
TEMPERATURE_K = 300.0
EQUILIBRATION_STEPS = 2500   # 10 ps
FRAMES_PER_RUN = 100


def extract_receptor(pdb_path, chain=None, out_path=None):
    """Writes one protein chain plus its catalytic zinc, dropping waters and ligands.

    With `chain=None` the chain of the first zinc found is used.
    """
    protein, zinc = [], None
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM") and line[17:20] == " ZN":
                if chain is None or line[21] == chain:
                    chain, zinc = line[21], zinc or line
            elif line.startswith("ATOM  ") and line[16] in (" ", "A"):
                protein.append(line)
    if zinc is None:
        raise ValueError(f"{pdb_path}: no catalytic zinc found")

    kept = [l for l in protein if l[21] == chain]
    out_path = out_path or os.path.splitext(pdb_path)[0] + f"_{chain}_receptor.pdb"
    with open(out_path, "w") as f:
        f.writelines(kept)
        f.write(zinc)
        f.write("END\n")
    return out_path, chain


def build_complex(receptor_pdb, ligand_sdf, padding_nm=1.0, ionic_strength_molar=0.15):
    """Protonates the receptor, adds the ligand and solvates.

    Returns (modeller, system_generator, openff_molecule).
    """
    from openff.toolkit import Molecule
    from openmm import app, unit
    from openmmforcefields.generators import SystemGenerator
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=receptor_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    ligand = Molecule.from_file(ligand_sdf)
    if ligand.n_conformers == 0:
        raise ValueError(f"{ligand_sdf}: no 3D conformer (the docked pose) in the file")

    # Zn(2+) parameters live in the water/ion file, not in amber14-all.
    generator = SystemGenerator(
        forcefields=["amber14-all.xml", "amber14/tip3p.xml"],
        small_molecule_forcefield="gaff-2.11",
        molecules=[ligand],
        forcefield_kwargs={
            "constraints": app.HBonds,
            "rigidWater": True,
            "removeCMMotion": False,
            "hydrogenMass": HYDROGEN_MASS * unit.amu,
        },
    )

    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.add(ligand.to_topology().to_openmm(), ligand.conformers[0].to_openmm())
    modeller.addSolvent(
        generator.forcefield, model="tip3p",
        padding=padding_nm * unit.nanometer,
        ionicStrength=ionic_strength_molar * unit.molar,
    )
    return modeller, generator, ligand


def run_simulation(modeller, generator, ligand, nanoseconds, out_dir,
                   platform_name=None, minimisation_steps=1000, progress=None):
    """Minimises, equilibrates and runs production MD.

    Returns {'topology_pdb', 'trajectory_dcd', 'log_csv', 'steps', 'ns_per_day'}.
    """
    import time

    import openmm
    from openmm import MonteCarloBarostat, app, unit

    os.makedirs(out_dir, exist_ok=True)
    topology_pdb = os.path.join(out_dir, "complex.pdb")
    trajectory_dcd = os.path.join(out_dir, "trajectory.dcd")
    log_csv = os.path.join(out_dir, "simulation_log.csv")

    system = generator.create_system(modeller.topology, molecules=[ligand])
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, 1.0 / unit.picosecond, TIMESTEP_PS * unit.picoseconds)

    platform = openmm.Platform.getPlatformByName(platform_name) if platform_name else None
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    if progress:
        progress("minimising")
    simulation.minimizeEnergy(maxIterations=minimisation_steps)

    # The minimised complex is the reference frame for all RMSD measurements.
    state = simulation.context.getState(getPositions=True)
    with open(topology_pdb, "w") as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f, keepIds=True)

    if progress:
        progress("equilibrating")
    simulation.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
    simulation.step(EQUILIBRATION_STEPS)

    # Constant pressure for production so the box density relaxes.
    system.addForce(MonteCarloBarostat(1.0 * unit.bar, TEMPERATURE_K * unit.kelvin, 25))
    simulation.context.reinitialize(preserveState=True)

    steps = max(FRAMES_PER_RUN, int(round(nanoseconds * 1000 / TIMESTEP_PS)))
    interval = max(1, steps // FRAMES_PER_RUN)
    simulation.reporters.append(app.DCDReporter(trajectory_dcd, interval))
    simulation.reporters.append(app.StateDataReporter(
        log_csv, interval, step=True, time=True, potentialEnergy=True,
        temperature=True, density=True, speed=True))

    if progress:
        progress(f"production: {steps} steps ({nanoseconds} ns)")
    started = time.time()
    simulation.step(steps)
    elapsed = time.time() - started

    return {
        "topology_pdb": topology_pdb,
        "trajectory_dcd": trajectory_dcd,
        "log_csv": log_csv,
        "steps": steps,
        "frames": steps // interval,
        "wall_seconds": round(elapsed, 1),
        "ns_per_day": round(nanoseconds * 86400 / elapsed, 1) if elapsed else None,
        "platform": simulation.context.getPlatform().getName(),
    }


def analyse_trajectory(topology_pdb, trajectory_dcd, ligand_resname="UNK"):
    """Per-frame stability metrics for the docked ligand.

    Returns a dict of arrays: time_ps, ligand_rmsd, backbone_rmsd,
    zinc_distance (shortest sulfonamide N to Zn) and hydrogen_bonds
    (ligand-protein count), plus the ligand/zinc atom selections used.
    """
    import mdtraj as md

    traj = md.load(trajectory_dcd, top=topology_pdb)
    traj.superpose(traj, 0, atom_indices=traj.topology.select("protein and backbone"))

    ligand_atoms = traj.topology.select(f"resname {ligand_resname} and not element H")
    if len(ligand_atoms) == 0:
        residues = sorted({r.name for r in traj.topology.residues})
        raise ValueError(f"no residue named {ligand_resname}; found {residues}")
    zinc_atoms = traj.topology.select("element Zn")
    backbone = traj.topology.select("protein and backbone")

    # Sulfonamide nitrogens: an N bonded to a sulfur inside the ligand.
    sulfurs = {a.index for a in traj.topology.atoms if a.index in set(ligand_atoms) and a.element.symbol == "S"}
    bonded = {}
    for a, b in traj.topology.bonds:
        bonded.setdefault(a.index, set()).add(b.index)
        bonded.setdefault(b.index, set()).add(a.index)
    sulfonamide_n = [i for i in ligand_atoms
                     if traj.topology.atom(i).element.symbol == "N" and bonded.get(i, set()) & sulfurs]

    # Ligand drift must be measured in the frame of the binding site, so it is
    # computed directly on the backbone-superposed coordinates. md.rmsd() would
    # re-superpose on the ligand itself and report its internal conformational
    # change instead, which is a different (and here misleading) quantity.
    displacement = traj.xyz[:, ligand_atoms, :] - traj.xyz[0, ligand_atoms, :]
    ligand_rmsd = np.sqrt((displacement ** 2).sum(axis=-1).mean(axis=-1)) * 10  # nm -> A

    # For the protein, superposing on the backbone is the conventional measure.
    backbone_rmsd = md.rmsd(traj, traj, 0, atom_indices=backbone) * 10

    zinc_distance = None
    if len(zinc_atoms) and sulfonamide_n:
        pairs = np.array([[n, zinc_atoms[0]] for n in sulfonamide_n])
        zinc_distance = md.compute_distances(traj, pairs).min(axis=1) * 10

    # Hydrogen bonds are searched on a protein+ligand slice: including ~12,000
    # waters makes this dominate the whole analysis and adds nothing, since only
    # ligand-protein bonds are counted. wernet_nilsson gives per-frame results,
    # which baker_hubbard (a frequency filter over the whole run) does not.
    complex_only = traj.atom_slice(traj.topology.select(f"protein or resname {ligand_resname}"))
    sliced_ligand = set(complex_only.topology.select(f"resname {ligand_resname}"))
    sliced_protein = set(complex_only.topology.select("protein"))
    hydrogen_bonds = [
        sum(1 for donor, _, acceptor in frame_bonds
            if ({donor, acceptor} & sliced_ligand) and ({donor, acceptor} & sliced_protein))
        for frame_bonds in md.wernet_nilsson(complex_only)
    ]

    return {
        "time_ps": traj.time if traj.time is not None else np.arange(traj.n_frames, dtype=float),
        "ligand_rmsd": ligand_rmsd,
        "backbone_rmsd": backbone_rmsd,
        "zinc_distance": zinc_distance,
        "hydrogen_bonds": np.array(hydrogen_bonds),
        "ligand_atoms": ligand_atoms,
        "sulfonamide_nitrogens": sulfonamide_n,
        "n_frames": traj.n_frames,
    }


def summarise(metrics, equilibration_fraction=0.2):
    """Averages the metrics over the post-equilibration part of the run."""
    start = int(len(metrics["ligand_rmsd"]) * equilibration_fraction)
    window = slice(start, None)

    summary = {
        "frames_analysed": int(len(metrics["ligand_rmsd"]) - start),
        "ligand_rmsd_mean": round(float(np.mean(metrics["ligand_rmsd"][window])), 2),
        "ligand_rmsd_max": round(float(np.max(metrics["ligand_rmsd"][window])), 2),
        "backbone_rmsd_mean": round(float(np.mean(metrics["backbone_rmsd"][window])), 2),
        "hydrogen_bonds_mean": round(float(np.mean(metrics["hydrogen_bonds"][window])), 2),
    }
    if metrics["zinc_distance"] is not None:
        distances = metrics["zinc_distance"][window]
        summary["zinc_distance_mean"] = round(float(np.mean(distances)), 2)
        summary["zinc_distance_final"] = round(float(distances[-1]), 2)
        summary["zinc_contact_fraction"] = round(float(np.mean(distances <= 2.5)), 2)
    return summary

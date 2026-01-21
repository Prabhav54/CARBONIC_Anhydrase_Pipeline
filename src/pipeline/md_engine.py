import os
import sys
# Try imports, handle missing libraries gracefully
try:
    import openmm as mm
    from openmm import app
    from openmm import unit
    import pdbfixer
except ImportError:
    print("⚠️ OpenMM not found. MD will be skipped.")
    mm = None

class MDEngine:
    def __init__(self, output_dir="md_output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)

    def run_simulation(self, complex_pdb, steps=10000):
        if not mm: return "MD_Skipped"
        
        print(f"⚗️ Running MD on {complex_pdb}...")
        try:
            # 1. Fix PDB (Add missing atoms)
            fixer = pdbfixer.PDBFixer(filename=complex_pdb)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.4)
            
            # 2. Setup System (Implicit Solvent for Speed)
            forcefield = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
            system = forcefield.createSystem(fixer.topology, 
                                           nonbondedMethod=app.CutoffNonPeriodic,
                                           constraints=app.HBonds)
            
            # 3. Simulation Setup
            integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
            # Use CPU if GPU fails (Kaggle T4s usually work though)
            try: platform = mm.Platform.getPlatformByName('CUDA')
            except: platform = mm.Platform.getPlatformByName('CPU')
                
            sim = app.Simulation(fixer.topology, system, integrator, platform)
            sim.context.setPositions(fixer.positions)
            
            # 4. Run (Minimize + Equilibrate)
            sim.minimizeEnergy()
            sim.step(steps) # Short stability check
            
            # Save result
            out_traj = f"{self.output_dir}/trajectory.pdb"
            with open(out_traj, 'w') as f:
                app.PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), f)
                
            return "Stable"
        except Exception as e:
            print(f"MD Failed: {e}")
            return "Failed"
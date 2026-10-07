# CA-Predictor

Virtual screening for **carbonic anhydrase (CA)** inhibitors. Paste SMILES strings,
pick a target structure, and get predicted potency, drug-likeness and — on demand —
real AutoDock Vina docking against the CA active site.

Built as a Flask app that deploys to Vercel as a single Python function.

---

## What it actually does

| Stage | Method | Where it runs |
|---|---|---|
| Potency (pIC50) | Random Forest + LSTM ensemble (0.7 / 0.3) on 1024-bit Morgan fingerprints | instantly, every candidate |
| Drug-likeness | RDKit: Lipinski Rule of 5, QED, TPSA, LogP | instantly, every candidate |
| Reference drugs | live ChEMBL mechanism-of-action lookup per isoform | instantly |
| Docking | AutoDock Vina with the **AutoDock4Zn** force field | on demand, one molecule per click |

### Supported targets

| PDB | Isoform | Relevance | ChEMBL target |
|---|---|---|---|
| 1CA2, 3HS4, 4ZAO | CA II | glaucoma, diuretics | CHEMBL205 |
| 3IAI, 5FL4, 5FL6 | CA IX | hypoxic tumours | CHEMBL3594 |
| 1JCZ, 5JN9 | CA XII | tumours | CHEMBL3242 |

---

## Honest limits

These matter if you intend to rely on the numbers.

**Docking does not always rank the native pose first.** Re-docking the
co-crystallised acetazolamide into 3HS4 recovers the crystal pose to
**1.6 Å RMSD** with the expected **2.2 Å** sulfonamide-N–Zn contact, but that
pose comes **second**; a pose 4.2 Å away scores slightly better. The app
therefore shows the top three poses with each one's zinc distance rather than a
single headline score. A pose that does not coordinate the zinc is probably not
the real binding mode, however well it scores.

**Plain Vina scoring is unusable for this protein.** Without the zinc term the
same re-docking lands **5.5 Å** from the crystal pose. That is why the
AutoDock4Zn maps exist; they are not an optional refinement.

**Potency predictions are interpolation, not discovery.** The model was trained
on ChEMBL IC50 data that includes many known CA inhibitors. A high score for a
marketed sulfonamide is a sanity check, not a finding. There is no scaffold-split
validation here, so expect optimistic numbers for chemistry resembling the
training set.

**The receptor is rigid and solvent is implicit.** Standard docking
approximations: no side-chain flexibility, no explicit waters, no entropy of
binding. Scores are a ranking aid, not binding free energies.

**One chain, one zinc.** Only the chain whose zinc is nearest the pocket centre
is kept. Interface or allosteric effects in the multimeric structures (3IAI,
5FL4, 5FL6, 5JN9) are not modelled.

---

## Running it locally

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows
pip install -r requirements.txt
python app.py                                          # http://127.0.0.1:5000
```

The committed artifacts (`artifacts/receptors/`, `artifacts/models/ensemble.npz`)
are all the app needs — no training step required.

Command line:

```bash
python main.py --target 3HS4 "CC(=O)Nc1nnc(s1)S(=O)(=O)N" "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
python main.py --target 5FL6 --smiles-file candidates.txt --dock --top 3
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"     # fast: ligand prep, model parity, routes
pytest -m slow           # docking, including the re-docking benchmark (~30 s)
```

---

## Deploying to Vercel

```bash
npm i -g vercel
vercel deploy
```

`vercel.json` sets 300 s `maxDuration` (docking takes 10–90 s per molecule) and
2 GB memory, and excludes the large offline data from the function bundle.
`.vercelignore` keeps that data out of the upload as well. Deployed size is about
**393 MB** of the 500 MB Python limit: 336 MB of dependencies plus 57 MB of
receptors, zinc maps and the model.

Two deployment constraints shaped the code:

- **No scikit-learn or TensorFlow at runtime.** They would not fit alongside
  RDKit and Vina, so `src/components/model_exporter.py` flattens the trained
  models into one 7.5 MB NumPy archive and `src/pipeline/potency_model.py`
  scores them with NumPy alone. Verified equal to the originals to **1.6e-14**
  (forest) and **1.7e-06** (LSTM) — see `tests/test_potency_model.py`.
- **No Meeko and no RDKit depiction.** Meeko needs SciPy and gemmi; RDKit's
  drawing module needs X11 system libraries that the serverless runtime lacks.
  So `src/pipeline/ligand_prep.py` writes PDBQT with RDKit only (atom types and
  torsion counts match Meeko exactly — `tests/test_ligand_prep.py`), and 2D
  structures are drawn in the browser.

### Plan limits

With Fluid compute (on by default), **Hobby allows 300 s `maxDuration`** — the
same default and maximum as Pro — so the docking route fits without a paid plan.
Hobby does cap memory at **2 GB / 1 vCPU**, which is why `vercel.json` requests
2048 MB; asking for more is rejected. Vina is configured for a single CPU
anyway, so the quoted 10–90 s docking times are what Hobby delivers.

If a large, flexible ligand still approaches the limit, set
`VINA_EXHAUSTIVENESS=4` as a project environment variable to halve the search
time at some cost in pose quality.

---

## Layout

```
app.py                      Flask routes: /, /predict, /dock
main.py                     command-line screening
src/pipeline/
  targets.py                supported PDB structures -> isoform + ChEMBL id
  potency_model.py          NumPy RF + LSTM ensemble, Morgan fingerprints
  predict_pipeline.py       screening: potency, ADMET, ChEMBL references
  ligand_prep.py            SMILES -> 3D PDBQT (RDKit only)
  docking_engine.py         Vina with AutoDock4Zn maps, pose + zinc distances
  admet_engine.py           Lipinski and QED
  pose_io.py                docked pose -> SDF / RMSD helpers
src/components/             training: ingestion, transformation, trainer, exporter
src/analysis/md.py          OpenMM molecular dynamics (offline, conda only)
data_curation_scripts/
  prepare_receptors.py      PDB -> charged receptor PDBQT (needs obabel)
  build_zinc_maps.py        AutoDock4Zn affinity maps (needs autogrid4 4.2.7)
  ad4zn/                    vendored AD4Zn force field (Apache-2.0, see its README)
  fetch_*.py, pocket_detection.py, ...   dataset construction
notebook/                   01 EDA, 02 training, 03 screening, 04 docking,
                            05 MD stability, 06 safety, 07 selectivity
tests/
```

### Rebuilding the docking inputs

Only needed if you change the target list. Both steps need conda packages that
are not pip-installable:

```bash
conda install -c conda-forge openbabel autogrid
python -m data_curation_scripts.prepare_receptors
python -m data_curation_scripts.build_zinc_maps     # ~90 s for 8 targets
```

### Molecular dynamics (notebook 05)

Docking scores one rigid pose; MD checks whether the pose survives once
everything can move. `notebook/05_MD_stability.ipynb` runs explicit-solvent
OpenMM on the docked complex and plots ligand RMSD, backbone RMSD, the
sulfonamide-N–Zn distance and hydrogen-bond counts.

```bash
conda create -n ca-md -c conda-forge python=3.11 openmm openmmforcefields \
    openff-toolkit pdbfixer mdtraj rdkit matplotlib
```

Two things to know before reading its output:

- **Measured throughput is about 2 ns/day on 4 CPU cores** for this 37,800-atom
  system. The notebook defaults to a 20 ps run that exercises the pipeline and
  produces real plots in minutes; a stability claim needs 10 ns or more, which
  means a CUDA GPU.
- **Zinc is a non-bonded ion** in Amber, so this force field under-binds the
  sulfonamide–zinc contact by construction. A lengthening N–Zn distance is a
  limitation of the model, not evidence against the pose.

Running the notebook as shipped shows why that second point matters: the docked
pose starts with the sulfonamide nitrogen 1.9 � from the zinc, and **energy
minimisation alone moves it to 3.2 �** before any dynamics run. The ligand stays
in the pocket (hydrogen bonds hold, backbone RMSD 0.66 �) but the coordination is
lost immediately. That is a verdict on the force field, not on acetazolamide.

No binding free energies are reported: MM-PBSA from a single short trajectory
is not trustworthy, and the previous version of this project presented exactly
such numbers as if they were measurements.

### Retraining

```bash
pip install -r requirements-dev.txt
python -m src.pipeline.train_pipeline        # writes artifacts/*.pkl, *.h5
python -m src.components.model_exporter      # -> artifacts/models/ensemble.npz
```

The app reads only `ensemble.npz`, so re-export after any retrain.

---

## Credits

Docking uses [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina) with
the AutoDock4Zn force field (Santos-Martins, Forli, Ramos & Olson,
*J. Chem. Inf. Model.* 2014, **54**, 2371–2379). Bioactivity data from
[ChEMBL](https://www.ebi.ac.uk/chembl/); structures from the
[RCSB PDB](https://www.rcsb.org/). Cheminformatics with [RDKit](https://www.rdkit.org/).

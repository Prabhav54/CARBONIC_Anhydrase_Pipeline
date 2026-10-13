# CA-Predictor

Virtual screening for **carbonic anhydrase (CA)** inhibitors. Type the PDB ID of
any catalytic human CA structure, paste SMILES strings, and get predicted
potency scored out of 10, drug-likeness, the binding site that will be docked
into, and (on demand) real AutoDock Vina docking with the AutoDock4Zn force
field.

Built as a Flask app that deploys to Vercel as a single Python function.

**Live:** https://carbonic-anhydrase-pipeline.vercel.app

Measured on that deployment: screening returns in 8 s for a CA II structure and
19 s for one whose ChEMBL target has more to fetch; a structure's docking site
resolves in 2 s; docking one molecule takes 58 to 60 s, including preparing the
receptor and building its zinc maps the first time. Acetazolamide into 3HS4
returns four poses, three of which coordinate the zinc, the best-scoring one at
1.94 A.

---

## What it actually does

The page is two steps, because the target decides the answer and a single
combined form hid that. **Step one** takes a PDB ID and shows what the structure
is, what inhibiting its isoform is for, the drugs ChEMBL records against it, the
genetic disease UniProt annotates for the gene, and an interactive 3D view of
the protein with the docking site, the histidine triad and the search box drawn
on it. **Step two** takes molecules and scores them.

| Stage | Method | Where it runs |
|---|---|---|
| Structure lookup | RCSB entry -> UniProt accession -> CA isoform; rejects anything else | step 1, ~2 s |
| Target context | curated role, ChEMBL mechanism drugs, UniProt disease annotations | step 1 |
| Binding site, in 3D | catalytic zinc, histidine triad, pocket lining within 8 A, search box | step 1 |
| Potency (0-10) | Random Forest + LSTM ensemble (0.7 / 0.3) on 1024 Morgan bits **plus four descriptors of the target protein** | step 2 |
| Drug-likeness | RDKit: all four Lipinski rules with measured values, QED, TPSA | step 2 |
| Known inhibitors | ChEMBL target resolved from that isoform's accession, plus the ligand co-crystallised in that entry | step 2 |
| Docking | receptor prepared and AutoDock4Zn maps built for that structure, then Vina | on demand, 45-60 s |

### Which structures

Any PDB entry whose polymer maps to one of the **twelve catalytically active
human isoforms** and which contains a zinc held by the histidine triad. The
isoform is read from the entry, so there is no list of supported structures to
maintain. Non-CA entries are rejected by name ("1HHO is not a human carbonic
anhydrase. It is *Structure of human oxyhaemoglobin*...").

| Isoform | UniProt | Measured molecules | Test R2 | Potency |
|---|---|---|---|---|
| CA I | P00915 | 7,325 | 0.62 | scored |
| CA II | P00918 | 8,353 | 0.64 | scored |
| CA IV | P22748 | 1,489 | 0.65 | scored |
| CA VA | P35218 | 550 | 0.46 | scored |
| CA VB | Q9Y2D0 | 404 | 0.54 | scored |
| CA VII | P43166 | 1,324 | 0.57 | scored |
| CA IX | Q16790 | 6,007 | 0.63 | scored |
| CA XII | O43570 | 4,730 | 0.62 | scored |
| CA XIII | Q8N1Q1 | 131 | 0.71 | scored |
| CA XIV | Q9ULX7 | 620 | 0.66 | scored |
| CA VI | P23280 | 217 | 0.16 | **withheld** |
| CA III | P07451 | 103 | 0.06 | **withheld** |

"Withheld" means the app docks the structure, shows its site and lists its real
known inhibitors, but prints no potency number. The gate is **measured accuracy,
not data volume**: an isoform is scored only if it has enough measured molecules
*and* the ensemble reaches R2 0.30 on held-out molecules of that isoform
(`data_curation_scripts/build_isoform_features.py`). CA III has 103 molecules,
which would clear any row-count threshold, but R2 0.06 on its held-out
molecules, so a score out of 10 for it would be noise with a number on it.

**Ten of the twelve are scored now; six were before.** CA XII and CA VII used to
report "no bioactivity data", which was wrong: the data existed, the curation
step had just never fetched it. See the retraining note below.

CA VIII, X and XI are in the family but lack the zinc triad, so they are not
offered at all.

### What the score out of 10 means

The percentile of the predicted pIC50 within the pIC50 values actually measured
against **that isoform** in ChEMBL. 7.0 means "predicted more potent than 70% of
known inhibitors of this isoform". The isoforms are not comparable on a single
axis, which is why the reference distribution is per isoform: the measured
median is 7.6 against CA IV and 5.5 against CA VA.

---

## Honest limits

These matter if you intend to rely on the numbers.

**Potency predictions are interpolation, not discovery.** On a split grouped by
molecule, the ensemble reaches **R2 0.671, MAE 0.487** across 31,253
measurements and twelve isoforms. Per isoform it ranges from 0.46 (CA VA) to
0.71 (CA XIII). There is no scaffold-split validation, so expect optimistic
numbers for chemistry resembling the training set. A high score for a marketed
sulfonamide is a sanity check, not a finding.

**Affinities pool IC50 and Ki.** Carbonic anhydrase inhibition is overwhelmingly
reported as Ki from stopped-flow assays: of the 31,253 measurements, about 96%
are Ki. Restricting to IC50 would discard most of the literature, so both are
used as ChEMBL's pChEMBL value and the column is named `pAffinity` rather than
`pIC50` so nothing claims more than that.

**The target genuinely changes the prediction.** Dropping the four protein
descriptors takes test R2 from **0.671 to 0.415**: on a twelve-isoform dataset
the ligand alone cannot explain the variance, because the same molecule is
measured against many isoforms with different results. Still, only twelve
distinct protein vectors exist, so the protein input is an isoform embedding
rather than a general model of CA binding sites, and it cannot generalise to an
isoform it has never seen.

Note the overall R2 fell from 0.76 to 0.671 when the dataset grew. That is not a
regression: the earlier number came from nine isoforms and 8,825 measurements
dominated by ligand-only signal, while MAE actually improved from 0.51 to 0.487.

**Docking does not always rank the coordinating pose first.** Re-docking the
crystal acetazolamide into 3HS4 recovers the pose to **1.7 A**, and there the
top-ranked pose does coordinate the zinc at 1.9 A. For 1JCZ (CA XII) and 1ZNC
(CA IV) the best-scoring pose sits 5 A from the zinc while poses 2 to 5
coordinate it properly. The app therefore shows several poses with each one's
zinc distance and contact residues rather than a single headline score. A pose
that does not coordinate the zinc is probably not the real binding mode, however
well it scores.

**Plain Vina scoring is unusable for this protein.** Without the zinc term the
same re-docking lands 5.5 A from the crystal pose. That is why the AutoDock4Zn
maps exist; they are not an optional refinement.

**The receptor is rigid and solvent is implicit.** No side-chain flexibility, no
explicit waters, no entropy of binding. Scores are a ranking aid, not binding
free energies.

**One chain, one zinc.** Only the chain whose zinc is held by three histidines is
kept. Interface and allosteric effects in multimeric entries are not modelled.

**Crystal structures are taken as given.** Heavy atoms are never moved, which is
deliberate (see the `--noopt` note below), so a structure with a distorted or
partially occupied active site is docked as deposited.

---

## Running it locally

Docking needs Linux: the `vina` wheel and the vendored AutoGrid binary are
Linux-only. Screening works anywhere.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python app.py                                          # http://127.0.0.1:5000
```

Command line:

```bash
python main.py --list-isoforms
python main.py --target 3HS4 "CC(=O)Nc1nnc(s1)S(=O)(=O)N" "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
python main.py --target 5FL6 --smiles-file candidates.txt --dock --top 3
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not slow and not network"   # fast: ligand prep, model parity, routes
pytest -m network                      # live RCSB, UniProt and ChEMBL lookups
pytest -m slow                         # receptor prep, map building, docking (~10 min)
```

---

## How docking works for an arbitrary structure

This is the part that changed most, and the reason the app is no longer limited
to eight hand-prepared targets.

The previous version prepared receptors offline with the `obabel` CLI and built
AutoDock4Zn maps offline with `autogrid4`, then committed 50 MB of gzipped maps
for eight structures. Neither tool is pip-installable, so the deployed app could
only ever dock what had been prepared in advance. Now, per request:

1. **Fetch** the entry and its coordinates from RCSB (cached on disk).
2. **Find the catalytic zinc** by looking for the zinc held by three histidines,
   and keep that chain. The box is centred on that zinc.
3. **Protonate and charge** the chain with `pdb2pqr` (pip-installable), then
   write PDBQT with AutoDock atom types directly. Vina reads charges and types
   from fixed columns, so the layout is checked before use.
4. **Build the AD4Zn maps** with a vendored 117 KB AutoGrid 4.2.9 binary
   (~13 s), cached in `/tmp` with the oldest map sets evicted.
5. **Dock** with Vina, exhaustiveness 16, up to 5 poses, reporting each pose's
   zinc distance and contact residues.

Total for a structure's first molecule: about 25 s of preparation plus 30 to 50 s
of search, inside the 300 s function budget. Later molecules against the same
structure skip straight to the search.

### Three bugs this fixed

Worth recording, because all three were silent:

- **The docking box did not contain the zinc.** The committed boxes came from a
  separate pocket search. For five of the eight targets (3IAI, 5FL4, 5FL6, 1JCZ,
  5JN9) the box centre sat 16 to 24 A from the kept zinc, outside a 20 A box
  entirely, in a region holding 130 to 228 atoms rather than the ~490 of a real
  buried pocket. No pose against those five could ever have made the
  sulfonamide-zinc contact the whole method depends on. Only the three CA II
  structures were ever right, which is why the old README only quoted 3HS4.
- **pdb2pqr flipped the histidine ring.** Its hydrogen-bond optimiser rotates
  the nearly symmetric imidazole when it judges that a better hydrogen bond, and
  it knows nothing about the zinc because the zinc is stripped before
  protonation. On 3HS4 it put ring carbon CE1 2.0 A from the zinc in place of
  NE2, destroying the triad, after which AutoDock4Zn finds no tetrahedral site
  and places no pseudo-atom. Hence `--noopt`: docking into a crystal structure
  should not rearrange it.
- **Insertion codes broke the zinc pseudo-atom step.** 1ZNC, 1JCZ and 3D0N
  contain residues such as `50A`. Writing that into the 4-character numeric
  residue column makes `zinc_pseudo.py` fail on `int(' 50A')`, so CA IV, CA XII
  and CA XIII silently produced no maps at all.

---

## Deploying to Vercel

```bash
npm i -g vercel
vercel deploy --prod
```

The project is linked as `prabhav54s-projects/carbonic-anhydrase-pipeline`.

`vercel.json` sets 300 s `maxDuration`, which is what docking needs and what
Hobby allows with Fluid compute. It no longer sets `memory`: the build warns that
the setting is ignored under Active CPU billing. Hobby caps at 1 vCPU, which is
what Vina is configured for.

The deployed function is **253 MB** of the 500 MB Python limit: dependencies plus
the 7.5 MB model, the 3.4 MB AutoGrid bundle and the isoform table. Only 43 files
are uploaded; everything offline is excluded by `.vercelignore` and
`excludeFiles`.

Three deployment constraints shaped the code:

- **No scikit-learn or TensorFlow at runtime.** They would not fit alongside
  RDKit and Vina, so `src/components/model_exporter.py` flattens the trained
  models into one 7.5 MB NumPy archive and `src/pipeline/potency_model.py`
  scores them with NumPy alone. Verified equal to the originals to 1e-12
  (forest) and 1e-4 (LSTM) in `tests/test_potency_model.py`.
- **No Open Babel and no AutoGrid from pip.** Receptor preparation moved to
  `pdb2pqr` plus direct PDBQT writing, and AutoGrid is vendored as a Linux
  binary with its three GCC runtime libraries under `src/bin/` (3.4 MB total,
  stripped). `.vercelignore` deliberately does **not** exclude `src/bin`.
- **No Meeko and no RDKit depiction.** Meeko needs SciPy and gemmi; RDKit's
  drawing module needs X11 libraries the runtime lacks. So
  `src/pipeline/ligand_prep.py` writes ligand PDBQT with RDKit only (atom types
  and torsion counts match Meeko exactly, `tests/test_ligand_prep.py`), and 2D
  structures are drawn in the browser.

---

## Layout

```
app.py                      Flask routes: /, /structure, /predict, /dock, /site/<id>
main.py                     command-line screening
src/pipeline/
  isoforms.py               the twelve catalytic human isoforms and their support
  structure_view.py         step one: everything shown about a target
  rcsb.py                   PDB ID -> isoform, metadata, coordinates, ligands
  chembl.py                 known inhibitors of one isoform, with provenance
  uniprot.py                genetic disease annotations for an isoform
  receptor_prep.py          PDB entry -> receptor PDBQT, box and site
  zn_maps.py                AutoDock4Zn maps at request time (vendored AutoGrid)
  docking_engine.py         Vina, poses with zinc distance and contact residues
  potency_model.py          NumPy RF + LSTM ensemble, fingerprint + protein features
  scoring.py                predicted pIC50 -> percentile out of 10
  predict_pipeline.py       screening: potency, ADMET, references
  admet_engine.py           Lipinski and QED
  ligand_prep.py            SMILES -> 3D PDBQT (RDKit only)
  pose_io.py                docked pose -> SDF / RMSD helpers
  ad4zn/                    vendored AD4Zn force field (Apache-2.0)
src/bin/linux-x86_64/       AutoGrid 4.2.9 + libstdc++, libgomp, libgcc_s (GPL-2.0)
src/components/             training: ingestion, transformation, trainer, exporter
src/analysis/md.py          OpenMM molecular dynamics (offline, conda only)
data_curation_scripts/
  fetch_chembl_bioactivity.py   rebuilds the training set from ChEMBL
  build_isoform_features.py     the isoform table the app serves
  fetch_*.py, feature_eng.py    earlier dataset construction
notebook/                   01 EDA, 02 training, 03 screening, 04 docking,
                            05 MD stability, 06 safety, 07 selectivity
tests/
```

### Retraining

```bash
pip install -r requirements-dev.txt
python -m data_curation_scripts.fetch_chembl_bioactivity  # rebuild the dataset
python -m src.pipeline.train_pipeline                     # trains, reports, exports
python -m data_curation_scripts.build_isoform_features    # rebuilds the isoform table
```

`fetch_chembl_bioactivity` is why ten isoforms are scored instead of six. The
previous training set held 8,825 measurements over nine isoforms and nothing at
all for CA VB, CA VII and CA XII, so the app reported "no bioactivity data" for
CA XII, a major tumour-associated target, while ChEMBL held 5,403 measurements
against it. Refetching gives 31,253 measurements over 9,536 molecules and all
twelve isoforms:

| | before | after |
|---|---|---|
| measurements | 8,825 | 31,253 |
| distinct molecules | 8,172 | 9,536 |
| isoforms with any data | 9 | 12 |
| isoforms scored | 6 | 10 |
| CA XII molecules | 0 | 4,730 |
| CA VII molecules | 0 | 1,324 |

`train_pipeline` splits **by molecule**, not by row. A plain random split leaks:
8,825 measurements cover only 8,172 distinct molecules because 653 were measured
against more than one isoform, so the same molecule landed in both train and
test and the reported R2 was partly a memorisation score. It also writes
`artifacts/models/isoform_metrics.json`, which decides which isoforms may be
scored at all.

The app reads only `ensemble.npz` and `isoforms.json`, so re-export and rebuild
the table after any retrain.

### Molecular dynamics (notebook 05)

Docking scores one rigid pose; MD checks whether the pose survives once
everything can move. `notebook/05_MD_stability.ipynb` runs explicit-solvent
OpenMM on the docked complex and plots ligand RMSD, backbone RMSD, the
sulfonamide-N-Zn distance and hydrogen-bond counts.

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
  sulfonamide-zinc contact by construction. A lengthening N-Zn distance is a
  limitation of the model, not evidence against the pose.

Running the notebook as shipped shows why that second point matters: the docked
pose starts with the sulfonamide nitrogen 1.9 A from the zinc, and **energy
minimisation alone moves it to 3.2 A** before any dynamics run. The ligand stays
in the pocket (hydrogen bonds hold, backbone RMSD 0.66 A) but the coordination is
lost immediately. That is a verdict on the force field, not on acetazolamide.

No binding free energies are reported: MM-PBSA from a single short trajectory is
not trustworthy, and an earlier version of this project presented exactly such
numbers as if they were measurements.

---

## Credits

Docking uses [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina) with
the AutoDock4Zn force field (Santos-Martins, Forli, Ramos & Olson,
*J. Chem. Inf. Model.* 2014, **54**, 2371-2379). Grid maps are built with
[AutoGrid 4.2.9](https://autodock.scripps.edu/) (GPL-2.0; see
`src/bin/linux-x86_64/COPYING.autogrid`). Receptor protonation and charges with
[pdb2pqr](https://github.com/Electrostatics/pdb2pqr). Bioactivity data from
[ChEMBL](https://www.ebi.ac.uk/chembl/); structures and chemical components from
the [RCSB PDB](https://www.rcsb.org/); isoform sequences from
[UniProt](https://www.uniprot.org/). Cheminformatics with
[RDKit](https://www.rdkit.org/).

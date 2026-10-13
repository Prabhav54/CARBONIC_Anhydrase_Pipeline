"""Builds artifacts/models/isoforms.json: the isoform table the app serves.

The retrained ensemble takes the four per-protein descriptors as input features
alongside the ligand fingerprint, so inference has to reproduce the exact values
used in training. Biopython is a development dependency, not a runtime one, so
the descriptors are computed here, once, and shipped as JSON.

The same file carries, per isoform:
  * the measured affinity distribution (as 101 quantile breakpoints), which
    src/pipeline/scoring.py turns into the 0-10 score, and
  * how many measurements that distribution rests on, so the app can say when a
    prediction has no support rather than printing a confident number.

Run from the project root (needs biopython and pandas):
    python -m data_curation_scripts.build_isoform_features
"""
import json
import os

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from src.components.data_transformation import affinity_column

METADATA_CSV = os.path.join("artifacts", "data", "isoform_metadata.csv")
TRAIN_CSV = os.path.join("artifacts", "train.csv")
TEST_CSV = os.path.join("artifacts", "test.csv")
METRICS_PATH = os.path.join("artifacts", "models", "isoform_metrics.json")
OUT_PATH = os.path.join("artifacts", "models", "isoforms.json")

# The twelve catalytically active human isoforms, by UniProt accession. CA VIII,
# X and XI are in the family but have no catalytic zinc site (they lack one or
# more of the three coordinating histidines), and CA5BP1 is a pseudogene, so
# none of them can be docked and none are offered.
#
# `gene` matches the Isoform_Name column of isoform_metadata.csv and the
# `target_name` labels in the training data are the roman-numeral `label`s.
# `where` is what and where the protein is; `therapeutic` is why anyone would
# inhibit it. The second is deliberately conservative, and the app pairs it with
# two things that are fetched rather than written here: the drugs ChEMBL
# annotates as acting on this target, and the genetic disease UniProt annotates
# for the gene. Those are different claims from "inhibiting this treats X" and
# the UI keeps them apart.
CATALYTIC_ISOFORMS = {
    "P00915": {
        "label": "CA I", "gene": "CA1",
        "where": "cytosolic; the high-abundance erythrocyte isoform",
        "therapeutic": "Rarely the intended target. It matters mostly as a "
                       "selectivity anti-target: a sulfonamide that also inhibits "
                       "CA I is less selective for whatever it was designed for.",
    },
    "P00918": {
        "label": "CA II", "gene": "CA2",
        "where": "cytosolic; the most studied isoform",
        "therapeutic": "The established target of the clinical carbonic anhydrase "
                       "inhibitors, used for glaucoma and ocular hypertension, as "
                       "diuretics, for altitude sickness, and as adjuncts in "
                       "epilepsy. It is also the main anti-target when designing "
                       "for the tumour-associated isoforms.",
    },
    "P07451": {
        "label": "CA III", "gene": "CA3",
        "where": "cytosolic, skeletal muscle; markedly lower turnover than CA II",
        "therapeutic": "Not an established drug target. Studied mainly as a marker "
                       "of muscle damage and oxidative stress.",
    },
    "P22748": {
        "label": "CA IV", "gene": "CA4",
        "where": "GPI-anchored to the membrane; kidney, lung and eye",
        "therapeutic": "Contributes to the renal and ocular effects of systemically "
                       "administered inhibitors, so it shapes the side-effect "
                       "profile of drugs aimed at other isoforms.",
    },
    "P35218": {
        "label": "CA VA", "gene": "CA5A",
        "where": "mitochondrial, liver; ureagenesis and gluconeogenesis",
        "therapeutic": "Inhibition reduces lipogenesis, which is the proposed basis "
                       "for the weight loss seen with the anticonvulsants topiramate "
                       "and zonisamide, both of which inhibit carbonic anhydrases.",
    },
    "Q9Y2D0": {
        "label": "CA VB", "gene": "CA5B",
        "where": "mitochondrial, more widely distributed than CA VA",
        "therapeutic": "Shares the mitochondrial, anti-obesity rationale of CA VA.",
    },
    "P23280": {
        "label": "CA VI", "gene": "CA6",
        "where": "the only secreted isoform; saliva and milk",
        "therapeutic": "Studied in taste perception and oral physiology rather than "
                       "as a drug target.",
    },
    "P43166": {
        "label": "CA VII", "gene": "CA7",
        "where": "cytosolic, brain",
        "therapeutic": "Implicated in neuronal excitability and proposed as a target "
                       "in epilepsy, where CA inhibition is already clinically "
                       "established through other isoforms.",
    },
    "Q16790": {
        "label": "CA IX", "gene": "CA9",
        "where": "transmembrane, induced by hypoxia; tumour-associated",
        "therapeutic": "The most actively pursued anticancer carbonic anhydrase "
                       "target: it is strongly expressed in hypoxic solid tumours "
                       "and largely absent from normal tissue, and inhibiting it "
                       "disrupts the pH regulation those tumours depend on. A "
                       "ureido-sulfonamide inhibitor has reached clinical trials.",
    },
    "O43570": {
        "label": "CA XII", "gene": "CA12",
        "where": "transmembrane; tumour-associated",
        "therapeutic": "Frequently co-expressed with CA IX in tumours and usually "
                       "considered alongside it. The design problem is selectivity "
                       "over the abundant cytosolic CA I and CA II.",
    },
    "Q8N1Q1": {
        "label": "CA XIII", "gene": "CA13",
        "where": "cytosolic; reproductive and gastrointestinal tissues",
        "therapeutic": "Not an established drug target; relevant as a selectivity "
                       "anti-target.",
    },
    "Q9ULX7": {
        "label": "CA XIV", "gene": "CA14",
        "where": "transmembrane; brain, kidney and retina",
        "therapeutic": "Studied in retinal and neuronal physiology; not an "
                       "established drug target.",
    },
}

# An isoform is scored only if it clears BOTH bars.
#
# The count bar is obvious: a handful of measurements cannot support a
# percentile. The R2 bar is the one that matters, and it is why this script
# reads the training metrics. CA XIII has 117 measured molecules, which clears
# any sensible count, but the ensemble's R2 on its held-out molecules is about
# 0.0 -- the model cannot rank CA XIII inhibitors at all. Scoring it anyway
# would dress noise up as a number out of 10.
MIN_MEASUREMENTS = 100
MIN_TEST_R2 = 0.30


def descriptors(sequence):
    """The four protein features, exactly as data_curation_scripts/feature_eng.py."""
    analysed = ProteinAnalysis(sequence)
    return {
        "hydrophobicity": analysed.gravy(),
        "weight": analysed.molecular_weight(),
        "isoelectric": analysed.isoelectric_point(),
        "aromaticity": analysed.aromaticity(),
    }


def measured_affinity(label, measurements, column):
    """Median affinity per distinct molecule measured against this isoform."""
    rows = measurements[measurements["target_name"] == label]
    if rows.empty:
        return np.array([])
    return rows.groupby("clean_smiles")[column].median().to_numpy()


def main():
    metadata = pd.read_csv(METADATA_CSV).set_index("UniProt_ID")
    measurements = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)])
    if not os.path.exists(METRICS_PATH):
        raise SystemExit(
            f"{METRICS_PATH} is missing. Run `python -m src.pipeline.train_pipeline` "
            f"first: which isoforms may be scored depends on the model's measured "
            f"accuracy on each of them.")
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    affinity = affinity_column(measurements)

    table = {}
    for accession, entry in CATALYTIC_ISOFORMS.items():
        label, gene = entry["label"], entry["gene"]
        row = metadata.loc[accession]
        assert row["Isoform_Name"] == gene, f"{accession}: expected {gene}, got {row['Isoform_Name']}"
        sequence = row["Sequence"]
        features = descriptors(sequence)

        # Inference must reproduce the training features bit for bit. Where this
        # isoform appears in the training data, check against what was used.
        training_rows = measurements[measurements["target_name"] == label]
        if not training_rows.empty:
            for key, column in (("hydrophobicity", "Protein_Hydrophobicity"),
                                ("weight", "Protein_Weight"),
                                ("isoelectric", "Protein_Isoelectric"),
                                ("aromaticity", "Protein_Aromaticity")):
                trained = training_rows[column].to_numpy()
                assert np.allclose(trained, features[key], rtol=0, atol=1e-9), (
                    f"{label}: recomputed {key} {features[key]!r} does not match "
                    f"the training value {trained[0]!r}")

        pic50 = measured_affinity(label, measurements, affinity)
        accuracy = metrics.get("per_isoform", {}).get(label, {})
        r2 = accuracy.get("r2")
        enough_data = len(pic50) >= MIN_MEASUREMENTS
        accurate = r2 is not None and r2 >= MIN_TEST_R2
        quantiles = (np.percentile(pic50, np.arange(101)).round(4).tolist()
                     if enough_data and accurate else None)
        table[accession] = {
            "label": label,
            "gene": gene,
            "role": entry["where"],
            "therapeutic": entry["therapeutic"],
            "sequence_length": int(row["Seq_Length"]),
            "descriptors": features,
            "measurements": int(len(pic50)),
            "test_r2": r2,
            "test_mae": accuracy.get("mae"),
            "test_n": accuracy.get("n_test"),
            "pic50_quantiles": quantiles,
        }
        support = f"{len(pic50)} molecules" if len(pic50) else "no measurements"
        if quantiles:
            verdict = f"scored (R2 {r2:.2f})"
        elif not enough_data:
            verdict = "WITHHELD: too few measurements"
        else:
            verdict = f"WITHHELD: R2 {r2:.2f} below {MIN_TEST_R2}" if r2 is not None \
                else "WITHHELD: never tested"
        print(f"{accession} {label:8s} {support:16s} {verdict}")

    # The family-wide distribution, used only to describe where an isoform's
    # own distribution sits; never as a substitute for a missing one.
    family = measurements.groupby(["clean_smiles", "target_name"])[affinity].median()
    payload = {
        "min_measurements": MIN_MEASUREMENTS,
        "family_pic50_quantiles": np.percentile(family.to_numpy(), np.arange(101)).round(4).tolist(),
        "family_measurements": int(len(family)),
        "isoforms": table,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n-> {OUT_PATH} ({len(table)} isoforms, "
          f"{sum(1 for v in table.values() if v['pic50_quantiles'])} scored)")


if __name__ == "__main__":
    main()

"""Rebuilds the training set from ChEMBL, for all twelve catalytic CA isoforms.

Why this exists: the previous training set held 8,825 measurements over nine
isoforms and had nothing at all for CA VB, CA VII and CA XII. The app therefore
told users "no bioactivity data for this isoform" for CA XII, a major
tumour-associated target, while ChEMBL holds over five thousand measurements
against it. The gap was in the curation step, not in the literature.

What it fetches: every IC50 and Ki measurement with a pChEMBL value against each
isoform's single-protein ChEMBL target, as an exact relation in nM.

Two choices worth knowing about:

  * **IC50 and Ki are pooled.** Most carbonic anhydrase inhibition in the
    literature is reported as Ki from stopped-flow assays, so restricting to
    IC50 would throw away the majority of the data. ChEMBL's pChEMBL value is
    -log10(molar) for either, and the pooled column is named `pAffinity` rather
    than `pIC50` so nothing downstream claims it is all IC50. The mix per
    isoform is printed and recorded.
  * **Duplicates are aggregated by median** per (molecule, isoform). The same
    compound is often measured many times against CA II by different groups.

Run from the project root (needs biopython, pandas, rdkit and network access):
    python -m data_curation_scripts.fetch_chembl_bioactivity
"""
import collections
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from rdkit import Chem, RDLogger

API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 1000
TIMEOUT = 90
RETRIES = 5

METADATA_CSV = os.path.join("artifacts", "data", "isoform_metadata.csv")
OUT_CSV = os.path.join("artifacts", "data", "clean_training_data.csv")
SUMMARY_JSON = os.path.join("artifacts", "data", "bioactivity_summary.json")

# label -> (UniProt accession, gene). The twelve catalytically active human
# isoforms; CA VIII, X and XI lack the zinc triad and are not inhibitor targets.
ISOFORMS = {
    "CA I": ("P00915", "CA1"), "CA II": ("P00918", "CA2"),
    "CA III": ("P07451", "CA3"), "CA IV": ("P22748", "CA4"),
    "CA VA": ("P35218", "CA5A"), "CA VB": ("Q9Y2D0", "CA5B"),
    "CA VI": ("P23280", "CA6"), "CA VII": ("P43166", "CA7"),
    "CA IX": ("Q16790", "CA9"), "CA XII": ("O43570", "CA12"),
    "CA XIII": ("Q8N1Q1", "CA13"), "CA XIV": ("Q9ULX7", "CA14"),
}

# Outside this range a reported affinity is almost always a transcription error
# or a limit-of-assay value rather than a measurement.
MIN_PAFFINITY, MAX_PAFFINITY = 3.0, 12.5

RDLogger.DisableLog("rdApp.*")


def _get(path, params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "ca-predictor-curation"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            if attempt + 1 == RETRIES:
                raise
            time.sleep(2 * (attempt + 1))
            print(f"    retrying after {type(e).__name__}")


def resolve_target(accession):
    """The single-protein ChEMBL target for a UniProt accession."""
    payload = _get("target.json", {"target_components__accession": accession, "limit": 20})
    single = [t for t in payload.get("targets", [])
              if t.get("target_type") == "SINGLE PROTEIN"]
    if not single:
        raise SystemExit(f"ChEMBL has no single-protein target for {accession}")
    return single[0]["target_chembl_id"]


def fetch_activities(chembl_id):
    """Every usable IC50/Ki measurement against one target, paginated."""
    rows, offset = [], 0
    while True:
        payload = _get("activity.json", {
            "target_chembl_id": chembl_id,
            "standard_type__in": "IC50,Ki",
            "standard_relation": "=",
            "standard_units": "nM",
            "pchembl_value__isnull": "false",
            "only": "molecule_chembl_id,canonical_smiles,pchembl_value,standard_type",
            "limit": PAGE_SIZE,
            "offset": offset,
        })
        batch = payload.get("activities", [])
        rows += batch
        total = payload["page_meta"]["total_count"]
        offset += PAGE_SIZE
        print(f"    {min(offset, total)}/{total}")
        if offset >= total or not batch:
            return rows


def clean_smiles(smiles):
    """Canonical SMILES of the largest fragment, or None.

    Salts and co-formers are stripped by keeping the largest fragment, which is
    what makes one molecule comparable across the assays that measured it.
    """
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None
    largest = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    if largest.GetNumHeavyAtoms() < 4:
        return None
    return Chem.MolToSmiles(largest)


def descriptors(sequence):
    """The four protein features, exactly as data_curation_scripts/feature_eng.py."""
    analysed = ProteinAnalysis(sequence)
    return {
        "Protein_Hydrophobicity": analysed.gravy(),
        "Protein_Weight": analysed.molecular_weight(),
        "Protein_Isoelectric": analysed.isoelectric_point(),
        "Protein_Aromaticity": analysed.aromaticity(),
    }


def main():
    metadata = pd.read_csv(METADATA_CSV).set_index("UniProt_ID")

    frames, summary = [], {}
    for label, (accession, gene) in ISOFORMS.items():
        chembl_id = resolve_target(accession)
        print(f"{label} ({accession} -> {chembl_id}):")
        activities = fetch_activities(chembl_id)

        records, kinds = [], collections.Counter()
        for activity in activities:
            smiles = clean_smiles(activity.get("canonical_smiles"))
            value = activity.get("pchembl_value")
            if not smiles or value is None:
                continue
            value = float(value)
            if not MIN_PAFFINITY <= value <= MAX_PAFFINITY:
                continue
            kinds[activity.get("standard_type")] += 1
            records.append({"clean_smiles": smiles, "pAffinity": value})

        if not records:
            print(f"  no usable measurements")
            summary[label] = {"chembl_target": chembl_id, "molecules": 0,
                              "measurements": 0, "types": {}}
            continue

        frame = pd.DataFrame(records)
        # One row per molecule: the median of however many times it was measured.
        frame = frame.groupby("clean_smiles", as_index=False)["pAffinity"].median()
        frame["target_name"] = label

        row = metadata.loc[accession]
        assert row["Isoform_Name"] == gene, f"{accession}: expected {gene}"
        for column, value in descriptors(row["Sequence"]).items():
            frame[column] = value

        frames.append(frame)
        summary[label] = {
            "chembl_target": chembl_id,
            "molecules": int(len(frame)),
            "measurements": int(sum(kinds.values())),
            "types": dict(kinds),
            "median_pAffinity": round(float(frame["pAffinity"].median()), 3),
        }
        print(f"  {len(frame)} molecules from {sum(kinds.values())} measurements "
              f"({dict(kinds)})")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset[["clean_smiles", "target_name", "pAffinity",
                       "Protein_Hydrophobicity", "Protein_Weight",
                       "Protein_Isoelectric", "Protein_Aromaticity"]]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    dataset.to_csv(OUT_CSV, index=False)

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'isoform':9s} {'molecules':>10s} {'measurements':>13s} {'median':>7s}")
    for label, stats in summary.items():
        print(f"{label:9s} {stats['molecules']:10d} {stats['measurements']:13d} "
              f"{stats.get('median_pAffinity', 0):7.2f}")
    print(f"\n-> {OUT_CSV}: {len(dataset)} rows, "
          f"{dataset['clean_smiles'].nunique()} distinct molecules, "
          f"{dataset['target_name'].nunique()} isoforms")


if __name__ == "__main__":
    main()

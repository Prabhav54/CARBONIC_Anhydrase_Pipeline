"""Virtual screening: predict potency and drug-likeness for a pool of molecules.

Docking is deliberately not part of this step. It takes tens of seconds per
molecule, so the app runs it on demand through src.pipeline.docking_engine.
"""
import urllib.parse

import requests

from src.logger import logging
from src.pipeline import admet_engine
from src.pipeline.potency_model import get_model, fingerprints
from src.pipeline.targets import DEFAULT_TARGET, TARGETS

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_TIMEOUT = 5
MAX_REFERENCE_DRUGS = 6

# Used when the ChEMBL API cannot be reached.
FALLBACK_DRUGS = [
    {"Molecule_Name": "Acetazolamide", "clean_smiles": "CC(=O)Nc1nnc(s1)S(=O)(=O)N"},
    {"Molecule_Name": "Methazolamide", "clean_smiles": "CC(=O)/N=c1/sc(S(N)(=O)=O)nn1C"},
    {"Molecule_Name": "Dorzolamide", "clean_smiles": "CCN[C@H]1C[C@@H](C)S(=O)(=O)c2sc(cc12)S(N)(=O)=O"},
    {"Molecule_Name": "Brinzolamide", "clean_smiles": "CCN[C@H]1CN(CCCOC)S(=O)(=O)c2sc(cc12)S(N)(=O)=O"},
    {"Molecule_Name": "Ethoxzolamide", "clean_smiles": "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"},
]


def resolve_target(query):
    """Maps a user-typed PDB ID to a supported target, or None if unsupported."""
    return query.strip().upper() if query.strip().upper() in TARGETS else None


def pubchem_link(smiles):
    return f"https://pubchem.ncbi.nlm.nih.gov/#query={urllib.parse.quote(smiles)}"


def fetch_reference_drugs(pdb_id):
    """Known inhibitors of this target's isoform, from ChEMBL's mechanism data."""
    chembl_id = TARGETS[pdb_id]["chembl_id"]
    try:
        mechanisms = requests.get(
            f"{CHEMBL_API}/mechanism.json",
            params={"target_chembl_id": chembl_id, "limit": 50},
            timeout=CHEMBL_TIMEOUT,
        ).json()
        molecule_ids = sorted({m["molecule_chembl_id"] for m in mechanisms.get("mechanisms", [])})
        if not molecule_ids:
            raise ValueError(f"ChEMBL lists no mechanisms for {chembl_id}")

        molecules = requests.get(
            f"{CHEMBL_API}/molecule.json",
            params={"molecule_chembl_id__in": ",".join(molecule_ids[:MAX_REFERENCE_DRUGS])},
            timeout=CHEMBL_TIMEOUT,
        ).json()

        drugs = []
        for molecule in molecules.get("molecules", []):
            smiles = (molecule.get("molecule_structures") or {}).get("canonical_smiles")
            if not smiles:
                continue
            name = molecule.get("pref_name") or molecule["molecule_chembl_id"]
            drugs.append({"Molecule_Name": name.title(), "clean_smiles": smiles})
        if not drugs:
            raise ValueError("no ChEMBL molecule had a structure")
        logging.info(f"fetched {len(drugs)} reference drugs for {chembl_id}")
        return drugs, "ChEMBL API"
    except Exception as e:
        logging.warning(f"ChEMBL lookup failed for {chembl_id}: {e}")
        return list(FALLBACK_DRUGS), "offline reference set"


def screen(candidates, pdb_id=DEFAULT_TARGET):
    """Scores candidates and reference drugs.

    `candidates` is a list of {'Molecule_Name', 'clean_smiles'}.
    Returns (references, novels, reference_source, skipped) with both lists
    ranked by predicted potency.
    """
    references, reference_source = fetch_reference_drugs(pdb_id)
    pool = [dict(c, is_novel=True) for c in candidates] + [dict(r, is_novel=False) for r in references]

    seen, unique = set(), []
    for entry in pool:
        if entry["clean_smiles"] not in seen:
            seen.add(entry["clean_smiles"])
            unique.append(entry)

    X, valid = fingerprints([e["clean_smiles"] for e in unique])
    skipped = [unique[i]["Molecule_Name"] for i in range(len(unique)) if i not in set(valid)]
    if not valid:
        return [], [], reference_source, skipped

    predictions = get_model().predict(X)

    rows = []
    for position, index in enumerate(valid):
        entry = unique[index]
        smiles = entry["clean_smiles"]
        pic50 = float(predictions[position])
        admet = admet_engine.evaluate(smiles) or {}
        rows.append({
            **entry,
            "Predicted_pIC50": round(pic50, 2),
            "IC50_nM": round(10 ** (9 - pic50), 2),
            "ADMET_QED": admet.get("QED"),
            "Violations": admet.get("Violations"),
            "Lipinski_Reasons": admet.get("Reasons"),
            "PubChem_Link": pubchem_link(smiles),
        })

    rows.sort(key=lambda r: r["Predicted_pIC50"], reverse=True)
    novels = [r for r in rows if r["is_novel"]]
    refs = [r for r in rows if not r["is_novel"]]
    return refs, novels, reference_source, skipped


class VirtualScreeningPipeline:
    """Kept for backwards compatibility with earlier notebooks."""

    def run_screening(self, candidates, pdb_id=DEFAULT_TARGET):
        return screen(candidates, pdb_id)

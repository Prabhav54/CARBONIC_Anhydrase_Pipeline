"""Virtual screening: score candidates against one carbonic anhydrase structure.

Everything here is keyed to the structure the user typed. The isoform comes from
RCSB, the prediction takes that isoform's protein descriptors as model input,
the 0-10 score is a percentile against that isoform's own measured inhibitors,
and the reference molecules are that isoform's known inhibitors plus whatever
was actually co-crystallised in the entry. Screening a CA II structure and a
CA IX structure gives different answers because every one of those inputs
differs.

Docking is deliberately not part of this step. It takes tens of seconds per
molecule, so the app runs it on demand through src.pipeline.docking_engine.
"""
import urllib.parse

from src.logger import logging
from src.pipeline import admet_engine, chembl, scoring
from src.pipeline.potency_model import features, get_model
from src.pipeline.rcsb import StructureError, describe


def pubchem_link(smiles):
    return f"https://pubchem.ncbi.nlm.nih.gov/#query={urllib.parse.quote(smiles)}"


def _reference_rows(structure):
    """Known inhibitors for this structure: its own ligand, then the isoform's.

    Returns (rows, note). The note says where the list came from, including
    when ChEMBL could not be reached -- in which case the only references are
    the ones the crystal structure itself provides.
    """
    rows = []
    for ligand in structure["ligands"]:
        rows.append({
            "name": f"{ligand['name']} ({ligand['comp_id']})",
            "smiles": ligand["smiles"],
            "chembl_id": None,
            "provenance": f"co-crystallised in {structure['pdb_id']}",
            "from_structure": True,
        })

    try:
        known, target = chembl.known_inhibitors(structure["uniprot"])
    except chembl.ChemblUnavailable as e:
        logging.warning(f"ChEMBL unavailable: {e}")
        note = ("ChEMBL could not be reached, so the only reference shown is the "
                "ligand co-crystallised in this structure."
                if rows else "ChEMBL could not be reached, so no reference "
                             "inhibitors are shown.")
        return rows, note

    seen = {r["smiles"] for r in rows}
    for row in known:
        if row["smiles"] in seen:
            continue
        seen.add(row["smiles"])
        rows.append({**row, "from_structure": False})

    isoform = structure["isoform"]
    if target is None:
        note = f"ChEMBL lists no target for {isoform.label} ({isoform.accession})."
    else:
        note = (f"Known {isoform.label} inhibitors from ChEMBL target "
                f"{target['chembl_id']}, plus the ligand in this crystal structure.")
    return rows, note


def _score_rows(entries, isoform, kind):
    """Predicts and scores a list of {'name', 'smiles', ...} against one isoform.

    Returns (rows, skipped). Rows carry the 0-10 score when the isoform has a
    reference distribution and None when it does not; nothing is invented for
    the isoforms with no bioactivity data.

    `kind` is 'candidate' or 'reference' and travels with the row, so the two
    sets can be put on one axis and still be told apart.
    """
    X, valid = features([e["smiles"] for e in entries], isoform)
    parsed = set(valid)
    skipped = [e["name"] for i, e in enumerate(entries) if i not in parsed]
    if not valid:
        return [], skipped

    predictions = get_model().predict(X)

    rows = []
    for position, index in enumerate(valid):
        entry = entries[index]
        smiles = entry["smiles"]
        pic50 = float(predictions[position])
        admet = admet_engine.evaluate(smiles) or {}
        row = {
            **entry,
            "kind": kind,
            "Predicted_pIC50": round(pic50, 2),
            "IC50_nM": round(10 ** (9 - pic50), 2),
            "ADMET_QED": admet.get("QED"),
            "MW": admet.get("MW"),
            "LogP": admet.get("LogP"),
            "TPSA": admet.get("TPSA"),
            # The four rules individually, so the UI can show which one failed
            # rather than only that something did.
            "Lipinski": admet.get("Lipinski", []),
            "Violations": admet.get("Violations"),
            "Lipinski_Pass": admet.get("Pass"),
            "Lipinski_Reasons": admet.get("Reasons"),
            "PubChem_Link": pubchem_link(smiles),
            "score": None,
            "percentile": None,
        }
        if isoform.is_scored:
            row.update(scoring.score_out_of_ten(pic50, isoform))
        rows.append(row)

    # Ranked by the score where there is one, by predicted pIC50 otherwise
    # (the two orderings agree; the percentile is monotonic in pIC50).
    rows.sort(key=lambda r: r["Predicted_pIC50"], reverse=True)
    return rows, skipped


def _ranking(novel_rows, reference_rows):
    """Where the best candidate falls among this isoform's known inhibitors.

    Both sets are scored by the same model, so this is a prediction-against-
    prediction comparison rather than prediction-against-measurement. That is
    the only fair way to rank them on one axis, and the UI says so.

    Returns None when there is nothing to compare: an isoform whose potency is
    withheld, or a structure with no references retrieved.
    """
    scored = [r for r in novel_rows if r["score"] is not None]
    refs = [r for r in reference_rows if r["score"] is not None]
    if not scored or not refs:
        return None
    best = scored[0]
    stronger = sum(1 for r in refs if r["Predicted_pIC50"] > best["Predicted_pIC50"])
    return {
        "name": best["name"],
        "score": best["score"],
        "rank": stronger + 1,
        "of": len(refs) + 1,
        "beats": len(refs) - stronger,
        "references": len(refs),
    }


def screen(candidates, pdb_id):
    """Screens `candidates` against the structure `pdb_id`.

    `candidates` is a list of {'name', 'smiles'}. Raises StructureError if the
    PDB ID is not a screenable human carbonic anhydrase.
    """
    structure = describe(pdb_id)
    isoform = structure["isoform"]

    references, reference_note = _reference_rows(structure)
    novel_rows, skipped = _score_rows(candidates, isoform, "candidate")
    reference_rows, reference_skipped = _score_rows(references, isoform, "reference")

    return {
        "structure": structure,
        "isoform": isoform,
        "novels": novel_rows,
        "refs": reference_rows,
        "reference_note": reference_note,
        "skipped": skipped + reference_skipped,
        "ranking": _ranking(novel_rows, reference_rows),
        "scored": isoform.is_scored,
        "scale": scoring.describe_scale(isoform),
    }


class VirtualScreeningPipeline:
    """Kept for backwards compatibility with earlier notebooks."""

    def run_screening(self, candidates, pdb_id):
        result = screen(candidates, pdb_id)
        return result["refs"], result["novels"], result["reference_note"], result["skipped"]

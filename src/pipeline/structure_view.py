"""Everything the app shows about a structure before any molecule is screened.

This is step one of the interface: you type a PDB ID and get back what the
structure is, what its isoform is used or investigated for, whether its potency
can be scored at all, and exactly where docking would happen. Assembling it in
one place keeps the page and the JSON route from drifting apart.

The three kinds of claim here are deliberately kept separate, because they are
not equally strong:

  * `isoform.therapeutic` is curated context about why the isoform is a target.
  * `drugs` are what ChEMBL annotates as acting on this target, with phases.
  * `diseases` are UniProt's genetic disease annotations for the gene, which
    describe what variants in it cause, not what inhibiting it treats.
"""
from src.logger import logging
from src.pipeline import chembl, receptor_prep, uniprot
from src.pipeline.rcsb import StructureError, describe


def _approved_and_trialled(accession):
    """Drugs ChEMBL annotates as acting on this target, most advanced first."""
    try:
        rows, target = chembl.known_inhibitors(accession)
    except chembl.ChemblUnavailable as e:
        logging.warning(f"ChEMBL unavailable for {accession}: {e}")
        return [], None
    annotated = [r for r in rows if r["source"] == "mechanism"]
    return annotated, target


def build(pdb_id, with_site=True):
    """Resolves a PDB ID into the structure panel's contents.

    Raises StructureError if the entry is not a screenable carbonic anhydrase.
    A missing docking site is not fatal: the structure is still described and
    the reason is reported, because potency does not need the site.
    """
    structure = describe(pdb_id)
    isoform = structure["isoform"]

    site, site_error = None, None
    if with_site:
        try:
            site = receptor_prep.inspect_site(structure["pdb_id"])
        except StructureError as e:
            site_error = str(e)

    drugs, target = _approved_and_trialled(structure["uniprot"])
    return {
        "structure": structure,
        "isoform": isoform,
        "site": site,
        "site_error": site_error,
        "diseases": uniprot.diseases(structure["uniprot"]),
        "drugs": drugs,
        "chembl_target": target,
        "scored": isoform.is_scored,
        "withheld_reason": isoform.withheld_reason,
    }


def as_json(view):
    """The same thing, JSON-safe, for the /structure route and the 3D viewer."""
    isoform = view["isoform"]
    structure = view["structure"]
    return {
        "pdb_id": structure["pdb_id"],
        "title": structure["title"],
        "method": structure["method"],
        "resolution": structure["resolution"],
        "deposited": structure["deposited"],
        "rcsb_url": structure["rcsb_url"],
        "image_url": structure["image_url"],
        "ligands": structure["ligands"],
        "isoform": {
            "label": isoform.label,
            "gene": isoform.gene,
            "accession": isoform.accession,
            "where": isoform.role,
            "therapeutic": isoform.therapeutic,
            "measurements": isoform.measurements,
            "test_r2": isoform.test_r2,
            "scored": isoform.is_scored,
            "withheld_reason": isoform.withheld_reason,
        },
        "diseases": view["diseases"],
        "drugs": [{"name": d["name"], "chembl_id": d["chembl_id"],
                   "provenance": d["provenance"]} for d in view["drugs"]],
        "site": view["site"],
        "site_error": view["site_error"],
    }

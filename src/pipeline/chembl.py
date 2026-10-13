"""Known inhibitors of one carbonic anhydrase isoform, from ChEMBL.

The previous version kept a single hardcoded list of five sulfonamides and
showed it for every target, which made "known inhibitors" look identical whether
you screened a CA II or a CA IX structure. Here the isoform's own ChEMBL target
is resolved from its UniProt accession and queried, so the list genuinely
differs between isoforms.

Two sources, in order of what they actually tell you:

  * mechanism -- molecules ChEMBL annotates as acting on this target, which is
    where the marketed drugs come from (acetazolamide, dorzolamide and so on).
    Only a few isoforms have these.
  * activity -- the most potent molecules with a measured IC50 or Ki against
    this target. Every isoform that has been assayed at all has these.

Each row carries which of the two it came from and its ChEMBL id, because
"annotated as a drug for this target" and "the most potent thing anyone has
measured against it" are different claims and should not be shown as one.
Nothing is substituted when ChEMBL cannot be reached: the caller is told.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

from src.logger import logging

API = "https://www.ebi.ac.uk/chembl/api/data"
# Measured under a burst of requests, ChEMBL occasionally takes longer than 8 s
# to answer, which showed up as whole screenings losing their reference list.
# One retry costs a slow request rather than an empty table.
TIMEOUT = 15
RETRIES = 3
# Wait before retrying, so a burst that tripped rate limiting is not
# immediately repeated.
BACKOFF_SECONDS = 1.0
MAX_REFERENCES = 6
# Only measurements at least this potent are offered as "known inhibitors";
# 10 uM is the usual floor for calling something active at all.
MAX_REFERENCE_IC50_NM = 10000


class ChemblUnavailable(RuntimeError):
    """ChEMBL could not be reached, so there is nothing real to show."""


def _get(path, params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "ca-predictor"})
    last_error = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            last_error = e
            if attempt + 1 < RETRIES:
                time.sleep(BACKOFF_SECONDS * (attempt + 1))
                logging.info(f"ChEMBL {path} retry after {e}")
    raise ChemblUnavailable(str(last_error))


@lru_cache(maxsize=32)
def target_for(accession):
    """The single-protein ChEMBL target for a UniProt accession, or None.

    Resolved rather than hardcoded: the previous table held three ChEMBL ids for
    three isoforms, which is why the other nine could not be screened at all.
    """
    payload = _get("target.json", {"target_components__accession": accession, "limit": 20})
    targets = payload.get("targets", [])
    single = [t for t in targets if t.get("target_type") == "SINGLE PROTEIN"]
    chosen = (single or targets or [None])[0]
    if chosen is None:
        logging.warning(f"ChEMBL has no target for {accession}")
        return None
    return {"chembl_id": chosen["target_chembl_id"],
            "pref_name": chosen.get("pref_name") or chosen["target_chembl_id"]}


def _molecule_details(molecule_ids):
    """Names and structures for a set of ChEMBL molecule ids."""
    if not molecule_ids:
        return {}
    payload = _get("molecule.json", {"molecule_chembl_id__in": ",".join(molecule_ids),
                                     "limit": len(molecule_ids)})
    details = {}
    for molecule in payload.get("molecules", []):
        smiles = (molecule.get("molecule_structures") or {}).get("canonical_smiles")
        if not smiles:
            continue
        name = molecule.get("pref_name") or molecule["molecule_chembl_id"]
        details[molecule["molecule_chembl_id"]] = {
            "name": name.title() if name.isupper() else name,
            "smiles": smiles,
            "max_phase": molecule.get("max_phase"),
        }
    return details


def _by_mechanism(chembl_id):
    """Molecules ChEMBL annotates as acting on this target."""
    payload = _get("mechanism.json", {"target_chembl_id": chembl_id, "limit": 50})
    ids = sorted({m["molecule_chembl_id"] for m in payload.get("mechanisms", [])
                  if m.get("molecule_chembl_id")})
    details = _molecule_details(ids[:MAX_REFERENCES])
    rows = []
    for chembl_molecule_id, detail in details.items():
        phase = detail.get("max_phase")
        if phase is not None and float(phase) >= 4:
            provenance = "approved drug acting on this isoform (ChEMBL mechanism)"
        elif phase:
            provenance = f"clinical phase {int(float(phase))} (ChEMBL mechanism)"
        else:
            provenance = "annotated as acting on this isoform (ChEMBL mechanism)"
        rows.append({
            "name": detail["name"],
            "smiles": detail["smiles"],
            "chembl_id": chembl_molecule_id,
            "source": "mechanism",
            "provenance": provenance,
            "measured": None,
        })
    return rows


def _by_activity(chembl_id):
    """The most potent molecules measured against this target."""
    payload = _get("activity.json", {
        "target_chembl_id": chembl_id,
        "standard_type__in": "IC50,Ki",
        "standard_units": "nM",
        "standard_relation": "=",
        "order_by": "standard_value",
        "limit": 60,
    })
    best = {}
    for activity in payload.get("activities", []):
        value, molecule_id = activity.get("standard_value"), activity.get("molecule_chembl_id")
        if not value or not molecule_id:
            continue
        value = float(value)
        if value <= 0 or value > MAX_REFERENCE_IC50_NM:
            continue
        if molecule_id not in best or value < best[molecule_id][0]:
            best[molecule_id] = (value, activity.get("standard_type"))

    ranked = sorted(best.items(), key=lambda kv: kv[1][0])[:MAX_REFERENCES]
    details = _molecule_details([molecule_id for molecule_id, _ in ranked])
    rows = []
    for molecule_id, (value, kind) in ranked:
        detail = details.get(molecule_id)
        if not detail:
            continue
        rows.append({
            "name": detail["name"],
            "smiles": detail["smiles"],
            "chembl_id": molecule_id,
            "source": "activity",
            "provenance": f"measured {kind} {value:g} nM against this isoform (ChEMBL)",
            "measured": {"type": kind, "value_nm": value},
        })
    return rows


@lru_cache(maxsize=16)
def known_inhibitors(accession):
    """Known inhibitors of this isoform, mechanism-annotated ones first.

    Returns (rows, target). Raises ChemblUnavailable if ChEMBL cannot be
    reached, so the caller can say so rather than show a stand-in.

    Cached per warm instance: the structure panel and the screening both want
    this list, and it does not change between two requests a second apart. The
    returned list is treated as read-only by callers.
    """
    target = target_for(accession)
    if target is None:
        return [], None

    rows = _by_mechanism(target["chembl_id"])
    if len(rows) < MAX_REFERENCES:
        known = {r["chembl_id"] for r in rows}
        for row in _by_activity(target["chembl_id"]):
            if row["chembl_id"] not in known:
                rows.append(row)
            if len(rows) >= MAX_REFERENCES:
                break
    logging.info(f"{accession} -> {target['chembl_id']}: {len(rows)} known inhibitors")
    return rows, target

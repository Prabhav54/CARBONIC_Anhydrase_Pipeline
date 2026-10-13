"""RCSB lookups: validate a typed PDB ID and fetch what the app needs from it.

Nothing about a structure is assumed or hardcoded. A typed ID is accepted only
if RCSB says its polymer maps to one of the twelve catalytic human carbonic
anhydrases (src/pipeline/isoforms.py), and everything shown about it -- title,
resolution, method, the ligand actually bound in the crystal -- comes from the
entry itself.

Responses are cached per warm instance and on disk under the temp directory, so
repeated screening against one structure costs one round trip.
"""
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

from src.logger import logging
from src.pipeline import isoforms

DATA_API = "https://data.rcsb.org/rest/v1/core"
FILE_API = "https://files.rcsb.org/download"
TIMEOUT = 15
# Resolving one structure takes several calls (entry, each polymer entity, each
# ligand). A single slow response should not cost the whole lookup, so reads are
# retried once before giving up.
RETRIES = 3
# Retries wait before trying again: resolving one structure makes several calls,
# and a page of lookups can trip rate limiting, which an immediate retry walks
# straight back into.
BACKOFF_SECONDS = 1.0
RETRY_STATUSES = {429, 500, 502, 503, 504}
CACHE_DIR = os.path.join(tempfile.gettempdir(), "ca_rcsb")

PDB_ID_PATTERN = re.compile(r"^[0-9][A-Z0-9]{3}$")

# Crystallisation additives, cryoprotectants, buffers and ions. These are in the
# crystal but they are not what anyone co-crystallised the protein with, so they
# are not reported as "the bound ligand".
NON_LIGAND_COMPONENTS = {
    "ZN", "HOH", "DOD", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "2PE", "P6G",
    "SO4", "PO4", "CL", "NA", "K", "MG", "CA", "MN", "FE", "CU", "NI", "CD", "HG",
    "ACT", "ACY", "DMS", "MES", "TRS", "EPE", "IMD", "CIT", "FLC", "TAR", "MLI",
    "NO3", "AZI", "BCT", "CO3", "BME", "MRD", "IPA", "MOH", "EOH", "ETX", "TLA",
    "SCN", "BR", "IOD", "F", "NH4", "UNX", "UNL",
}


class StructureError(ValueError):
    """A typed PDB ID that cannot be screened, with a reason fit to show a user."""


def _get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ca-predictor"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            # 404 is an answer. 429 and 5xx are the service asking us to wait,
            # and retrying those immediately is what makes a burst of lookups
            # fail as a group.
            if e.code not in RETRY_STATUSES or attempt + 1 >= RETRIES:
                raise
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
            logging.info(f"RCSB retry after HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            if attempt + 1 >= RETRIES:
                raise
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
            logging.info(f"RCSB retry after {e}")


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def normalise_id(raw):
    """Upper-cases and validates the four-character PDB ID shape."""
    candidate = (raw or "").strip().upper()
    if not PDB_ID_PATTERN.match(candidate):
        raise StructureError(
            f"{candidate or 'That'} is not a PDB ID. A PDB ID is four characters "
            f"starting with a digit, like 3HS4 or 5FL6.")
    return candidate


def _entity_accessions(pdb_id, entity_ids):
    accessions = []
    for entity_id in entity_ids:
        entity = _get_json(f"{DATA_API}/polymer_entity/{pdb_id}/{entity_id}")
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers", {})
        accessions += identifiers.get("uniprot_ids") or []
    return accessions


def _bound_ligands(pdb_id, entity_ids):
    """The co-crystallised ligands of this entry, with their real SMILES."""
    ligands = []
    for entity_id in entity_ids:
        entity = _get_json(f"{DATA_API}/nonpolymer_entity/{pdb_id}/{entity_id}")
        component = entity.get("pdbx_entity_nonpoly", {})
        comp_id = component.get("comp_id")
        if not comp_id or comp_id in NON_LIGAND_COMPONENTS:
            continue
        chem_comp = _get_json(f"{DATA_API}/chemcomp/{comp_id}")
        descriptor = chem_comp.get("rcsb_chem_comp_descriptor", {})
        smiles = descriptor.get("SMILES_stereo") or descriptor.get("SMILES")
        if not smiles:
            continue
        ligands.append({
            "comp_id": comp_id,
            "name": (chem_comp.get("chem_comp", {}).get("name") or comp_id).strip().title(),
            "formula": chem_comp.get("chem_comp", {}).get("formula"),
            "smiles": smiles,
        })
    return ligands


@lru_cache(maxsize=64)
def describe(pdb_id):
    """Resolves a PDB ID to a carbonic anhydrase structure, or raises StructureError.

    Returns the entry's own metadata plus the Isoform its polymer maps to.
    """
    pdb_id = normalise_id(pdb_id)
    try:
        entry = _get_json(f"{DATA_API}/entry/{pdb_id}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise StructureError(f"The PDB has no entry {pdb_id}.")
        raise StructureError(f"RCSB could not be reached for {pdb_id} (HTTP {e.code}).")
    except Exception as e:
        logging.warning(f"RCSB entry lookup failed for {pdb_id}: {e}")
        raise StructureError(f"RCSB could not be reached to look up {pdb_id}.")

    identifiers = entry.get("rcsb_entry_container_identifiers", {})
    accessions = _entity_accessions(pdb_id, identifiers.get("polymer_entity_ids") or [])

    isoform = next((i for i in (isoforms.by_accession(a) for a in accessions) if i), None)
    if isoform is None:
        title = (entry.get("struct", {}).get("title") or "").strip()
        detail = f' It is "{title}".' if title else ""
        raise StructureError(
            f"{pdb_id} is not a human carbonic anhydrase.{detail} This app screens "
            f"the twelve catalytic human isoforms, CA I to CA XIV.")

    info = entry.get("rcsb_entry_info", {})
    resolutions = info.get("resolution_combined") or []
    return {
        "pdb_id": pdb_id,
        "title": (entry.get("struct", {}).get("title") or pdb_id).strip(),
        "method": (info.get("experimental_method") or "").strip(),
        "resolution": round(resolutions[0], 2) if resolutions else None,
        "deposited": (entry.get("rcsb_accession_info", {}).get("deposit_date") or "")[:10],
        "uniprot": isoform.accession,
        "isoform": isoform,
        "ligands": _bound_ligands(pdb_id, identifiers.get("non_polymer_entity_ids") or []),
        "rcsb_url": f"https://www.rcsb.org/structure/{pdb_id}",
        # RCSB renders a still of the deposited assembly for every entry.
        "image_url": f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_assembly-1.jpeg",
    }


def coordinates(pdb_id):
    """The entry's PDB-format coordinates, cached on disk.

    A handful of very large entries are distributed only as mmCIF; carbonic
    anhydrases are small enough that the legacy format always exists, and the
    error says so plainly if one ever does not.
    """
    pdb_id = normalise_id(pdb_id)
    path = _cache_path(f"{pdb_id.lower()}.pdb")
    if os.path.exists(path) and os.path.getsize(path):
        with open(path) as f:
            return f.read()

    try:
        request = urllib.request.Request(f"{FILE_API}/{pdb_id}.pdb",
                                         headers={"User-Agent": "ca-predictor"})
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise StructureError(
                f"RCSB has no PDB-format coordinate file for {pdb_id}, so it "
                f"cannot be prepared for docking.")
        raise StructureError(f"Could not download {pdb_id} (HTTP {e.code}).")
    except Exception as e:
        logging.warning(f"coordinate download failed for {pdb_id}: {e}")
        raise StructureError(f"Could not download the coordinates for {pdb_id}.")

    # Write via a temporary file so a concurrent request never reads a partial one.
    with tempfile.NamedTemporaryFile("w", dir=CACHE_DIR, delete=False) as staged:
        staged.write(text)
    os.replace(staged.name, path)
    return text

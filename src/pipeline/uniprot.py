"""Disease annotations for an isoform, from UniProt.

This answers "what is this protein implicated in" with something citable rather
than something written from memory. UniProt's DISEASE comments describe the
genetic disease caused by variants in the gene, which is a different claim from
"inhibiting this protein treats X": CA II variants cause osteopetrosis, while
CA II inhibitors are used for glaucoma. The app shows the two separately and
labels them, and this module supplies only the first.

Only three of the twelve isoforms carry such an annotation (CA II, CA IV and
CA XII), so the absence of one is normal and is not an error.
"""
import json
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

from src.logger import logging

API = "https://rest.uniprot.org/uniprotkb"
TIMEOUT = 12
# Because a failed lookup is indistinguishable from "this gene has no annotated
# disease", one transient timeout would quietly remove the panel. Retrying makes
# the empty answer mean what it says.
RETRIES = 2


def fetch(accession):
    """The raw UniProt payload for one accession. Raises on failure."""
    url = f"{API}/{accession}.json?{urllib.parse.urlencode({'fields': 'cc_disease'})}"
    request = urllib.request.Request(url, headers={"User-Agent": "ca-predictor"})
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 >= RETRIES:
                raise


@lru_cache(maxsize=32)
def diseases(accession):
    """UniProt's disease annotations for one accession, possibly empty.

    Returns a list of {name, acronym, description, mim}. Never raises: the
    disease note is context, so losing it must not cost the user a screening.
    """
    try:
        payload = fetch(accession)
    except Exception as e:
        logging.warning(f"UniProt disease lookup failed for {accession}: {e}")
        return []

    found = []
    for comment in payload.get("comments", []):
        if comment.get("commentType") != "DISEASE":
            continue
        disease = comment.get("disease") or {}
        name = disease.get("diseaseId")
        if not name:
            continue
        description = (disease.get("description") or "").strip()
        reference = disease.get("diseaseCrossReference") or {}
        found.append({
            "name": name,
            "acronym": disease.get("acronym"),
            "description": description,
            "mim": reference.get("id") if reference.get("database") == "MIM" else None,
        })
    return found

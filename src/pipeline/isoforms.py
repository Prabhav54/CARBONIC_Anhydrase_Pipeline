"""The twelve catalytically active human carbonic anhydrases.

Keyed by UniProt accession, because that is what RCSB reports for a structure's
polymer entity and it is the only identifier that ties a PDB entry, a ChEMBL
target and a set of protein descriptors together without guesswork.

The table itself is built offline by data_curation_scripts/build_isoform_features.py
(the descriptors need Biopython, which the deployed function does not carry) and
shipped as artifacts/models/isoforms.json.
"""
import json
import os
from functools import lru_cache

TABLE_PATH = os.path.join("artifacts", "models", "isoforms.json")


class Isoform:
    """One isoform: what it is, how the model sees it, how well it is measured."""

    def __init__(self, accession, record, min_measurements):
        self.accession = accession
        self.label = record["label"]              # 'CA II'
        self.gene = record["gene"]                # 'CA2'
        self.role = record["role"]                # where the protein is
        # Why anyone would inhibit it. Deliberately conservative, and shown
        # alongside two fetched things it must not be confused with: the drugs
        # ChEMBL annotates against this target, and the genetic disease UniProt
        # annotates for the gene.
        self.therapeutic = record.get("therapeutic", "")
        self.sequence_length = record["sequence_length"]
        self.descriptors = record["descriptors"]
        self.measurements = record["measurements"]
        # How well the ensemble actually predicts this isoform, on molecules it
        # never saw in training. None means it was never tested on this one.
        self.test_r2 = record.get("test_r2")
        self.test_mae = record.get("test_mae")
        self.test_n = record.get("test_n")
        self._quantiles = record["pic50_quantiles"]
        self._min_measurements = min_measurements

    @property
    def is_scored(self):
        """False when potency for this isoform cannot be predicted honestly.

        Two ways to land here: too little bioactivity data (CA VB, CA VII and
        CA XII have none at all; CA VI has 8 molecules, CA XIV one), or enough
        data but no demonstrated accuracy on it (CA XIII, 117 molecules, test
        R2 about 0.0). Either way the structure can still be docked and its site
        shown; only the number out of 10 is withheld.
        """
        return self._quantiles is not None

    @property
    def withheld_reason(self):
        """Why potency is not scored for this isoform, or None if it is."""
        if self.is_scored:
            return None
        if self.measurements == 0:
            return (f"No {self.label} bioactivity data exists in the training set, "
                    f"so the model has never learned this isoform.")
        if self.measurements < self._min_measurements:
            return (f"Only {self.measurements} molecules have been measured against "
                    f"{self.label}, too few to place a prediction on a scale.")
        if self.test_r2 is not None:
            return (f"The model scores R2 {self.test_r2:.2f} on held-out {self.label} "
                    f"molecules, so it cannot rank them reliably.")
        return f"The model's accuracy on {self.label} has not been measured."

    @property
    def quantiles(self):
        if self._quantiles is None:
            raise ValueError(f"{self.label} has no measured pIC50 distribution "
                             f"({self.measurements} measurements, "
                             f"{self._min_measurements} needed)")
        return self._quantiles

    def feature_vector(self):
        """The four protein descriptors, in the order the ensemble was trained on."""
        d = self.descriptors
        return [d["hydrophobicity"], d["weight"], d["isoelectric"], d["aromaticity"]]

    def __repr__(self):
        return f"<Isoform {self.label} {self.accession}>"


@lru_cache(maxsize=1)
def _table():
    with open(TABLE_PATH) as f:
        payload = json.load(f)
    minimum = payload["min_measurements"]
    isoforms = {a: Isoform(a, r, minimum) for a, r in payload["isoforms"].items()}
    return payload, isoforms


def all_isoforms():
    """Every supported isoform, in roman-numeral order."""
    _, isoforms = _table()
    return list(isoforms.values())


def by_accession(accession):
    """The isoform for a UniProt accession, or None if it is not a catalytic CA."""
    _, isoforms = _table()
    return isoforms.get(accession.upper())


def by_label(label):
    """The isoform for a roman-numeral label such as 'CA II'."""
    _, isoforms = _table()
    for isoform in isoforms.values():
        if isoform.label == label:
            return isoform
    return None


def family_quantiles():
    """The pIC50 distribution across every measured isoform.

    Used only to describe where one isoform's distribution sits relative to the
    family, never as a stand-in for an isoform that has no data of its own.
    """
    payload, _ = _table()
    return payload["family_pic50_quantiles"]


def family_measurements():
    payload, _ = _table()
    return payload["family_measurements"]


# The four descriptor columns, in training order. src/components/data_transformation.py
# appends them to the fingerprint in exactly this order.
DESCRIPTOR_COLUMNS = ["Protein_Hydrophobicity", "Protein_Weight",
                      "Protein_Isoelectric", "Protein_Aromaticity"]

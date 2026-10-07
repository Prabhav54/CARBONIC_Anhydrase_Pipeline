"""Supported docking targets: PDB structure -> CA isoform and ChEMBL target."""

TARGETS = {
    "1CA2": {"isoform": "CA II", "chembl_id": "CHEMBL205"},
    "3HS4": {"isoform": "CA II", "chembl_id": "CHEMBL205"},
    "4ZAO": {"isoform": "CA II", "chembl_id": "CHEMBL205"},
    "3IAI": {"isoform": "CA IX", "chembl_id": "CHEMBL3594"},
    "5FL4": {"isoform": "CA IX", "chembl_id": "CHEMBL3594"},
    "5FL6": {"isoform": "CA IX", "chembl_id": "CHEMBL3594"},
    "1JCZ": {"isoform": "CA XII", "chembl_id": "CHEMBL3242"},
    "5JN9": {"isoform": "CA XII", "chembl_id": "CHEMBL3242"},
}

DEFAULT_TARGET = "3HS4"

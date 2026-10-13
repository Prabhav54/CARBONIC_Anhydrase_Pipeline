"""Drug-likeness metrics: Lipinski's Rule of 5 and the QED score.

Each of the four rules is reported with the measured value, its limit and
whether it passed, so the UI can show why a molecule failed rather than only
that it did.
"""
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

# MW <= 500, LogP <= 5, H-donors <= 5, H-acceptors <= 10. Lipinski's own
# formulation tolerates one violation, which is why marketed drugs that break a
# single rule still count as orally drug-like.
LIPINSKI_RULES = (
    ("Molecular weight", "MW", lambda m: Descriptors.MolWt(m), 500, "<="),
    ("Lipophilicity", "LogP", lambda m: Crippen.MolLogP(m), 5, "<="),
    ("H-bond donors", "HBD", lambda m: Descriptors.NumHDonors(m), 5, "<="),
    ("H-bond acceptors", "HBA", lambda m: Descriptors.NumHAcceptors(m), 10, "<="),
)
MAX_TOLERATED_VIOLATIONS = 1


def lipinski_detail(mol):
    """The four rules, each with its measured value, limit and verdict."""
    checks = []
    for name, short, measure, limit, _ in LIPINSKI_RULES:
        value = measure(mol)
        checks.append({
            "name": name,
            "short": short,
            "value": round(float(value), 2),
            "limit": limit,
            "passed": bool(value <= limit),
        })
    return checks


def evaluate(smiles):
    """Returns a metrics dict, or None if the SMILES cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    checks = lipinski_detail(mol)
    broken = [f"{c['short']} {c['value']} > {c['limit']}" for c in checks if not c["passed"]]
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "QED": round(QED.qed(mol), 3),
        "Lipinski": checks,
        "Violations": len(broken),
        "Reasons": ", ".join(broken) if broken else "None",
        "Pass": len(broken) <= MAX_TOLERATED_VIOLATIONS,
    }


class ADMETEngine:
    """Kept for backwards compatibility with earlier notebooks."""

    def evaluate(self, smiles):
        return evaluate(smiles)

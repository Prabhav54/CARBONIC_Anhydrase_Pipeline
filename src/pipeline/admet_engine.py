"""Drug-likeness metrics: Lipinski's Rule of 5 and the QED score."""
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED

# MW < 500, LogP < 5, H-donors < 5, H-acceptors < 10; one violation is tolerated.
LIPINSKI_RULES = (
    ("MW > 500", lambda m: Descriptors.MolWt(m) > 500),
    ("LogP > 5", lambda m: Crippen.MolLogP(m) > 5),
    ("H-donors > 5", lambda m: Descriptors.NumHDonors(m) > 5),
    ("H-acceptors > 10", lambda m: Descriptors.NumHAcceptors(m) > 10),
)
MAX_TOLERATED_VIOLATIONS = 1


def evaluate(smiles):
    """Returns a metrics dict, or None if the SMILES cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    broken = [name for name, violates in LIPINSKI_RULES if violates(mol)]
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 2),
        "QED": round(QED.qed(mol), 3),
        "Violations": len(broken),
        "Reasons": ", ".join(broken) if broken else "None",
        "Pass": len(broken) <= MAX_TOLERATED_VIOLATIONS,
    }


class ADMETEngine:
    """Kept for backwards compatibility with earlier notebooks."""

    def evaluate(self, smiles):
        return evaluate(smiles)

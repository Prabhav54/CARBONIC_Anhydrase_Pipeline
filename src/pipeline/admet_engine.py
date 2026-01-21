from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen

class ADMETEngine:
    def evaluate(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        
        # 1. Calculate Core Metrics
        mw = Descriptors.MolWt(mol)           # Molecular Weight
        logp = Crippen.MolLogP(mol)           # Solubility (Lipophilicity)
        hbd = Descriptors.NumHDonors(mol)     # H-Bond Donors
        hba = Descriptors.NumHAcceptors(mol)  # H-Bond Acceptors
        psa = Descriptors.TPSA(mol)           # Polar Surface Area
        qed = QED.qed(mol)                    # Drug-Likeness (0 to 1)
        
        # 2. Lipinski's Rule of 5 (The "Is this a drug?" Test)
        # Rules: MW < 500, LogP < 5, HBD < 5, HBA < 10
        violations = 0
        violation_reasons = []
        
        if mw > 500: 
            violations += 1
            violation_reasons.append("MW > 500")
        if logp > 5: 
            violations += 1
            violation_reasons.append("LogP > 5")
        if hbd > 5: 
            violations += 1
            violation_reasons.append("H-Donors > 5")
        if hba > 10: 
            violations += 1
            violation_reasons.append("H-Acceptors > 10")
        
        # We allow 1 violation (standard practice)
        is_pass = violations <= 1
        
        return {
            "MW": round(mw, 2),
            "LogP": round(logp, 2),
            "QED": round(qed, 3),
            "Violations": violations,
            "Reasons": ", ".join(violation_reasons) if violation_reasons else "None",
            "Pass": is_pass
        }
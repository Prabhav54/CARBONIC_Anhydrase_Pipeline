from tdc.single_pred import ADME, Tox

class ADMETFilter:
    def __init__(self):
        # Initialize models once (they download automatically)
        self.caco2 = ADME(name='Caco2_Wang') # Absorption
        self.herg = Tox(name='hERG')         # Heart Toxicity

    def check_safety(self, smiles):
        # 1. Absorption (Caco-2): > -5.15 is good
        caco2_score = self.caco2.predict(smiles)
        
        # 2. Toxicity (hERG): We want LOW probability (0)
        # Note: TDC models vary, check if it returns probability or binary.
        # Usually for hERG, we want to avoid positives.
        herg_score = self.herg.predict(smiles) 
        
        return {
            "Absorption": round(caco2_score, 2),
            "hERG_Toxic": herg_score, # If > 0.5, it's toxic
            "Safe": caco2_score > -5.15 and herg_score < 0.5
        }

# --- TEST IT ---
if __name__ == "__main__":
    filter = ADMETFilter()
    print(filter.check_safety("CC(=O)Nc1nnc(s1)S(=O)(=O)N"))
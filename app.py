import os
import pandas as pd
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import VirtualScreeningPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', refs=None, novels=None, target_pdb="5FL6", error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        query = request.form.get('search_query', '').strip()
        target_pdb = "3HS4"
        raw_smiles = []

        if len(query) == 4 and query.isalnum():
            target_pdb = query.upper()
            # Expanded fallback pool to show a full Top 10 list
            raw_smiles = [
                'O=S(=O)(N)c1ccc(cc1)C(=O)O', 'CC(=O)Oc1ccccc1C(=O)O', 
                'Cc1nnc(S(=O)(=O)N)s1', 'O=C(NO)c1ccc(S(=O)(=O)N)cc1',
                'NS(=O)(=O)c1ccc(NC(=O)c2ccccc2)cc1', 
                'CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@@H](N(C)C)[C@H]3O)[C@H](C)C[C@@](C)(O)[C@H](O)[C@H](C)C(=O)[C@@H](C)[C@@H]1O',
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C',
                'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5',
                'C1=CC=C(C=C1)C2=CC(=O)C3=C(C=CC(=C3O2)O)O', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
                'CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC'
            ]
        else:
            raw_smiles = [s.strip() for s in query.replace(',', '\n').split('\n') if s.strip()]

        if not raw_smiles:
            return render_template('home.html', refs=None, novels=None, target_pdb=target_pdb, error="Please enter valid SMILES or a PDB ID.")

        user_mol_names = [f"Novel_Candidate_{i+1}" for i in range(len(raw_smiles))]
        pool_df = pd.DataFrame({'Molecule_Name': user_mol_names, 'clean_smiles': raw_smiles, 'Source': ['User Input'] * len(raw_smiles)})

        pipeline = VirtualScreeningPipeline()
        ref_df, novel_df = pipeline.run_screening(pool_df, pdb_id=target_pdb)
        
        return render_template('home.html', refs=ref_df.to_dict('records'), novels=novel_df.to_dict('records'), target_pdb=target_pdb, error=None)

    except Exception as e:
        return render_template('home.html', refs=None, novels=None, target_pdb="3HS4", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
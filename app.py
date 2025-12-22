from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import VirtualScreeningPipeline

application = Flask(__name__)
app = application

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        pdb_id = request.form.get('pdb_id')
        print(f"Received PDB ID: {pdb_id}")
        
        pipeline = VirtualScreeningPipeline()
        
        try:
            # Run the Virtual Screening
            top_drugs = pipeline.run_screening(pdb_id)
            
            # Convert to list of dictionaries for HTML
            results = top_drugs.to_dict(orient='records')
            
            return render_template('home.html', results=results, pdb_id=pdb_id)
            
        except Exception as e:
            print(e)
            return render_template('home.html', error="Invalid PDB ID or Server Error")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
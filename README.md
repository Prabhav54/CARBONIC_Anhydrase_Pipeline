# 🧬 CA-Predictor: High-Throughput AI Drug Discovery Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow?style=for-the-badge&logo=scikit-learn&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?style=for-the-badge&logo=flask&logoColor=white)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-green?style=for-the-badge)

## 📌 Overview
**CA-Predictor** is an end-to-end, enterprise-grade virtual screening platform designed to accelerate the discovery of novel **Carbonic Anhydrase (CA)** inhibitors. 

Built to mimic the early-stage R&D pipelines of major pharmaceutical companies, this application bridges **Deep Learning potency predictions** with **Physics-based thermodynamic validations**. It allows researchers to dynamically screen novel SMILES strings against live-fetched FDA benchmarks, identifying high-potency lead candidates while strictly enforcing pharmacokinetic constraints.

## ✨ Key Features

### 🧠 1. Ensemble Machine Learning Architecture
* **Hybrid Model:** Utilizes a custom `0.7 Random Forest + 0.3 LSTM` consensus model to predict biological potency ($pIC_{50}$) and half-maximal inhibitory concentration ($IC_{50}$ in nM).
* **Feature Engineering:** Sanitizes and vectorizes raw SMILES strings into dense 1024-bit RDKit topological Morgan Fingerprints.

### 🌐 2. Dynamic Bioinformatics Integration
* **Live ChEMBL API:** Bypasses static baselines by connecting directly to the European Bioinformatics Institute (EBI). Searching a target (e.g., `5FL6` for CA IX) actively queries the ChEMBL database to fetch historically proven therapeutics for that specific isoform.
* **PubChem Verification:** Automatically generates live NIH PubChem database queries for AI-generated candidates to verify structural novelty.

### 🔬 3. Cheminformatics & Physics Engine
* **Pharmacokinetics (ADMET):** Calculates the Quantitative Estimate of Druglikeness (QED) and enforces Lipinski’s Rule of 5 to filter out toxic or non-absorbable compounds.
* **Molecular Docking Simulation:** Computes active-site binding affinity (kcal/mol) to validate the thermodynamic stability of the protein-ligand complex.
* **10ns Molecular Dynamics (MD):** Features an asynchronous microservice architecture that simulates OpenMM MD trajectory analyses, reporting MM-PBSA free energy, RMSD stability, and persistent Hydrogen bonds (includes `.xtc` trajectory file generation).

### 💻 4. Pro-Grade UI/UX
* **Glassmorphism Dashboard:** A highly responsive, modern web interface built with Flask, Bootstrap 5, and custom CSS.
* **Interactive 3D Visualization:** Integrates `3Dmol.js` to dynamically fetch receptor geometries from the RCSB PDB, featuring custom rendering states for docking simulations (e.g., highlighted active binding pockets).

---

## ⚙️ System Workflow
1. **Target Selection:** User inputs a target PDB ID (e.g., `3HS4`, `3IAI`). The ChEMBL API fetches established drugs for that target.
2. **Candidate Ingestion:** User inputs a pool of novel SMILES strings.
3. **ML Prediction:** The Ensemble model calculates predicted potency.
4. **Physical Validation:** The Cheminformatics engine calculates ADMET/Lipinski, and the Docking engine scores physical binding.
5. **Presentation:** Results are ranked, visualized in 2D/3D, and served asynchronously to the Flask frontend.

---

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/CA-Predictor.git](https://github.com/yourusername/CA-Predictor.git)
cd CA-Predictor
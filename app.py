"""Flask entrypoint. On Vercel the `app` object below is served as a function."""
import os

from flask import Flask, jsonify, render_template, request

from src.logger import logging
from src.pipeline import predict_pipeline
from src.pipeline.docking_engine import get_engine
from src.pipeline.targets import DEFAULT_TARGET, TARGETS

app = Flask(__name__)

# Clearly-labelled starting set: approved CA inhibitors plus a few non-binders,
# so a first-time visitor can see the model separate them.
EXAMPLE_CANDIDATES = [
    ("Acetazolamide (approved)", "CC(=O)Nc1nnc(s1)S(=O)(=O)N"),
    ("Ethoxzolamide (approved)", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"),
    ("SLC-0111 (in trials)", "O=C(Nc1ccc(F)cc1)Nc1ccc(S(N)(=O)=O)cc1"),
    ("Celecoxib (off-target binder)", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),
    ("Ibuprofen (non-binder)", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("Caffeine (non-binder)", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]


def parse_candidates(raw):
    """Splits a textarea of SMILES (one per line, optional `name, smiles`)."""
    candidates = []
    for line in raw.replace("\r", "").split("\n"):
        line = line.strip()
        if not line:
            continue
        if "," in line:
            name, _, smiles = line.rpartition(",")
            name, smiles = name.strip(), smiles.strip()
        else:
            name, smiles = "", line
        candidates.append({
            "Molecule_Name": name or f"Candidate {len(candidates) + 1}",
            "clean_smiles": smiles,
        })
    return candidates


@app.route("/")
def home():
    return render_template(
        "home.html",
        targets=TARGETS,
        target_pdb=DEFAULT_TARGET,
        examples=EXAMPLE_CANDIDATES,
        smiles_input="\n".join(f"{name}, {smiles}" for name, smiles in EXAMPLE_CANDIDATES),
    )


@app.route("/predict", methods=["POST"])
def predict():
    target_pdb = request.form.get("target_pdb", DEFAULT_TARGET).upper()
    smiles_input = request.form.get("smiles_input", "")

    page = {
        "targets": TARGETS,
        "target_pdb": target_pdb if target_pdb in TARGETS else DEFAULT_TARGET,
        "examples": EXAMPLE_CANDIDATES,
        "smiles_input": smiles_input,
    }
    if target_pdb not in TARGETS:
        return render_template("home.html", **page, error=f"Unsupported target {target_pdb}.")

    candidates = parse_candidates(smiles_input)
    if not candidates:
        return render_template("home.html", **page, error="Enter at least one SMILES string.")

    try:
        refs, novels, reference_source, skipped = predict_pipeline.screen(candidates, target_pdb)
    except Exception:
        logging.exception("screening failed")
        return render_template("home.html", **page, error="Screening failed. Please try again.")

    return render_template(
        "home.html", **page, refs=refs, novels=novels,
        reference_source=reference_source, skipped=skipped,
    )


@app.route("/dock", methods=["POST"])
def dock():
    """Docks one molecule. Slow (tens of seconds), so the UI calls it per row."""
    data = request.get_json(silent=True) or {}
    target_pdb = str(data.get("target_pdb", "")).upper()
    smiles = str(data.get("smiles", "")).strip()

    if target_pdb not in TARGETS:
        return jsonify({"error": f"Unsupported target {target_pdb}."}), 400
    if not smiles:
        return jsonify({"error": "No SMILES provided."}), 400

    try:
        result = get_engine(target_pdb).dock(smiles)
    except Exception:
        logging.exception(f"docking failed for {smiles!r} against {target_pdb}")
        return jsonify({"error": "Docking failed."}), 500

    if "error" in result:
        return jsonify(result), 422
    return jsonify({
        **result,
        "target_pdb": target_pdb,
        "isoform": TARGETS[target_pdb]["isoform"],
        "scoring": "AutoDock4Zn",
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)

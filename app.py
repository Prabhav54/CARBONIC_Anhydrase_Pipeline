"""Flask entrypoint. On Vercel the `app` object below is served as a function.

The page is two steps, and they are separate submits on purpose. Step one takes
a PDB ID and shows what the structure is, what its isoform is used for and where
docking would happen. Step two takes molecules. Keeping them apart makes the
dependency visible: the same molecule scores differently against different
isoforms, which a single combined form hid.
"""
import json
import os

from flask import Flask, jsonify, render_template, request

from src.logger import logging
from src.pipeline import isoforms, predict_pipeline, receptor_prep, structure_view
from src.pipeline.docking_engine import get_engine
from src.pipeline.rcsb import StructureError

app = Flask(__name__)

# Clearly-labelled starting set: approved CA inhibitors plus a few non-binders,
# so a first-time visitor can see the model separate them. The target is not
# pre-filled -- which structure to screen against is the user's choice, and
# defaulting it hid the fact that the answer depends on it.
EXAMPLE_CANDIDATES = [
    ("Acetazolamide (approved)", "CC(=O)Nc1nnc(s1)S(=O)(=O)N"),
    ("Ethoxzolamide (approved)", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"),
    ("SLC-0111 (in trials)", "O=C(Nc1ccc(F)cc1)Nc1ccc(S(N)(=O)=O)cc1"),
    ("Celecoxib (off-target binder)", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),
    ("Ibuprofen (non-binder)", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("Caffeine (non-binder)", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
]

EXAMPLE_SMILES = "\n".join(f"{name}, {smiles}" for name, smiles in EXAMPLE_CANDIDATES)

# Structures worth starting from, one per therapeutic story. Offered as
# one-click examples rather than as a default, so the field stays empty until
# the user chooses.
SUGGESTED_STRUCTURES = [
    ("3HS4", "CA II", "glaucoma target, with acetazolamide bound"),
    ("5FL6", "CA IX", "hypoxic tumour target"),
    ("1JCZ", "CA XII", "tumour-associated"),
    ("2CAB", "CA I", "erythrocyte anti-target"),
]


@app.template_filter("ordinal")
def ordinal(value):
    """1 -> 1st, 2 -> 2nd, 62 -> 62nd, 11 -> 11th."""
    number = int(value)
    if number % 100 in (11, 12, 13):
        return f"{number}th"
    return f"{number}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th') }"


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
            "name": name or f"Candidate {len(candidates) + 1}",
            "smiles": smiles,
        })
    return candidates


def page(**extra):
    """Template context shared by every render of the single page."""
    context = {
        "isoforms": isoforms.all_isoforms(),
        "family_measurements": isoforms.family_measurements(),
        "example_smiles": EXAMPLE_SMILES,
        "suggested": SUGGESTED_STRUCTURES,
        "target_pdb": "",
        "smiles_input": "",
        "view": None,
        "view_json": "null",
        "result": None,
        "error": None,
    }
    context.update(extra)
    return context


def _load_structure(target_pdb, context):
    """Resolves the structure for step one, or records why it could not be."""
    try:
        view = structure_view.build(target_pdb)
    except StructureError as e:
        context["error"] = str(e)
        return None
    except Exception:
        logging.exception(f"could not describe {target_pdb}")
        context["error"] = f"Could not look up {target_pdb}. Please try again."
        return None

    context["view"] = view
    context["view_json"] = json.dumps(structure_view.as_json(view))
    context["target_pdb"] = view["structure"]["pdb_id"]
    return view


@app.route("/")
def home():
    return render_template("home.html", **page())


@app.route("/structure", methods=["POST"])
def structure():
    """Step one: resolve a PDB ID and show the target and its docking site."""
    target_pdb = request.form.get("target_pdb", "").strip().upper()
    context = page(target_pdb=target_pdb, smiles_input=EXAMPLE_SMILES)
    if not target_pdb:
        context["error"] = "Enter a PDB ID of a human carbonic anhydrase structure."
        return render_template("home.html", **context)
    _load_structure(target_pdb, context)
    return render_template("home.html", **context)


@app.route("/predict", methods=["POST"])
def predict():
    """Step two: score molecules against the structure chosen in step one."""
    target_pdb = request.form.get("target_pdb", "").strip().upper()
    smiles_input = request.form.get("smiles_input", "")
    context = page(target_pdb=target_pdb, smiles_input=smiles_input)

    if not target_pdb:
        context["error"] = "Choose a structure first."
        return render_template("home.html", **context)

    view = _load_structure(target_pdb, context)
    if view is None:
        return render_template("home.html", **context)

    candidates = parse_candidates(smiles_input)
    if not candidates:
        context["error"] = "Enter at least one SMILES string."
        return render_template("home.html", **context)

    try:
        context["result"] = predict_pipeline.screen(candidates, target_pdb)
    except StructureError as e:
        context["error"] = str(e)
    except Exception:
        logging.exception(f"screening failed for {target_pdb}")
        context["error"] = "Screening failed. Please try again."
    return render_template("home.html", **context)


@app.route("/structure/<pdb_id>")
def structure_json(pdb_id):
    """The structure panel's contents as JSON."""
    try:
        return jsonify(structure_view.as_json(structure_view.build(pdb_id)))
    except StructureError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/dock", methods=["POST"])
def dock():
    """Docks one molecule. Slow (tens of seconds), so the UI calls it per row."""
    data = request.get_json(silent=True) or {}
    target_pdb = str(data.get("target_pdb", "")).strip().upper()
    smiles = str(data.get("smiles", "")).strip()

    if not smiles:
        return jsonify({"error": "No SMILES provided."}), 400

    try:
        engine = get_engine(target_pdb)
    except StructureError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        logging.exception(f"could not prepare {target_pdb} for docking")
        return jsonify({"error": "This structure could not be prepared for docking."}), 500

    try:
        result = engine.dock(smiles)
    except Exception:
        logging.exception(f"docking failed for {smiles!r} against {target_pdb}")
        return jsonify({"error": "Docking failed."}), 500

    if "error" in result:
        return jsonify(result), 422
    return jsonify({
        **result,
        "target_pdb": engine.pdb_id,
        "isoform": engine.structure["isoform"].label,
        "scoring": "AutoDock4Zn",
    })


@app.route("/site/<pdb_id>")
def site(pdb_id):
    """The docking site of a structure, without docking anything into it."""
    try:
        return jsonify(receptor_prep.inspect_site(pdb_id))
    except StructureError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)

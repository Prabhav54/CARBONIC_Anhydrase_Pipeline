"""Command-line screening: predict potency, then optionally dock.

    python main.py --target 3HS4 "CC(=O)Nc1nnc(s1)S(=O)(=O)N" "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    python main.py --target 5FL6 --smiles-file candidates.txt --dock --top 3
    python main.py --list-isoforms

`--target` takes any PDB entry of a catalytic human carbonic anhydrase; the
isoform is resolved from the entry itself rather than from a fixed list.
"""
import argparse
import sys

from src.pipeline import isoforms, predict_pipeline, receptor_prep
from src.pipeline.rcsb import StructureError


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smiles", nargs="*", help="candidate SMILES strings")
    parser.add_argument("--smiles-file", help="file with one SMILES per line")
    parser.add_argument("--target", help="PDB ID to screen against (required)")
    parser.add_argument("--dock", action="store_true", help="also dock the top candidates (slow)")
    parser.add_argument("--top", type=int, default=3, help="how many candidates to dock (default 3)")
    parser.add_argument("--list-isoforms", action="store_true",
                        help="show the supported isoforms and which of them are scored")
    return parser.parse_args(argv)


def collect_smiles(args):
    smiles = list(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file) as f:
            smiles += [line.strip() for line in f if line.strip()]
    return smiles


def list_isoforms():
    print(f"{'isoform':9s} {'gene':6s} {'uniprot':9s} {'measured':>9s} {'test R2':>8s}  potency")
    for isoform in isoforms.all_isoforms():
        r2 = "-" if isoform.test_r2 is None else f"{isoform.test_r2:.2f}"
        verdict = "scored" if isoform.is_scored else "withheld"
        print(f"{isoform.label:9s} {isoform.gene:6s} {isoform.accession:9s} "
              f"{isoform.measurements:9d} {r2:>8s}  {verdict}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.list_isoforms:
        return list_isoforms()

    if not args.target:
        print("No target given. Pass --target with a PDB ID, for example --target 3HS4.",
              file=sys.stderr)
        return 2

    smiles = collect_smiles(args)
    if not smiles:
        print("No candidates given. Pass SMILES strings or --smiles-file.", file=sys.stderr)
        return 2

    candidates = [{"name": f"Candidate {i + 1}", "smiles": s} for i, s in enumerate(smiles)]
    try:
        result = predict_pipeline.screen(candidates, args.target)
    except StructureError as e:
        print(f"{e}", file=sys.stderr)
        return 2

    structure, isoform = result["structure"], result["isoform"]
    print(f"{structure['pdb_id']}  {isoform.label} ({isoform.accession})  "
          f"{structure['resolution'] or '?'} A")
    print(f"{structure['title']}\n")
    print(f"{result['scale']}\n")

    if result["skipped"]:
        print(f"Skipped {len(result['skipped'])} unparseable input(s): "
              f"{', '.join(result['skipped'])}\n", file=sys.stderr)

    def table(rows, heading):
        print(heading)
        print(f"  {'candidate':26s} {'score':>7s} {'pIC50':>7s} {'IC50 nM':>12s} "
              f"{'QED':>6s} {'viol':>5s}")
        for row in rows:
            score = "withheld" if row["score"] is None else f"{row['score']:.1f}/10"
            print(f"  {row['name'][:26]:26s} {score:>7s} {row['Predicted_pIC50']:7.2f} "
                  f"{row['IC50_nM']:12.2f} {row['ADMET_QED']:6.3f} {row['Violations']:5d}")

    if result["novels"]:
        table(result["novels"], "Candidates:")
    if result["refs"]:
        print()
        table(result["refs"], f"Known {isoform.label} inhibitors ({result['reference_note']}):")

    if args.dock and result["novels"]:
        from src.pipeline.docking_engine import get_engine

        try:
            site = receptor_prep.inspect_site(structure["pdb_id"])
        except StructureError as e:
            print(f"\nCannot dock: {e}", file=sys.stderr)
            return 1
        triad = ", ".join(f"{r['name']}{r['number']}" for r in site["coordinating_residues"])
        print(f"\nDocking site: chain {site['chain']}, zinc at "
              f"{', '.join(str(c) for c in site['zinc'])}, held by {triad}")

        engine = get_engine(structure["pdb_id"])
        print(f"Docking the top {min(args.top, len(result['novels']))} candidate(s) "
              f"with AutoDock4Zn scoring:")
        for row in result["novels"][: args.top]:
            docked = engine.dock(row["smiles"])
            if "error" in docked:
                print(f"  {row['name']}: {docked['error']}")
                continue
            print(f"  {row['name']}:")
            for pose in docked["poses"]:
                distance = "n/a" if pose["zinc_distance"] is None else f"{pose['zinc_distance']:.2f} A"
                contact = "zinc contact" if pose["zinc_contact"] else "no zinc contact"
                contacts = ", ".join(f"{c['name']}{c['number']}" for c in pose["contacts"][:5])
                print(f"    pose {pose['rank']}: {pose['score']:7.2f} kcal/mol   "
                      f"N-Zn {distance:>7s}   {contact}")
                print(f"      contacts: {contacts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

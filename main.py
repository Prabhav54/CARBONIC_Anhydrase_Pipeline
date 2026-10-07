"""Command-line screening: predict potency, then optionally dock.

    python main.py --target 3HS4 "CC(=O)Nc1nnc(s1)S(=O)(=O)N" "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
    python main.py --target 5FL6 --smiles-file candidates.txt --dock --top 3
"""
import argparse
import sys

from src.pipeline import predict_pipeline
from src.pipeline.targets import DEFAULT_TARGET, TARGETS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smiles", nargs="*", help="candidate SMILES strings")
    parser.add_argument("--smiles-file", help="file with one SMILES per line")
    parser.add_argument("--target", default=DEFAULT_TARGET, choices=sorted(TARGETS), help="PDB structure to screen against")
    parser.add_argument("--dock", action="store_true", help="also dock the top candidates (slow)")
    parser.add_argument("--top", type=int, default=3, help="how many candidates to dock (default 3)")
    return parser.parse_args(argv)


def collect_smiles(args):
    smiles = list(args.smiles)
    if args.smiles_file:
        with open(args.smiles_file) as f:
            smiles += [line.strip() for line in f if line.strip()]
    return smiles


def main(argv=None):
    args = parse_args(argv)
    smiles = collect_smiles(args)
    if not smiles:
        print("No candidates given. Pass SMILES strings or --smiles-file.", file=sys.stderr)
        return 2

    candidates = [{"Molecule_Name": f"Candidate {i + 1}", "clean_smiles": s} for i, s in enumerate(smiles)]
    print(f"Screening {len(candidates)} candidate(s) against {args.target} ({TARGETS[args.target]['isoform']})\n")

    refs, novels, reference_source, skipped = predict_pipeline.screen(candidates, args.target)
    if skipped:
        print(f"Skipped {len(skipped)} unparseable input(s): {', '.join(skipped)}\n", file=sys.stderr)

    print(f"{'candidate':24s} {'pIC50':>7s} {'IC50 nM':>12s} {'QED':>6s} {'viol':>5s}")
    for row in novels:
        print(f"{row['Molecule_Name'][:24]:24s} {row['Predicted_pIC50']:7.2f} "
              f"{row['IC50_nM']:12.2f} {row['ADMET_QED']:6.3f} {row['Violations']:5d}")

    print(f"\nKnown inhibitors for comparison ({reference_source}):")
    for row in refs:
        print(f"{row['Molecule_Name'][:24]:24s} {row['Predicted_pIC50']:7.2f} {row['IC50_nM']:12.2f}")

    if args.dock and novels:
        from src.pipeline.docking_engine import get_engine

        engine = get_engine(args.target)
        print(f"\nDocking the top {min(args.top, len(novels))} candidate(s) with AutoDock4Zn scoring:")
        for row in novels[: args.top]:
            result = engine.dock(row["clean_smiles"])
            if "error" in result:
                print(f"  {row['Molecule_Name']}: {result['error']}")
                continue
            print(f"  {row['Molecule_Name']}:")
            for pose in result["poses"]:
                distance = "n/a" if pose["zinc_distance"] is None else f"{pose['zinc_distance']:.2f} A"
                contact = "zinc contact" if pose["zinc_contact"] else "no zinc contact"
                print(f"    pose {pose['rank']}: {pose['score']:7.2f} kcal/mol   N-Zn {distance:>7s}   {contact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

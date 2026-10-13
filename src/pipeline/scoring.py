"""Turns a predicted pIC50 into a score out of 10.

A raw pIC50 means nothing to most readers, and a linear rescaling of it would
mean just as little: 7/10 would be an arbitrary point on an arbitrary axis. The
score here is a percentile instead -- where the prediction falls within the
pIC50 values actually measured against that isoform in ChEMBL. 7.0 means the
molecule is predicted more potent than 70% of the known inhibitors of that
isoform, which is a claim that can be checked.

The reference distribution is per isoform, because the isoforms are not
comparable: the measured median against CA IV is 7.6 and against CA VA 5.5, so
the same pIC50 is unremarkable for one and strong for the other.

Isoforms without enough measurements get no score at all. See
src/pipeline/isoforms.py; the app shows those structures with the potency
withheld rather than printing a number nothing supports.
"""
import numpy as np

# Below this the molecule is weaker than almost everything measured; above it,
# stronger. Both ends are reported as such rather than as 0.0 or 10.0, which
# would read as "no data" and "perfect".
FLOOR, CEILING = 0.1, 9.9


def score_out_of_ten(pic50, isoform):
    """Scores one prediction against the isoform's measured distribution.

    Returns a dict with the score, the percentile it came from, and enough
    context for the UI to say what the score is relative to.
    """
    if not isoform.is_scored:
        raise ValueError(f"{isoform.label} has no reference distribution")

    quantiles = np.asarray(isoform.quantiles)
    # quantiles[i] is the i-th percentile, so interpolating the prediction back
    # through them gives its percentile directly.
    percentile = float(np.interp(pic50, quantiles, np.arange(101),
                                 left=0.0, right=100.0))
    score = min(max(percentile / 10.0, FLOOR), CEILING)
    return {
        "score": round(score, 1),
        "percentile": round(percentile),
        "reference_isoform": isoform.label,
        "reference_n": isoform.measurements,
        "reference_median": round(float(quantiles[50]), 2),
    }


def describe_scale(isoform):
    """One sentence stating what the score is measured against."""
    if not isoform.is_scored:
        return isoform.withheld_reason
    quantiles = isoform.quantiles
    accuracy = (f" The ensemble predicts held-out {isoform.label} molecules with "
                f"R2 {isoform.test_r2:.2f} (MAE {isoform.test_mae:.2f} pIC50)."
                if isoform.test_r2 is not None else "")
    return (f"Out of 10 by percentile against {isoform.measurements} molecules "
            f"measured against {isoform.label} in ChEMBL (median pIC50 "
            f"{quantiles[50]:.1f}, so 5.0/10 means as potent as the typical "
            f"known {isoform.label} inhibitor).{accuracy}")

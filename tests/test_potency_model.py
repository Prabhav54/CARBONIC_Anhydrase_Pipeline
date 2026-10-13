"""Checks the NumPy serving model reproduces the trained scikit-learn/Keras models.

scikit-learn, TensorFlow and joblib are dev-only, so those comparisons skip
without them.

The model takes the ligand fingerprint followed by the four protein descriptors
of the isoform being screened, so these tests build features for a named
isoform rather than from SMILES alone.
"""
import numpy as np
import pytest

from src.pipeline import isoforms
from src.pipeline.potency_model import (FP_BITS, N_PROTEIN_FEATURES, PotencyModel,
                                        features, fingerprints)

DRUGS = [
    "CC(=O)Nc1nnc(s1)S(=O)(=O)N",                       # acetazolamide
    "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",                      # ethoxzolamide
    "CCN[C@H]1C[C@@H](C)S(=O)(=O)c2sc(cc12)S(N)(=O)=O",  # dorzolamide
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                        # caffeine
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                        # ibuprofen
]

N_FEATURES = FP_BITS + N_PROTEIN_FEATURES


@pytest.fixture(scope="module")
def model():
    return PotencyModel()


@pytest.fixture(scope="module")
def ca2():
    return isoforms.by_label("CA II")


@pytest.fixture(scope="module")
def X(ca2):
    matrix, valid = features(DRUGS, ca2)
    assert valid == list(range(len(DRUGS)))
    return matrix


def test_feature_layout_is_fingerprint_then_protein(X, ca2):
    assert X.shape == (len(DRUGS), N_FEATURES)
    bits = X[:, :FP_BITS]
    assert set(np.unique(bits)) <= {0.0, 1.0}
    # Every row carries the same protein descriptors, in training order.
    for row in X[:, FP_BITS:]:
        np.testing.assert_allclose(row, ca2.feature_vector(), rtol=0, atol=0)


def test_fingerprints_shape_and_bits():
    matrix, _ = fingerprints(DRUGS)
    assert matrix.shape == (len(DRUGS), FP_BITS)


def test_invalid_smiles_are_dropped(ca2):
    matrix, valid = features(["CC(=O)Nc1nnc(s1)S(=O)(=O)N", "not a molecule", "CCO"], ca2)
    assert valid == [0, 2]
    assert len(matrix) == 2


def test_a_wrong_feature_width_is_rejected(model):
    """Serving the wrong layout must fail loudly, not score the wrong columns."""
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((2, FP_BITS)))


def test_predictions_are_in_a_plausible_pic50_range(model, X):
    predictions = model.predict(X)
    assert predictions.shape == (len(DRUGS),)
    assert np.all((predictions > 2) & (predictions < 12)), predictions


def test_predictions_depend_on_the_isoform(model):
    """The whole point of the retrain: the target has to change the answer.

    The previous ensemble took the fingerprint alone, so one molecule scored
    identically against every structure.
    """
    predictions = {}
    for label in ("CA I", "CA II", "CA IX", "CA VA"):
        matrix, _ = features([DRUGS[0]], isoforms.by_label(label))
        predictions[label] = float(model.predict(matrix)[0])
    spread = max(predictions.values()) - min(predictions.values())
    assert spread > 0.1, f"prediction barely moves across isoforms: {predictions}"


def test_ensemble_is_the_documented_weighted_sum(model, X):
    expected = 0.7 * model.predict_rf(X) + 0.3 * model.predict_lstm(X)
    np.testing.assert_allclose(model.predict(X), expected, rtol=0, atol=0)


def test_random_forest_matches_sklearn(model, X):
    joblib = pytest.importorskip("joblib", reason="dev-only dependency")
    pytest.importorskip("sklearn", reason="dev-only dependency")

    reference = joblib.load("artifacts/best_ml_model.pkl").predict(X)
    np.testing.assert_allclose(model.predict_rf(X), reference, rtol=0, atol=1e-12)


def test_lstm_matches_tensorflow(model, X):
    joblib = pytest.importorskip("joblib", reason="dev-only dependency")
    keras = pytest.importorskip("tensorflow.keras", reason="dev-only dependency")

    scaler = joblib.load("artifacts/scaler.pkl")
    scaled = scaler.transform(X).reshape(len(X), 1, N_FEATURES)
    reference = keras.models.load_model(
        "artifacts/dnn_model.h5", compile=False).predict(scaled, verbose=0).flatten()
    np.testing.assert_allclose(model.predict_lstm(X), reference, rtol=0, atol=1e-4)


def test_known_inhibitors_outrank_non_binders(model, X):
    """A weak but meaningful sanity check on the trained model."""
    predictions = dict(zip(DRUGS, model.predict(X)))
    sulfonamides = [predictions[s] for s in DRUGS[:3]]
    non_binders = [predictions[s] for s in DRUGS[3:]]
    assert min(sulfonamides) > max(non_binders), predictions


def test_ranking_places_the_best_candidate_among_the_known_inhibitors():
    """The headline comparison: where the best of yours falls among the knowns.

    Both sides are model predictions, so the ranking is computed from
    Predicted_pIC50 on both and nothing is compared against a measured value.
    """
    from src.pipeline.predict_pipeline import _ranking

    novels = [{"name": "Mine", "score": 6.0, "Predicted_pIC50": 7.4},
              {"name": "Weak", "score": 2.0, "Predicted_pIC50": 5.1}]
    refs = [{"name": "A", "score": 9.9, "Predicted_pIC50": 9.2},
            {"name": "B", "score": 8.0, "Predicted_pIC50": 8.1},
            {"name": "C", "score": 3.0, "Predicted_pIC50": 6.0}]

    ranking = _ranking(novels, refs)
    assert ranking["name"] == "Mine"
    assert ranking["rank"] == 3        # two knowns are stronger
    assert ranking["of"] == 4          # itself plus the three knowns
    assert ranking["beats"] == 1       # it is ahead of C
    assert ranking["references"] == 3


def test_ranking_is_withheld_when_there_is_nothing_to_compare():
    from src.pipeline.predict_pipeline import _ranking

    unscored = [{"name": "Mine", "score": None, "Predicted_pIC50": 7.4}]
    scored = [{"name": "A", "score": 9.9, "Predicted_pIC50": 9.2}]
    assert _ranking(unscored, scored) is None     # isoform has no score at all
    assert _ranking(scored, []) is None           # no references retrieved

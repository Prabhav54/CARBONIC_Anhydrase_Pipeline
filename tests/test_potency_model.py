"""Checks the NumPy serving model reproduces the trained scikit-learn/Keras models.

The exact tolerances here are the ones measured when the serving path was
written: 1.6e-14 for the forest and 1.7e-06 for the LSTM. scikit-learn,
TensorFlow and joblib are dev-only, so those comparisons skip without them.
"""
import numpy as np
import pytest

from src.pipeline.potency_model import FP_BITS, PotencyModel, fingerprints

DRUGS = [
    "CC(=O)Nc1nnc(s1)S(=O)(=O)N",                       # acetazolamide
    "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",                      # ethoxzolamide
    "CCN[C@H]1C[C@@H](C)S(=O)(=O)c2sc(cc12)S(N)(=O)=O",  # dorzolamide
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                        # caffeine
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",                        # ibuprofen
]


@pytest.fixture(scope="module")
def model():
    return PotencyModel()


@pytest.fixture(scope="module")
def features():
    X, valid = fingerprints(DRUGS)
    assert valid == list(range(len(DRUGS)))
    return X


def test_fingerprints_shape_and_bits(features):
    assert features.shape == (len(DRUGS), FP_BITS)
    assert set(np.unique(features)) <= {0.0, 1.0}


def test_invalid_smiles_are_dropped():
    X, valid = fingerprints(["CC(=O)Nc1nnc(s1)S(=O)(=O)N", "not a molecule", "CCO"])
    assert valid == [0, 2]
    assert len(X) == 2


def test_predictions_are_in_a_plausible_pic50_range(model, features):
    predictions = model.predict(features)
    assert predictions.shape == (len(DRUGS),)
    assert np.all((predictions > 2) & (predictions < 12)), predictions


def test_ensemble_is_the_documented_weighted_sum(model, features):
    expected = 0.7 * model.predict_rf(features) + 0.3 * model.predict_lstm(features)
    np.testing.assert_allclose(model.predict(features), expected, rtol=0, atol=0)


def test_random_forest_matches_sklearn(model, features):
    joblib = pytest.importorskip("joblib", reason="dev-only dependency")
    pytest.importorskip("sklearn", reason="dev-only dependency")

    reference = joblib.load("artifacts/best_ml_model.pkl").predict(features)
    np.testing.assert_allclose(model.predict_rf(features), reference, rtol=0, atol=1e-12)


def test_lstm_matches_tensorflow(model, features):
    joblib = pytest.importorskip("joblib", reason="dev-only dependency")
    keras = pytest.importorskip("tensorflow.keras", reason="dev-only dependency")

    scaler = joblib.load("artifacts/scaler.pkl")
    scaled = scaler.transform(features).reshape(len(features), 1, FP_BITS)
    reference = keras.models.load_model("artifacts/dnn_model.h5", compile=False).predict(scaled, verbose=0).flatten()
    np.testing.assert_allclose(model.predict_lstm(features), reference, rtol=0, atol=1e-4)


def test_known_inhibitors_outrank_non_binders(model, features):
    """A weak but meaningful sanity check on the trained model."""
    predictions = dict(zip(DRUGS, model.predict(features)))
    sulfonamides = [predictions[s] for s in DRUGS[:3]]
    non_binders = [predictions[s] for s in DRUGS[3:]]
    assert min(sulfonamides) > max(non_binders), predictions

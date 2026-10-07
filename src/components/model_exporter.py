"""Exports the trained ensemble to a single NumPy archive for serving.

The web app runs on Vercel, where scikit-learn + SciPy + TensorFlow do not fit
in the function bundle. Every model in the ensemble is small enough to score
with plain NumPy, so training artifacts are flattened into arrays here and
`src.pipeline.potency_model.PotencyModel` reproduces the predictions exactly.

Training-time only: needs scikit-learn, joblib and h5py.
"""
import json
import os
import sys

import h5py
import joblib
import numpy as np

from src.exception import CustomException
from src.logger import logging

RF_PATH = os.path.join("artifacts", "best_ml_model.pkl")
SCALER_PATH = os.path.join("artifacts", "scaler.pkl")
LSTM_PATH = os.path.join("artifacts", "dnn_model.h5")
EXPORT_PATH = os.path.join("artifacts", "models", "ensemble.npz")


def _forest_arrays(rf):
    """Concatenates all trees into flat arrays with global child indices."""
    left, right, feature, threshold, value, roots = [], [], [], [], [], []
    offset = 0
    for estimator in rf.estimators_:
        tree = estimator.tree_
        is_leaf = tree.children_left == -1
        left.append(np.where(is_leaf, -1, tree.children_left + offset))
        right.append(np.where(is_leaf, -1, tree.children_right + offset))
        feature.append(np.where(is_leaf, 0, tree.feature))
        threshold.append(tree.threshold)
        value.append(tree.value[:, 0, 0])
        roots.append(offset)
        offset += tree.node_count
    return {
        "rf_left": np.concatenate(left).astype(np.int32),
        "rf_right": np.concatenate(right).astype(np.int32),
        "rf_feature": np.concatenate(feature).astype(np.int16),
        "rf_threshold": np.concatenate(threshold).astype(np.float64),
        "rf_value": np.concatenate(value).astype(np.float64),
        "rf_roots": np.array(roots, dtype=np.int32),
        "rf_max_depth": np.array(max(e.tree_.max_depth for e in rf.estimators_)),
    }


def _lstm_arrays(h5_path):
    """Reads Dense/LSTM weights from a Keras 3 .h5 file, in layer order."""
    arrays = {}
    with h5py.File(h5_path, "r") as f:
        config = json.loads(f.attrs["model_config"])
        layers = [l for l in config["config"]["layers"] if l["class_name"] in ("LSTM", "Dense")]
        for i, layer in enumerate(layers):
            name = layer["config"]["name"]
            group = f["model_weights"][name]
            weights = {}
            group.visititems(lambda n, o: weights.__setitem__(n.split("/")[-1], o[()]) if isinstance(o, h5py.Dataset) else None)
            if layer["class_name"] == "LSTM":
                if layer["config"]["activation"] != "relu" or layer["config"]["recurrent_activation"] != "sigmoid":
                    raise ValueError(f"unsupported LSTM activations in {name}")
                arrays[f"lstm{i}_kernel"] = weights["kernel"]
                arrays[f"lstm{i}_bias"] = weights["bias"]
            else:
                arrays[f"dense{i}_kernel"] = weights["kernel"]
                arrays[f"dense{i}_bias"] = weights["bias"]
        arrays["nn_layers"] = np.array([l["class_name"] for l in layers])
        arrays["nn_activations"] = np.array([l["config"]["activation"] for l in layers])
    return arrays


def export_ensemble(rf_path=RF_PATH, scaler_path=SCALER_PATH, lstm_path=LSTM_PATH, out_path=EXPORT_PATH):
    try:
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        arrays = {
            **_forest_arrays(rf),
            **_lstm_arrays(lstm_path),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **arrays)
        logging.info(f"Exported serving ensemble to {out_path}")
        return out_path
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    print(export_ensemble())

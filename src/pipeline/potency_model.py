"""pIC50 ensemble (0.7 Random Forest + 0.3 LSTM) scored with NumPy only.

Loads the archive written by `src.components.model_exporter`. Keeping inference
free of scikit-learn/TensorFlow is what lets the app fit in a Vercel function.

The model takes the 1024-bit Morgan fingerprint of the ligand followed by the
four protein descriptors of the isoform being screened, so a prediction is for a
(molecule, isoform) pair rather than for a molecule alone.
"""
import os
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

ENSEMBLE_PATH = os.path.join("artifacts", "models", "ensemble.npz")
RF_WEIGHT, LSTM_WEIGHT = 0.7, 0.3
FP_RADIUS, FP_BITS = 2, 1024  # must match training
N_PROTEIN_FEATURES = 4

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def fingerprints(smiles_list):
    """Returns (bit matrix, indices of SMILES that parsed)."""
    rows, valid = [], []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        rows.append(_morgan.GetFingerprintAsNumPy(mol))
        valid.append(i)
    return np.array(rows, dtype=np.float64).reshape(len(rows), FP_BITS), valid


def features(smiles_list, isoform):
    """Builds the model input for a list of molecules against one isoform.

    Mirrors src.components.data_transformation.build_features: fingerprint bits
    first, then the protein descriptors in their training order.
    """
    X_fp, valid = fingerprints(smiles_list)
    if not valid:
        return np.empty((0, FP_BITS + N_PROTEIN_FEATURES)), valid
    protein = np.tile(np.asarray(isoform.feature_vector(), dtype=np.float64), (len(valid), 1))
    return np.hstack([X_fp, protein]), valid


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(x, 0.0)


class PotencyModel:
    def __init__(self, path=ENSEMBLE_PATH):
        with np.load(path) as data:
            self.w = {k: data[k] for k in data.files}
        self.n_features = int(self.w["n_features"]) if "n_features" in self.w else FP_BITS

    def _check_width(self, X):
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"the exported ensemble takes {self.n_features} features but was "
                f"given {X.shape[1]}; re-run src.components.model_exporter after "
                f"retraining")

    def predict_rf(self, X):
        w = self.w
        # Walk every tree for every sample at once: nodes[s, t] is sample s's node in tree t.
        nodes = np.broadcast_to(w["rf_roots"], (len(X), len(w["rf_roots"]))).copy()
        rows = np.arange(len(X))[:, None]
        for _ in range(int(w["rf_max_depth"])):
            left = w["rf_left"][nodes]
            internal = left != -1
            if not internal.any():
                break
            # scikit-learn compares float32-cast features against float64 thresholds
            x = X.astype(np.float32)[rows, w["rf_feature"][nodes]]
            go_left = x <= w["rf_threshold"][nodes]
            nodes = np.where(internal, np.where(go_left, left, w["rf_right"][nodes]), nodes)
        return w["rf_value"][nodes].mean(axis=1)

    def predict_lstm(self, X):
        # Input is a length-1 sequence starting from zero state, so each LSTM
        # layer reduces to: c = i * relu(g); h = o * relu(c) (Keras gate order i, f, g, o).
        w = self.w
        h = (X - w["scaler_mean"]) / w["scaler_scale"]
        for i, (kind, activation) in enumerate(zip(w["nn_layers"], w["nn_activations"])):
            if kind == "LSTM":
                z = h @ w[f"lstm{i}_kernel"] + w[f"lstm{i}_bias"]
                gate_i, _, gate_g, gate_o = np.split(z, 4, axis=1)
                c = _sigmoid(gate_i) * _relu(gate_g)
                h = _sigmoid(gate_o) * _relu(c)
            else:
                h = h @ w[f"dense{i}_kernel"] + w[f"dense{i}_bias"]
                if activation == "relu":
                    h = _relu(h)
        return h[:, 0]

    def predict(self, X):
        self._check_width(X)
        return RF_WEIGHT * self.predict_rf(X) + LSTM_WEIGHT * self.predict_lstm(X)


@lru_cache(maxsize=1)
def get_model():
    """Loaded once per warm serverless instance."""
    return PotencyModel()

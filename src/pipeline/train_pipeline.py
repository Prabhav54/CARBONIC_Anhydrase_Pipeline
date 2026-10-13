"""Trains the potency ensemble and exports it for serving.

Run from the project root, with the development requirements installed:
    python -m src.pipeline.train_pipeline

It also writes artifacts/models/isoform_metrics.json: the measured test R2 and
MAE per isoform. data_curation_scripts/build_isoform_features.py uses those to
decide which isoforms the app is allowed to score, so the decision rests on
demonstrated accuracy rather than on a row count. CA XIII is why this exists:
117 measured molecules clears any sensible count threshold, but the model's R2
on its held-out molecules is about 0.0, so a 0-10 score for it would be noise
with a number on it.

Two things this reports that the previous version did not:

  * R2 on a split grouped by molecule. The old split was random over rows, and
    because 653 molecules were measured against more than one isoform, the same
    molecule appeared in both train and test. The number it printed was
    therefore partly a memorisation score.
  * An ablation against a ligand-only model. The deployed ensemble used to take
    the fingerprint alone, so its prediction could not depend on which isoform
    was being screened. The ablation is the evidence that adding the four
    protein descriptors actually buys something, rather than just making the
    output look target-specific.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, FINGERPRINT_BITS
from src.components.model_exporter import export_ensemble
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

METRICS_PATH = os.path.join("artifacts", "models", "isoform_metrics.json")


def _per_isoform_report(test_df, valid_idx, y_true, y_pred):
    """R2 and MAE per isoform: one number for the whole set hides a lot."""
    labels = test_df.iloc[valid_idx]["target_name"].to_numpy()
    rows = []
    for label in sorted(set(labels)):
        mask = labels == label
        if mask.sum() < 2:
            rows.append((label, int(mask.sum()), float("nan"), float("nan")))
            continue
        rows.append((label, int(mask.sum()),
                     r2_score(y_true[mask], y_pred[mask]),
                     mean_absolute_error(y_true[mask], y_pred[mask])))
    return rows


def _ligand_only_ablation(X_train, y_train, X_test, y_test):
    """The same forest, trained on the fingerprint alone."""
    forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    forest.fit(X_train[:, :FINGERPRINT_BITS], y_train)
    predictions = forest.predict(X_test[:, :FINGERPRINT_BITS])
    return r2_score(y_test, predictions), mean_absolute_error(y_test, predictions)


class TrainPipeline:
    def run_pipeline(self):
        """Executes the full training lifecycle from start to finish."""
        try:
            logging.info("Initiating Full Training Pipeline...")

            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_training_ingestion()

            transformer = DataTransformation()
            X_train, y_train, X_test, y_test = transformer.initiate_training_transformation(
                train_path, test_path)

            trainer = ModelTrainer()
            predictions = trainer.initiate_model_training(X_train, y_train, X_test, y_test)

            print("\n=== Ensemble on a molecule-grouped split "
                  "(every test molecule is unseen) ===")
            print(f"  features        : {X_train.shape[1]} "
                  f"({FINGERPRINT_BITS} fingerprint + 4 protein)")
            print(f"  train / test    : {len(y_train)} / {len(y_test)} measurements")
            print(f"  R2              : {r2_score(y_test, predictions):.4f}")
            print(f"  MAE (pIC50)     : {mean_absolute_error(y_test, predictions):.4f}")

            test_df = pd.read_csv(test_path)
            _, valid_idx = transformer.build_features(test_df)
            report = _per_isoform_report(test_df, valid_idx, y_test, predictions)
            print("\n  per isoform:")
            for label, n, r2, mae in report:
                print(f"    {label:9s} n={n:5d}  R2={r2:7.4f}  MAE={mae:.4f}")

            metrics = {
                "split": "grouped by molecule (GroupShuffleSplit, test_size=0.2, seed 42)",
                "n_features": int(X_train.shape[1]),
                "overall": {
                    "n_test": int(len(y_test)),
                    "r2": round(float(r2_score(y_test, predictions)), 4),
                    "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
                },
                "per_isoform": {
                    label: {"n_test": n,
                            "r2": None if np.isnan(r2) else round(float(r2), 4),
                            "mae": None if np.isnan(mae) else round(float(mae), 4)}
                    for label, n, r2, mae in report
                },
            }
            os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
            with open(METRICS_PATH, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n  wrote {METRICS_PATH}")

            print("\n=== Ablation: does the target actually matter? ===")
            r2_ligand, mae_ligand = _ligand_only_ablation(X_train, y_train, X_test, y_test)
            print(f"  fingerprint only      : R2={r2_ligand:7.4f}  MAE={mae_ligand:.4f}")
            print(f"  fingerprint + protein : "
                  f"R2={r2_score(y_test, predictions):7.4f}  "
                  f"MAE={mean_absolute_error(y_test, predictions):.4f}")

            export_path = export_ensemble()
            print(f"\nExported serving ensemble -> {export_path}")
            logging.info("Training Pipeline Completed Successfully.")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    TrainPipeline().run_pipeline()

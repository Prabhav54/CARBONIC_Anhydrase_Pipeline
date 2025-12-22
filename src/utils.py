import os
import sys
import pickle
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise Exception(f"Error saving object: {e}")

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Exception(f"Error loading object: {e}")

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Trains multiple models and returns a report of their R2 scores.
    Implements the 'Gemstone' GridSearch logic.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[model_name]

            print(f"   --> Training {model_name}...")

            # Hyperparameter Tuning
            gs = RandomizedSearchCV(model, para, cv=3, n_iter=5, n_jobs=-1, verbose=0, random_state=42)
            gs.fit(X_train, y_train)

            # Train with best params
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict & Score
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            print(f"       Score: {test_model_score:.4f}")

        return report

    except Exception as e:
        raise Exception(f"Error during model evaluation: {e}")
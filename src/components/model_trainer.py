import os
import sys
from dataclasses import dataclass

# --- IMPORT MODELS ---
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        """
        Helper function to loop through models and evaluate them.
        """
        report = {}
        logging.info(f"Starting Model Tournament with {len(models)} contenders...")

        for i, (name, model) in enumerate(models.items()):
            try:
                # Train
                model.fit(X_train, y_train) 

                # Predict
                y_test_pred = model.predict(X_test)
                
                # Score (R2)
                test_model_score = r2_score(y_test, y_test_pred)

                report[name] = test_model_score
                logging.info(f"  > {name:<25} R2 Score: {test_model_score:.4f}")

            except Exception as e:
                logging.warning(f"  > {name} failed: {str(e)}")
                report[name] = -float('inf') # Penalize failed models

        return report

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # --- DEFINING THE CONTENDERS ---
            models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=-1),
                "CatBoostRegressor": CatBoostRegressor(verbose=False, allow_writing_files=False),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
            }

            # --- START THE TOURNAMENT ---
            model_report: dict = self.evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models
            )

            # --- FIND THE WINNER ---
            # Sort scores to find the best
            best_model_score = max(sorted(model_report.values()))
            
            # Get the name of the best model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            logging.info("---------------------------------------------------------")
            logging.info(f"🏆 TOURNAMENT WINNER: {best_model_name}")
            logging.info(f"   Score (R2 on Unseen Isoforms): {best_model_score:.4f}")
            logging.info("---------------------------------------------------------")

            if best_model_score < 0.1:
                logging.warning("No model performed well. Consider adding more features or checking data quality.")

            # --- SAVE THE CHAMPION ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            
            return r2

        except Exception as e:
            raise CustomException(e, sys)
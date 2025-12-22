import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        print("🤖 [3/3] Model Training Started ")
        try:
            # Split Data
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # 1. Define Models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
    #            "Gradient Boosting": GradientBoostingRegressor(),
     #           "XGBRegressor": XGBRegressor(),
    #            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    #            "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # 2. Define Hyperparameters
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [None, 10, 20, 30]
                },
      #         "Gradient Boosting": {
      #              'learning_rate': [0.1, 0.01, 0.05, 0.001],
      #              'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
      #              'n_estimators': [8, 16, 32, 64, 128, 256]
      #          },
       #         "XGBRegressor": {
      #              'learning_rate': [0.1, 0.01, 0.05, 0.001],
      #              'n_estimators': [8, 16, 32, 64, 128, 256]
      #          },
      #          "CatBoosting Regressor": {
      #              'depth': [6, 8, 10],
      #              'learning_rate': [0.01, 0.05, 0.1],
      #              'iterations': [30, 50, 100]
      #          },
      #          "AdaBoost Regressor": {
      #              'learning_rate': [0.1, 0.01, 0.5, 0.001],
      #              'n_estimators': [8, 16, 32, 64, 128, 256]
      #          }
            }

            # 3. Evaluate All Models
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # 4. Get Best Model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"\n🏆 Best Model Found: {best_model_name} (R2 Score: {best_model_score:.4f})")

            if best_model_score < 0.6:
                print("⚠️ Warning: No model achieved > 0.6 R2 score.")

            # 5. Save Best Model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print("✅ Model Saved Successfully")
            return best_model_score

        except Exception as e:
            raise Exception(f"Model Training Error: {e}")
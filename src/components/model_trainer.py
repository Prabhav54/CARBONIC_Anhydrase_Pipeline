import os
import sys
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    scaler_path: str = os.path.join("artifacts", "scaler.pkl")
    rf_model_path: str = os.path.join("artifacts", "best_ml_model.pkl")
    lstm_model_path: str = os.path.join("artifacts", "dnn_model.h5")

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def build_lstm_model(self, input_shape):
        """Builds the Keras LSTM Architecture."""
        model = Sequential([
            LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        """Trains the models, calculates the Ensemble score, and saves artifacts."""
        try:
            logging.info("Starting Model Training Pipeline...")
            os.makedirs("artifacts", exist_ok=True)

            # --- 1. Train & Save Scaler ---
            print("   -> Fitting StandardScaler...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            joblib.dump(scaler, self.trainer_config.scaler_path)
            print(f"   -> Created: {self.trainer_config.scaler_path}")

            # --- 2. Train & Save Random Forest ---
            print("   -> Training Random Forest Regressor (This might take a minute)...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train) 
            
            joblib.dump(rf_model, self.trainer_config.rf_model_path)
            print(f"   -> Created: {self.trainer_config.rf_model_path}")

            # --- 3. Train & Save LSTM ---
            print("   -> Training Deep Learning LSTM Model...")
            X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

            lstm_model = self.build_lstm_model(input_shape=(1, X_train_scaled.shape[1]))
            
            # Training the Deep Learning model (10 epochs for speed)
            lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, 
                           validation_data=(X_test_lstm, y_test), verbose=0)
            
            lstm_model.save(self.trainer_config.lstm_model_path)
            print(f"   -> Created: {self.trainer_config.lstm_model_path}")
            
            # --- 4. Evaluate the Ensemble (0.7 RF + 0.3 LSTM) ---
            print("   -> Evaluating 0.7 RF + 0.3 LSTM Ensemble on Test Data...")
            
            rf_preds = rf_model.predict(X_test)
            # .flatten() ensures the LSTM output shape matches the RF output shape
            lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten() 
            
            # The official consensus formula
            ensemble_preds = (0.7 * rf_preds) + (0.3 * lstm_preds)
            
            return ensemble_preds

        except Exception as e:
            raise CustomException(e, sys)

# =====================================================================
# TEST BLOCK: THIS LINKS INGESTION -> TRANSFORMATION -> TRAINING
# =====================================================================
if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    print("🚀 Starting the FULL Training Pipeline...")
    try:
        print("\n[Step 1] Ingesting Data...")
        ingestor = DataIngestion()
        train_path, test_path = ingestor.initiate_training_ingestion()

        print("\n[Step 2] Transforming Data into Fingerprints...")
        transformer = DataTransformation()
        X_train, y_train, X_test, y_test = transformer.initiate_training_transformation(train_path, test_path)

        print("\n[Step 3] Training Models and Generating Ensemble Score...")
        trainer = ModelTrainer()
        ensemble_predictions = trainer.initiate_model_training(X_train, y_train, X_test, y_test)

        # Step 4: Evaluate the Ensemble
        score = r2_score(y_test, ensemble_predictions)
        print(f"\n✅ FULL Pipeline Successful!")
        print(f"🎯 Real Data ENSEMBLE R2 Score (0.7 RF + 0.3 LSTM): {score:.4f}")
        print("📂 Look inside your 'artifacts/' folder. The .pkl and .h5 files are ready for deployment!")

    except Exception as e:
        print(f"\n❌ Error during full pipeline run: {e}")
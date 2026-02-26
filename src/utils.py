import os
import sys
import joblib
from src.exception import CustomException
from src.logger import logging
import tensorflow as tf

def load_object(file_path):
    """Loads a pickle object (like the Scaler or Random Forest)."""
    try:
        logging.info(f"Loading object from {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)

def load_keras_model(file_path):
    """Loads the Deep Learning (LSTM/DNN) .h5 model."""
    try:
        logging.info(f"Loading Keras model from {file_path}")
        # compile=False avoids the custom metrics deserialization error you faced earlier
        return tf.keras.models.load_model(file_path, compile=False)
    except Exception as e:
        raise CustomException(e, sys)
import sys

import joblib

from src.exception import CustomException
from src.logger import logging


def load_object(file_path):
    """Loads a pickled object (the scaler or the Random Forest)."""
    try:
        logging.info(f"Loading object from {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def load_keras_model(file_path):
    """Loads the Keras .h5 model.

    TensorFlow is a training-time dependency and is not installed in the
    deployed app, so it is imported only when this is actually called.
    Serving uses src.pipeline.potency_model instead.
    """
    try:
        import tensorflow as tf

        logging.info(f"Loading Keras model from {file_path}")
        # compile=False skips deserialising the training-time metrics.
        return tf.keras.models.load_model(file_path, compile=False)
    except Exception as e:
        raise CustomException(e, sys)

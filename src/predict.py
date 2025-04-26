import joblib
import pandas as pd
import os

from config import MODELS_DIR

def load_model(model_name):
    """
    Load a saved scikit-learn model pipeline.

    Args:
        model_name (str): Saved model name

    Returns:
        model: Loaded pipeline object
    """
    model_path = os.path.join(MODELS_DIR, model_name + '.joblib')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_risk(model, new_data):
    """
    Predict risk (0 or 1) for new patient data.

    Args:
        model (Pipeline): Trained model
        new_data (pd.DataFrame): New data with the same schema as train data

    Returns:
        pd.Series: Predicted labels (0 = No Disease, 1 = Disease)
    """
    return model.predict(new_data)


def predict_risk_proba(model, new_data):
    """
    Predict risk probabilities for new patient data.

    Args:
        model (Pipeline): Trained model
        new_data (pd.DataFrame): New data

    Returns:
        np.ndarray: Array with two columns [P(No Disease), P(Disease)]
    """
    return model.predict_proba(new_data)
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import TEST_PROPORTION, RANDOM_STATE, MODELS_DIR
from src.model_factory import build_pipeline


def train_model(
        X,
        y,
        model_name, 
        preprocessor,
        test_size=TEST_PROPORTION, 
        random_state=RANDOM_STATE):
    """
    Train a logistic regression model using a preprocessing pipeline.

    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        model_path (str): Path to save the trained model
        test_size (float): Proportion of the dataset to include in the test
        random_state (int): Random seed

    Returns:
        pipeline (Pipeline): Trained scikit-learn pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    """
    #print("🧪 Loaded dataset:")
    #print(X.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline(preprocessor)

    pipeline.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save the trained model
    model_path = os.path.join(MODELS_DIR, model_name + '.joblib')
    joblib.dump(pipeline, model_path)

    return pipeline, X_test, y_test
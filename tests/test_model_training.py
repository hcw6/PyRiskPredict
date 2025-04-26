import os
from src.training import train_model
from src.data_processing import build_preprocessor
from data.load_data import load_data__pima_diabetes
from config import MODEL_NAME

def test_train_model_and_save():
    X, y, numeric, categorical, passthrough = load_data__pima_diabetes()
    preprocessor = build_preprocessor(numeric, categorical, passthrough)

    pipeline, X_test, y_test = train_model(X, y, MODEL_NAME, preprocessor)

    # Check that pipeline was trained and can predict
    preds = pipeline.predict(X_test)
    assert len(preds) == len(y_test)

    # Check that model file was saved
    model_path = os.path.join("models", f"{MODEL_NAME}.joblib")
    assert os.path.exists(model_path), f"Model file {model_path} not found"

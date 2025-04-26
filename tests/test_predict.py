import pandas as pd
from src.predict import load_model, predict_risk
from src.training import train_model
from src.data_processing import build_preprocessor
from data.load_data import load_data__pima_diabetes

def test_predict_risk():
    X, y, numeric, categorical, passthrough = load_data__pima_diabetes()
    preprocessor = build_preprocessor(numeric, categorical, passthrough)
    model_name = "test_predict_model"

    pipeline, _, _ = train_model(X, y, model_name, preprocessor)
    model = load_model(model_name)

    new_data = X.iloc[:3].copy()
    preds = predict_risk(model, new_data)

    assert len(preds) == 3
    assert all(p in [0, 1] for p in preds), "Predictions must be binary"

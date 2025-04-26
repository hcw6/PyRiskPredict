import pytest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.evaluation import evaluate_model
from src.training import train_model
from src.data_processing import build_preprocessor
from data.load_data import load_data__pima_diabetes

@pytest.fixture
def trained_pipeline():
    X, y, numeric, categorical, passthrough = load_data__pima_diabetes()
    preprocessor = build_preprocessor(numeric, categorical, passthrough)
    pipeline, X_test, y_test = train_model(
        X, 
        y, 
        "test_eval_model", 
        preprocessor)
    return pipeline, X_test, y_test

def test_evaluate_model_does_not_crash(trained_pipeline):
    pipeline, X_test, y_test = trained_pipeline
    try:
        evaluate_model(pipeline, X_test, y_test)
    except Exception as e:
        pytest.fail(f"evaluate_model crashed: {e}")

def test_metrics_are_computable(trained_pipeline):
    pipeline, X_test, y_test = trained_pipeline
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # these should run without error
    _ = classification_report(y_test, y_pred)
    _ = confusion_matrix(y_test, y_pred)
    _ = roc_auc_score(y_test, y_prob)

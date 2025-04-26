import pytest
from src.model_factory import get_model

@pytest.mark.parametrize("algo_id, expected_type", [
    (0, "LogisticRegression"),
    (1, "RandomForestClassifier"),
    (2, "XGBClassifier"),
])
def test_get_model(monkeypatch, algo_id, expected_type):
    import src.model_factory as mf
    monkeypatch.setattr(mf, "ALGORITHM_ID", algo_id)
    model = mf.get_model()
    assert model.__class__.__name__ == expected_type

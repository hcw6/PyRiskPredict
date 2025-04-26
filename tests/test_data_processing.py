from src.data_processing import build_preprocessor
import pandas as pd

def test_preprocessor_fit_transform():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, None],
        "cat1": ["a", "b", "a"],
        "pass1": [0, 1, 1]
    })

    numeric = ["num1"]
    categorical = ["cat1"]
    passthrough = ["pass1"]

    preprocessor = build_preprocessor(numeric, categorical, passthrough)
    transformed = preprocessor.fit_transform(df)
    
    # Should return a numpy array with shape (3, n_features)
    assert transformed.shape[0] == df.shape[0]
    assert transformed.shape[1] >= len(numeric) + len(passthrough)

    # Check for no NaNs in the transformed output
    assert not pd.isnull(transformed).any(),"Transformed output contains NaN"
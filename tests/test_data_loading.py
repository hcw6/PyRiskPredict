import pandas as pd
import numpy as np

from data.load_data import (
    load_data__heart_disease,
    load_data__surviving_on_titanic,
    load_data__pima_diabetes,
    load_data__breast_cancer
)

def test_heart_disease():
    check_feature_lists(load_data__heart_disease)

def test_titanic():
    check_feature_lists(load_data__surviving_on_titanic)

def test_pima_diabetes():
    check_feature_lists(load_data__pima_diabetes)

def test_breast_cancer():
    check_feature_lists(load_data__breast_cancer)


def check_feature_lists(function):
    X, y, numeric, categorical, passthrough = function()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.isnull().all().any()
    assert len(X) == len(y)

    for col in numeric:
        assert (
            np.issubdtype(X[col].dtype, np.number),
            f"{col} should be numeric"
        )

    t = ['object', 'category']
    for col in categorical:
        assert X[col].dtype.name in t, f"{col} should be categorical"

    all_lists = numeric + categorical + passthrough
    assert (
        all(col in X.columns for col in all_lists),
        "Some columns in lists not found in X"
    )
    assert (
        len(set(numeric) & set(categorical)) == 0,
        "Overlap between numeric and categorical"
    )
    assert (
        len(set(numeric) & set(passthrough)) == 0,
        "Overlap between numeric and passthrough"
    )
    assert (
        len(set(categorical) & set(passthrough)) == 0,
        "Overlap between categorical and passthrough"
    )

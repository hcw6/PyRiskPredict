from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(numeric: list, categorical: list, passthrough: list):
    """
    Build a preprocessing pipeline for the heart disease dataset.

    Returns:
        ColumnTransformer: preprocessing pipeline that handles numeric,
        categorical, and passthrough features
    """
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    passthrough_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric),
        ('cat', categorical_transformer, categorical),
        ('pass', passthrough_transformer, passthrough)
    ])

    return preprocessor
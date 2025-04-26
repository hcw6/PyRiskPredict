from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import (
    ALGORITHM_ID,
    RANDOM_STATE,
    LOGR_MAX_ITER,
    RANDF_N_ESTIMATORS,
    XGB_USE_LABEL_ENCODER,
    XGB_EVAL_METRICS
)

# ML_ALGORITHM:
# 0 — Logistic Regression
# 1 — Random Forest
# 2 — XGBoost

def get_model():
    match ALGORITHM_ID:
        case 0:
            return LogisticRegression(max_iter=LOGR_MAX_ITER)
        case 1:
            return RandomForestClassifier(
                n_estimators=RANDF_N_ESTIMATORS,
                random_state=RANDOM_STATE)
        case 2:
            return XGBClassifier(
                use_label_encoder=XGB_USE_LABEL_ENCODER,
                eval_metric=XGB_EVAL_METRICS, 
                random_state=RANDOM_STATE)
        case _:
            raise ValueError(f"Unsupported ML_ALGORITHM: {ALGORITHM_ID}")


def build_pipeline(preprocessor):
    model = get_model()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline
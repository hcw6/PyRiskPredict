TEST_PROPORTION = 0.2
MODELS_DIR = 'models'
RANDOM_STATE = 12

LOGR_MAX_ITER = 1000
RANDF_N_ESTIMATORS = 100
XGB_USE_LABEL_ENCODER = False
XGB_EVAL_METRICS = 'logloss'


# execution setup: select dataset and ML algorithm
ALGORITHM_ID = 0
DATASET_ID = 0

DATASET_NAMES = {
    0: "UCI Heart Disease",
    1: "Titanic",
    2: "PIMA Indians Diabetes",
    3: "Breast Cancer Wisconsin"
}

ALGORITHM_NAMES = {
    0: "Logistic Regression",
    1: "Random Forest",
    2: "XGBoost"
}

match DATASET_ID:
    case 0:
        NEGATIVE_CLASS_NAME = "No Disease"
        POSITIVE_CLASS_NAME = "Disease"
    case 1:
        NEGATIVE_CLASS_NAME = "Not Survived"
        POSITIVE_CLASS_NAME = "Survived"
    case 2:
        NEGATIVE_CLASS_NAME = "No Diabetes"
        POSITIVE_CLASS_NAME = "Diabetes"
    case 3:
        NEGATIVE_CLASS_NAME = "Benign"
        POSITIVE_CLASS_NAME = "Malignant"
    case _:
        raise ValueError(f"Unsupported DATASET_ID: {DATASET_ID}")
    
MODEL_NAME = (
    DATASET_NAMES[DATASET_ID].lower().replace(' ','_') + '_(' +
    ALGORITHM_NAMES[ALGORITHM_ID].lower().replace(' ','_')+ ')'
)


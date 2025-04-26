import pandas as pd

from src.data_processing import build_preprocessor
from src.training import train_model
from src.evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve
)
from config import (
    NEGATIVE_CLASS_NAME,
    POSITIVE_CLASS_NAME,
    DATASET_ID,
    MODEL_NAME,
    ALGORITHM_ID,
    ALGORITHM_NAMES,
    DATASET_NAMES
)


match DATASET_ID:
    case 0:
        from data.load_data import load_data__heart_disease as load_dataset
    case 1:
        from data.load_data import (
            load_data__surviving_on_titanic as load_dataset)
    case 2:
        from data.load_data import load_data__pima_diabetes as load_dataset
    case 3:
        from data.load_data import load_data__breast_cancer as load_dataset
    case _:
        raise ValueError(f"Unsupported DATASET_ID: {DATASET_ID}")


if __name__ == "__main__":
    print(f"🚀 Running {ALGORITHM_NAMES[ALGORITHM_ID]} on {DATASET_NAMES[DATASET_ID]}")

    X, y, numeric, categorical, passthrough  = load_dataset()
        
    preprocessor = build_preprocessor(numeric, categorical, passthrough)

    pipeline, X_test, y_test = train_model(X, y, MODEL_NAME, preprocessor)

    evaluate_model(pipeline, X_test, y_test)

    plot_confusion_matrix(
        pipeline,
        X_test,
        y_test, 
        class_names=[NEGATIVE_CLASS_NAME ,POSITIVE_CLASS_NAME])

    plot_roc_curve(pipeline, X_test, y_test)



    from src.predict import load_model, predict_risk
    log_reg_model = load_model(model_name = MODEL_NAME)

    new_data = pd.DataFrame([
        {
            "age": 60,
            "trestbps": 145,
            "chol": 233,
            "thalach": 150,
            "oldpeak": 2.3,
            "cp": 3,
            "restecg": 0,
            "slope": 0,
            "thal": 2,
            "sex": 1,
            "fbs": 0,
            "exang": 1,
            "ca": 0
        },
        {
            "age": 67,
            "trestbps": 180,
            "chol": 564,
            "thalach": 90,
            "oldpeak": 4.2,
            "cp": 1,
            "restecg": 1,
            "slope": 2,
            "thal": 1,
            "sex": 1,
            "fbs": 1,
            "exang": 1,
            "ca": 3
        }])
    predictions = predict_risk(log_reg_model, new_data)
    print(f"Risk prediction for healthy patient: {predictions[0]}")
    print(f"Risk prediction for seek patient: {predictions[1]}")
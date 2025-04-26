from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate the trained model and print performance metrics.

    Args:
        pipeline (Pipeline): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True labels
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def plot_confusion_matrix(
        pipeline, 
        X_test, 
        y_test, 
        class_names):
    """
    Plot confusion matrix for classification results.

    Args:
        pipeline (Pipeline): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True labels
        class_names (tuple): Class labels First is 0, Second is 1
    """
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d",
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(
        pipeline, 
        X_test, 
        y_test):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()
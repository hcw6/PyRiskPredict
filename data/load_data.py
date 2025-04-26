from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_data__heart_disease():
    """
    Load and preprocess the UCI Heart Disease dataset.
    Returns:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Binary target series (0 = no disease, 1 = disease)
    """
    df = fetch_ucirepo(id=45)  
    X = df.data.features.copy()
    y = df.data.targets.copy()

    # Convert target to binary: 0 = no disease, 1 = any level of disease
    y_binary = (y['num'] > 0).astype(int)

    X["cp"] = X["cp"].astype("category")
    X["restecg"] = X["restecg"].astype("category")
    X["slope"] = X["slope"].astype("category")
    X["thal"] = X["thal"].astype("category")

    numeric = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical = ['cp', 'restecg', 'slope', 'thal']
    passthrough = ['sex', 'fbs', 'exang', 'ca']

    return X, y_binary, numeric, categorical, passthrough


def load_data__surviving_on_titanic():
    """
    Load Titanic dataset from CSV and preprocess basic features.

    Returns:
        X (pd.DataFrame): Feature dataframe
        y (pd.Series): Binary target (0 = Died, 1 = Survived)
        numeric, categorical, passthrough: Feature lists for processing
    """
    def extract_name_parts(df):
        name_split = df["Name"].str.extract(r'(?P<LastName>[^,]+), (?P<Title>[^.]+)\. (?P<FirstName>.+)')
        df["LastName"] = name_split["LastName"]
        df["Title"] = name_split["Title"].str.strip()
        df["FirstName"] = name_split["FirstName"].str.strip()

        df["Title"] = df["Title"].replace({
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Countess": "Royalty",
            "Lady": "Royalty",
            "Sir": "Royalty",
            "Don": "Royalty",
            "Jonkheer": "Royalty",
            "Dr": "Professional",
            "Rev": "Professional",
            "Col": "Military",
            "Major": "Military",
            "Capt": "Military"
        })
        return df

    df = pd.read_csv("data/titanic_data.csv")

    y = df["Survived"]

    df = extract_name_parts(df)
    df["Pclass"] = df["Pclass"].astype("category")

    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"]].copy()

    numeric = ["Age", "SibSp", "Parch", "Fare"]
    categorical = ["Sex", "Embarked", "Pclass", "Title"]
    passthrough = []

    return X, y, numeric, categorical, passthrough


def load_data__pima_diabetes():
    """
    Load and preprocess the PIMA Indians Diabetes dataset (Kaggle).
    
    Returns:
        X (pd.DataFrame): Feature dataframe with patient health indicators.
        y (pd.Series): Binary target (0 = No Diabetes, 1 = Diabetes)
        numeric (List[str]): List of numeric feature names
        categorical (List[str]): Empty (all features are numeric)
        passthrough (List[str]): Empty
    """    
    df = pd.read_csv("data/diabetes.csv")

    y = df["Outcome"]
    X = df.drop(columns=["Outcome"])

    numeric = X.columns.tolist()
    categorical = []
    passthrough = []

    return X, y, numeric, categorical, passthrough


def load_data__breast_cancer():
    """
    Load and preprocess the Breast Cancer Wisconsin dataset (diagnostic version).

    Returns:
        X (pd.DataFrame): Feature dataframe with cell nucleus measurements
        y (pd.Series): Binary target (0 = Benign, 1 = Malignant)
        numeric (List[str]): All features are numeric
        categorical (List[str]): Empty
        passthrough (List[str]): Empty
    """
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    y_binary = (y.iloc[:, 0] == "M").astype(int)  
    # M = malignant = 1
    # "diagnosis" → 0/1

    numeric = X.columns.tolist()
    categorical = []
    passthrough = []

    return X, y_binary, numeric, categorical, passthrough

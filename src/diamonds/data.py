import pandas as pd
import seaborn as sns
# Import other necessary libraries here
# ML
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(cache = True) -> pd.DataFrame:
    """
    Load the diamonds dataset.

    Parameters
    ----------
    cache : bool, optional
        Whether to cache the dataset, by default True

    Returns
    -------
    pd.DataFrame
        The diamonds dataset
    """
    df = sns.load_dataset("diamonds", cache=cache)
    print("Loading the diamonds dataset...")
    print(df.head)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The diamonds dataset

    Returns
    -------
    pd.DataFrame
        The cleaned diamonds dataset
    """
    print("Cleaning the diamonds dataset...")
    num_rows_with_na = df.isna().any(axis=1).sum()
    df = df.dropna()
    print(f"Number of rows with missing values: {num_rows_with_na}")

    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned diamonds dataset

    Returns
    -------
    pd.DataFrame
        The preprocessed diamonds dataset
    """
    print("Preprocessing the diamonds dataset...")
    # Implement preprocessing steps here (e.g., encoding categorical variables, feature engineering, etc.)
    df_cat = df.select_dtypes(include="category")
    print(df_cat.describe())
    cat_pipe = Pipeline(
    [ ("cat_imp",SimpleImputer(strategy="most_frequent"))
      ,("ohe",OneHotEncoder(drop="first",sparse_output=False))
        ])

    num_pipe = Pipeline(
    [("knn_imp", KNNImputer(n_neighbors=5))
     ,("scaler", StandardScaler())
      ])
    
    preprocessor = ColumnTransformer(
    [("numeric",num_pipe, make_column_selector(dtype_include="number"))
    ,("categorical", cat_pipe, make_column_selector(dtype_exclude="number"))
      ]).set_output(transform="pandas")

    df_processed = preprocessor.fit_transform(df)
    print("Preprocessing completed.")

    print(df_processed.head())

    return df_processed

def create_X_y(df: pd.DataFrame) ->tuple[pd.DataFrame, pd.Series]:
    """
    Create the feature matrix X and target vector y from the diamonds dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed diamonds dataset

    Returns
    -------
    (pd.DataFrame, pd.Series)
        The feature matrix X and target vector y
    """
    print("Creating feature matrix X and target vector y...")
    X = df.drop(columns=["price"])
    y = df["price"]
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print(X.head())
    print(y.head())

    print("Feature matrix X and target vector y created.")

    return X, y


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    X, y = create_X_y(df_clean)
    df_preprocessed = preprocess_data(df_clean)

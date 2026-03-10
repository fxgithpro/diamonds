import pandas as pd
import seaborn as sns
# Import other necessary libraries here
# ML
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diamonds import logger
from diamonds.model import create_preproc
from diamonds.registry import load_prepoc,save_preproc
from diamonds.params import MODEL_PATH

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
    # Remove null entries
    num_rows_with_na = df.isna().any(axis=1).sum()
    df = df.dropna()
    logger.info(f"Number of rows with missing values: {num_rows_with_na}")
    
    # Remove the rows containing '0' in one cell 
    rows = len(df)
    def keep_not_null(row) :
        if 0 in row.values : return False
        return True
    
    df_clean = df[df.apply(keep_not_null,axis=1)]
    logger.info(f"Cleaned the rows {rows} rows : {len(df_clean)}")

    return df

def preprocess_data(df: pd.DataFrame, train=False) -> pd.DataFrame:
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
    # Allow the reuse the previous built pipeline 
    if not train:
        preprocessor = create_preproc()
        preprocessor = preprocessor.fit(df)
        save_preproc(preprocessor,MODEL_PATH)
    else:
        preprocessor = load_prepoc()
        
    df_processed = preprocessor.transform(df)
    print("Preprocessing completed.")

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

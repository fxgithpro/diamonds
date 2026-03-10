# ML
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def create_model(model_name: str) -> BaseEstimator:
    """
    Create an untrained model with the best hyperparameters found during tuning.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g. "ridge", "random_forest")

    Returns
    -------
    BaseEstimator
        The model ready to be fitted
    """ 

    if model_name == "LinearRegression":
         return LinearRegression()
    elif model_name == "random_forest":
        return RandomForestRegressor(n_estimators=100, max_depth=10)
    elif model_name == "KNN":
        return KNeighborsRegressor(n_neighbors=5)
    elif model_name == "SVR":
        return SVR(kernel='rbf', C=1.0, epsilon=0.1)
    else : return None

def create_preproc() -> Pipeline:
    """
    Create a preprocessing pipeline.
    """
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

    return preprocessor

def train_model(model, X_train, y_train):
    """
    Train the model 
    
    Parameters
    ----------
        - X_train : input data to use for training the model
        - y_train : expected results
        
    Returns
    -------
        The trained model
    """
    trained_model = model.fit(X_train, y_train)
    return trained_model

def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """
    Launch a prediction and compute some metrics to evaluate
    
    Returns
    -------
     A dictionnary with
      - mean absolute error
      - mean squared error
      - r^2 score
      - mean absolute percentage error
      
      Format of the dictionnary
      {"mae":mae,"mse":mse,"r2":r2,"mape":mape}
     
    """
    # NB : mae, mse, r2_score, mape
    # Only print the metrics for now
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    r2  = r2_score(y_test,y_pred) 
    mape = mean_absolute_percentage_error(y_test,y_pred)
    scores = {"mae":mae,"mse":mse,"r2":r2,"mape":mape}
    print(scores)
    return scores

def predict(model, X):
    """
    Make predictions using the trained model.

    Parameters
    ----------
    model : any
        The trained model
    X : pd.DataFrame
        The raw data

    Returns
    -------
    pd.Series
        The predicted values
    """
    y_pred = model.predict(X)
    return y_pred

if __name__ == "__main__":
    import seaborn as sns
    print("Creating model...")
    model = create_model("random_forest")
    print(model)
    print("Creating preprocessor...")
    toto =sns.load_dataset("diamonds")
    titi=toto.drop(columns=["price"])
    preprocessor= create_preproc()
    tata=preprocessor.fit_transform(titi)
    print(tata)
    print(preprocessor)

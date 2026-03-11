
import os
import pickle
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_FOLDER
 
import mlflow
import loguru

logger = loguru.logger

def save_model(estimator: BaseEstimator, name : str,registry:str="local"):
    """Save the model to the specified path."""
    # Implement the logic to save the model (e.g., using pickle, joblib, etc.)
    estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
    with open(estimator_path, "wb") as f:
        pickle.dump(estimator, f)
        logger.info(f"Save model {name} in local")
        
    if registry == "mlflow":
        mlflow.sklearn.log_model(estimator,name="diamonds_model")
        logger.info(f"Save model {name} on mlflow server")
        
        
def load_model(name: str, registry:str="local") -> BaseEstimator:
    """Load the model from the specified path."""
    if registry == "mlflow":
        model_uri = f"models/diamonds/latest"
        estimator = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Load model from mlflow server")
    else :
        estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
        with open(estimator_path, "rb") as f:
            estimator = pickle.load(f)
            logger.info(f"Load model from local")
        
    
    return estimator
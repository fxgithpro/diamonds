import os

DATA_PATH= "data"
MODEL_FOLDER = "models"


MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "local")
MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "local")
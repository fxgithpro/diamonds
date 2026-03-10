from diamonds import data
from diamonds import model
from diamonds import params
from diamonds import registry
from diamonds import logger

import sys 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train(
    model_name: str = "baseline",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Simple end‑to‑end pipeline:

    - load and clean the raw data
    - preprocess it and build X, y
    - split into train / test
    - build the model and preprocessing
    - train, evaluate, and save the trained model
    """
    # 1) Data
    print("Load data...")
    df_data = data.load_data(params.MODEL_REGISTRY=="local")
    
    if df_data.size() == 0:
        logger.warning("Empty data set")
        sys.exit(1)
    
    print("Clean data...")
    df_clean = data.clean_data(df_data)
    
    print("Split data...")
    X,y = data.create_X_y(df_clean)
    
    # Not necessary because the preprocessing is done further
    #df_preproc = data.preprocess_data(df_clean)
    
    X_train, X_test, y_train, y_test  = train_test_split(X,y, random_state=random_state)
  
    # 2) Model + preprocessing
    print(f"Build not trained model {model_name} ...")
    built_model = model.create_model(model_name)
    
    if built_model is None or type(built_model)!=RandomForestRegressor:
        logger.error("The model built is not from the expected type")
        sys.exit(1)
        
    print("Build preprocessor pipeline...")
    preprocessor = model.create_preproc()
    
    print("Preprocessing the data...")
    preprocessor.fit(X_train)
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled  = preprocessor.transform(X_test)
    
    #Train the model 
    print("Trained the model...")
    trained_model = model.train_model(built_model,X_train_scaled,y_train)
    
    # 3) Evaluation
    print("Evaluate model...")
    model.evaluate_model(trained_model,X_test_scaled,y_test)
  
    # 4) Persistence
    print("Save data...")
    registry.save_preproc(preprocessor,".")
    registry.save_model(trained_model,".")
    
    print("Model ready to be used!!!")


if __name__ == "__main__":
    train("random_forest")


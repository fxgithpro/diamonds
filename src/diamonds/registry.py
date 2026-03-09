import os
import pickle

from sklearn.base import BaseEstimator
from diamonds import logger
from diamonds.params import MODEL_REGISTRY 

def save_model(model, path):
    """Save the model to the specified path."""
    # Check the existence 
    if not os.path.exists(path) :
        os.mkdir(path)
        
    with open(os.path.join(path,"model.pkl"),"wb") as savefile:
        pickle.dump(model,savefile)
        

def load_model(path) -> BaseEstimator:
    """Load the model from the specified path."""
    # Implement the logic to load the model (e.g., using pickle, joblib, etc.)
    if not os.path.exists(os.path.join(path,"model.pkl")) :
        logger.error("No trained model to load")
        return None
        
    model = None
    with open(os.path.join(path,"model.pkl"),"rb") as datafile:
        model = pickle.load(datafile)
    
    return model

if __name__ == "__main__" :
    from sklearn.linear_model import LinearRegression
    
    linear_reg_obj = LinearRegression()
    
    save_model(linear_reg_obj,".")
    linear_reg_obj_loaded = load_model(".")
    
    print(f"{type(linear_reg_obj_loaded)}")
    
    if type(linear_reg_obj) == type(linear_reg_obj_loaded):
        print("Success")
    else:
        print("Fail")
    
    
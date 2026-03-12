from fastapi import FastAPI
from pydantic import BaseModel

from data_model import Diamant,DiamantResponse

import pandas as pd

from diamonds.data     import preprocess_data
from diamonds.registry import load_model

#Load the model (no need to load the model every time)
model = load_model("model")

rest_api = FastAPI(title="Diamonds REST API",
                   description="REST API to request price prediction from the trained model")


@rest_api.get("/")
def test_req():
    """
        Test API to check the REST API service is available
    """
    
    return {"Desc" : "Default page for the REST API"}

@rest_api.post("/price")
def price_req(diamant:Diamant,response_model=DiamantResponse):
    """
        Estimate the price of the diamond based on its characteristics
    """
    
    # Convert the received data into a DataFrame to be able to use them with the model
    dumped_data = diamant.model_dump()
    #print(f"Dumped data:\n{dumped_data}")
    
    try:
        #df_diamant = pd.DataFrame.from_dict(dumped_data,orient="index")
        df_diamant = pd.DataFrame([dumped_data])
    except Exception as e :
        print(e)
        return DiamantResponse(**dumped_data,price=0)
    
    
    # To be sure the format of the data fit the model waiting input
    df_clean_diamant = preprocess_data(df_diamant,False)
    #launch a prediction
    price_pred = model.predict(df_clean_diamant)
    
    return DiamantResponse(**dumped_data,price=price_pred[0])

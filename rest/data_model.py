from pydantic import BaseModel


class Diamant(BaseModel):
    """
    Class to serialize the data to be used with our model 
    """
    carat    : float
    cut      : str
    color    : str
    clarity  : str
    depth    : float 
    table    : float 
    x        : float 
    y        : float 
    z        : float
    
class DiamantResponse(Diamant):
    """
    Class to serialize the data to be used with our model 
    """
    price    : float = 0.0
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

with open('pipe.pkl', 'rb') as f:
    model = pickle.load(f)

class PriceItem(BaseModel):
    Location:str
    Year:int
    Kilometers_Driven:int
    Fuel_Type:str
    Transmission:str
    Owner_Type:str
    Seats:int
    Company:str
    Mileage_km_per_kg:float
    Engine_cc:float
    Power_bhp:float

@app.post('/')
async def price_endpoint(item:PriceItem):

    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    yhat = model.predict(df)
    print(yhat)
    return {'Predcition': np.round(float(yhat),2)}
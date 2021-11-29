# main.py

import sys
from fastapi import FastAPI
from pydantic import  BaseModel
from Task2 import whatjob
import uvicorn

# initiate API
app = FastAPI(title="What Job Type", description="X0PA TASk2 API", version="1.0")

def get_prediction(param1):
    
    whatjobClass=whatjob()
    y=whatjobClass.getPrediction(param1)
    return {'Job_Type': y[0]}


# define model for post request.
class ModelParams(BaseModel):
    param1: str


@app.post("/predict",tags=["predictions"])
def predict(params: ModelParams):

    pred = whatjob.getPrediction(params.param1)

    return pred

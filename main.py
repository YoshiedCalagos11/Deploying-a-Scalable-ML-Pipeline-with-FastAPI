import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import process_data
from ml.model import inference


app = FastAPI()


class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        populate_by_name = True


with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

@app.get("/")
async def root():
    return {"message": "Hello from the API!"}


@app.post("/predict")
async def predict(data: CensusData):
    df = pd.DataFrame([data.model_dump(by_alias=True)])
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
    )

    preds = inference(model, X)
    pred_value = int(preds[0])

    label = ">50K" if pred_value == 1 else "<=50K"
    return {"result": label}

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the model using joblib
heart_disease_model  = joblib.load("predict_heart_di.pkl")
diabetes_model = joblib.load("predict_diabetes.pkl")

class HeartDiseaseInput(BaseModel):
    Age: float
    Sex: float
    Cp: float
    Trestbps: float
    Chol: float
    Fbs: float
    Restecg: float
    Thalach: float
    Exang: float
    Oldpeak: float
    Slope: float
    Ca: float
    Thal: float

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get('/')
async def default_endpoint():
    return {"Connection stablished!"}

@app.post('/predict_heart_di')
async def predict_heart_di(item: HeartDiseaseInput):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = heart_disease_model.predict_heart_disease(df)

    return {"prediction": int(yhat[0])}

@app.post('/predict_diabetes')
async def predict_diabetes(item: DiabetesInput):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = diabetes_model.predict(df)

    return {"prediction": int(yhat[0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
 
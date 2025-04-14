# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("model.pkl")

# FastAPI instance
app = FastAPI(title="Iris Classifier API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your frontend domain if you want to restrict access
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods like GET, POST, etc.
    allow_headers=["*"],  # Allow all headers
)

# Define input schema
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define prediction endpoint
@app.post("/predict")
def predict(data: IrisData):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(features)[0]
    return {
        "prediction": int(prediction),
        "class": ["setosa", "versicolor", "virginica"][prediction]
    }
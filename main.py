# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load model
model = tf.keras.models.load_model("app/model.h5")

# Define input data structure
class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([data.features])  # Add batch dimension
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}

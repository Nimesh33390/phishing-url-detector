from fastapi import FastAPI
from tensorflow import keras
from Feature_Extractor import extract_features
from pydantic import BaseModel

app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    model = keras.models.load_model("Malicious_URL_Prediction.h5")  # Replace with the actual path to your model file

def predict_url(url, model):
    url_features = extract_features(url)
    prediction = model.predict([url_features])
    probability = prediction[0][0] * 100
    probability = round(probability, 3)
    return probability

class URLInput(BaseModel):
    url: str

@app.post("/predict")
def predict(input: URLInput):
    probability = predict_url(input.url, model)
    return {"probability": probability, "message": f"There is {probability}% chance the URL is malicious!"}
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Server is stable now"}

@app.post("/predict")
def predict(text: str):
    return {"prediction": "demo mode"}
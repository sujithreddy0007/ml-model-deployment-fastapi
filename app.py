from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = joblib.load("model.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, feature1: float = Form(...), feature2: float = Form(...)):
    input_data = np.array([[feature1, feature2]])
    prediction = model.predict(input_data)
    return templates.TemplateResponse("result.html", {"request": request, "prediction": prediction[0]})

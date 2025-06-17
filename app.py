from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import joblib
import os

app = FastAPI()

# Mount static directory (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Jinja2 templates from "templates" directory
templates = Jinja2Templates(directory="templates")

# Load the trained model
model = joblib.load("model.pkl")  # Make sure model.pkl is in the same directory


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, feature1: float = Form(...), feature2: float = Form(...)):
    data = np.array([[feature1, feature2]])
    prediction = model.predict(data)[0]
    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction
    })


# Only needed for local development
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # default 8000 locally
    uvicorn.run("app:app", host="0.0.0.0", port=port)

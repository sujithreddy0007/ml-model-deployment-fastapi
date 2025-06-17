from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle

app = FastAPI()

# Load model
model = pickle.load(open("classifier.pkl", "rb"))

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  variance: float = Form(...),
                  skewness: float = Form(...),
                  curtosis: float = Form(...),
                  entropy: float = Form(...)):
    prediction = model.predict([[variance, skewness, curtosis, entropy]])[0]
    result = "Fake Note" if prediction > 0.5 else "Bank Note"
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

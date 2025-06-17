import uvicorn
from fastapi import FastAPI  # ✅ Fixed typo: FastApi ➜ FastAPI

app = FastAPI()  # ✅ Fixed typo

@app.get("/")
def index():
    return {"message": "Hello from FastAPI updated"}

@app.get("/welcome")
def get_name(name: str):
    return {'welcome': f'{name}'}  # ✅ Fixed string formatting and key

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)  # ✅ Corrected IP (0.0 ➜ 1)

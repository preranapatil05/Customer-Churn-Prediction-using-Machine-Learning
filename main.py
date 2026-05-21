from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.api.routes import router

app = FastAPI()

app.include_router(router)

templates = Jinja2Templates(directory="src/templates")

app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
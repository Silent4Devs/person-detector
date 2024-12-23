import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
#from utils.camera import start_detection
from config.api import detection

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.title = os.getenv("APP_NAME")
app.version = os.getenv("APP_VERSION")

items = ["Item 1", "Item 2"]

@app.get("/", response_class=HTMLResponse)
def get_items(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "items": items})


@app.get("/items", response_class=HTMLResponse)
def get_items(request: Request):
    return templates.TemplateResponse("items.html", {"request": request, "items": items})

# if __name__ == "__main__":
#     start_detection()
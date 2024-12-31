import os
from fastapi import FastAPI, Request, Form
from threading import Thread
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from routes.detections import detection
from config.whichcamera import rtsp_url

from utils.camera import DetectionTask
import uvicorn

detection_task = DetectionTask(rtsp_url)

def run_detection():
    detection_task.start()

app = FastAPI()

def start_background_detection():
    thread = Thread(target=run_detection, daemon=True)  # Daemon para finalizar con la app
    thread.start()

start_background_detection()  # Llamar a la función para iniciar la detección al iniciar

templates = Jinja2Templates(directory="templates")

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.include_router(detection)

app.title = os.getenv("APP_NAME")
app.version = os.getenv("APP_VERSION")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
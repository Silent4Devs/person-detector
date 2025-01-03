import os
from fastapi import FastAPI, Request, Form
from threading import Thread
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from routes.detections import detection
from config.whichcamera import rtsp_url, ENDPOINT
from utils.camera import DetectionTask
import uvicorn
from config.database import get_db_connection
from config.database import create_detections_table
from fastapi.middleware.cors import CORSMiddleware

create_detections_table(get_db_connection())

detection_task = DetectionTask(rtsp_url)

def run_detection():
    detection_task.start()

app = FastAPI()

def start_background_detection():
    thread = Thread(target=run_detection, daemon=True)  # Daemon para finalizar con la app
    thread.start()

start_background_detection()  # Llamar a la función para iniciar la detección al iniciar

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="detections/images"), name="images")
templates = Jinja2Templates(directory="templates")

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection)

app.title = os.getenv("APP_NAME")
app.version = os.getenv("APP_VERSION")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
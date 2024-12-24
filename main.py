import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from utils.camera import start_detection
from config.api import detection


load_dotenv()

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.title = os.getenv("APP_NAME")
app.version = os.getenv("APP_VERSION")

app.include_router(detection)

@app.get('/', tags=["Home"])
def message():
    """
    Ruta principal que retorna mensaje de bienvenida
    """
    return {"Hello World!"}

if __name__ == "__main__":
    import sys

    # Condicional para decidir si ejecutar `start_detection()`
    if "start_detection" in sys.argv:
        from utils.camera import start_detection
        start_detection()
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
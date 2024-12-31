import os
from typing import List
from fastapi import FastAPI, HTTPException, APIRouter
from config.database import get_db_connection
from models.detections import Detection
from fastapi.responses import FileResponse

# Crear la aplicación FastAPI
app = FastAPI()
detection=APIRouter()

path=os.getenv("path")

@detection.get("/detections", response_model=List[Detection])
async def get_all_detections():
    """
    Devuelve todas las detecciones de la base de datos.
    """
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = "SELECT * FROM detections ORDER BY id DESC"
    cursor.execute(query)
    detections = cursor.fetchall()

    cursor.close()
    connection.close()

    if not detections:
        raise HTTPException(status_code=404, detail="No se encontraron detecciones.")

    return detections

@detection.get("/photos")
async def get_photos():
    """
    Devuelve todas las imágenes .jpg en la carpeta detections con un enlace para ver y descargar.
    """
    file_names = [f for f in os.listdir(path) if f.endswith('.jpg')]

    # Crear la respuesta con las URLs
    photos = [{"filename": f, "view_url": f"/photo/{f}"} for f in file_names]

    return photos

# Ruta para devolver una imagen específica y permitir descarga
@detection.get("/photo/{photo_name}")
async def get_photo(photo_name: str, download: bool = False):
    file_path = os.path.join(path, photo_name)

    if os.path.exists(file_path):
        # Si se pasa el parámetro `download=true`, se fuerza la descarga
        if download:
            return FileResponse(file_path, media_type="image/jpeg", filename=photo_name, headers={"Content-Disposition": f"attachment; filename={photo_name}"})

        # Si no se pasa el parámetro o es `download=false`, se muestra la imagen en el navegador
        return FileResponse(file_path, media_type="image/jpeg", filename=photo_name)

    return {"error": "File not found!"}

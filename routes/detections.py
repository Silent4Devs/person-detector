from typing import List
from fastapi import FastAPI, HTTPException, APIRouter
from datetime import datetime
from pydantic import BaseModel
from config.database import get_db_connection
from models.detections import Detection

# Crear la aplicación FastAPI
app = FastAPI()
detection=APIRouter()

@detection.get("/detections", response_model=List[Detection])
def get_all_detections():
    """
    Devuelve todas las detecciones de la base de datos.
    """
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = "SELECT * FROM detections"
    cursor.execute(query)
    detections = cursor.fetchall()

    cursor.close()
    connection.close()

    if not detections:
        raise HTTPException(status_code=404, detail="No se encontraron detecciones.")

    return detections

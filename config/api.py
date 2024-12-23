"""
Este módulo define la configuración y API principal para el proyecto.
"""

from typing import List
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from config.database import get_db_connection
# Modelo para representar una detección
class Detection(BaseModel):
    """
    Clase de ejemplo para gestionar la API.
    """
    id: int
    gender_detected: str
    datetime_detected: str
    photo: str
    photo_context: str
    #ia_analysis: str

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

# @detection.get("/detections/{detection_id}", response_model=Detection)
# def get_detection_by_id(detection_id: int):
#     """
#     Devuelve una detección específica por ID.
#     """
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)

#     query = "SELECT * FROM detections WHERE id = %s"
#     cursor.execute(query, (detection_id,))
#     detection = cursor.fetchone()

#     cursor.close()
#     connection.close()

#     if not detection:
#         raise HTTPException(status_code=404, detail="Detección no encontrada.")

#     return detection

# @app.post("/detections", response_model=Detection)
# def create_detection(detection: Detection):
#     """
#     Inserta una nueva detección en la base de datos.
#     """
#     connection = get_db_connection()
#     cursor = connection.cursor()

#     query = """
#         INSERT INTO detections (gender_detected, datetime_detected, photo, photo_context)
#         VALUES (%s, %s, %s, %s, %s)
#     """
#     values = (
#         detection.gender_detected,
#         detection.datetime_detected,
#         detection.photo,
#         detection.photo_context,
#         #detection.ia_analysis
#     )

#     cursor.execute(query, values)
#     connection.commit()
#     detection_id = cursor.lastrowid

#     cursor.close()
#     connection.close()

#     detection.id = detection_id
#     return detection

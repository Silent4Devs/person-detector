"""
Este módulo define la configuración y API principal para el proyecto.
"""

from typing import List
from fastapi import FastAPI, HTTPException, APIRouter
from datetime import datetime
from pydantic import BaseModel
from config.database import get_db_connection

class Detection(BaseModel):
    """
    Clase de ejemplo para gestionar la API.
    """
    id: int
    gender_detected: str
    datetime_detected: datetime
    photo: str
    photo_context: str
    #ia_analysis: str

class PhotoResponse(BaseModel):
    photo: str
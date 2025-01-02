import os
from dotenv import load_dotenv

load_dotenv()

ENDPOINT=os.getenv("ENDPOINT")

# Configuración de variables de entorno
if os.getenv("IsLap"):
    rtsp_url = os.getenv("rtsp_url")
else:
    rtsp_url = 0

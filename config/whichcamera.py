import os
from dotenv import load_dotenv

load_dotenv()

# Configuración de variables de entorno
if os.getenv("IsLap"):
    rtsp_url = 0
else:
    rtsp_url = os.getenv("rtsp_url")

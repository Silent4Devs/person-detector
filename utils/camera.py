import cv2
import sys
import os
from datetime import datetime, timedelta
from ultralytics import YOLO
from dotenv import load_dotenv
from utils.detections import analyze_person
from config.database import get_db_connection, insert_into_database
# from config.whichcamera import rtsp_url

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuración de variables de entorno
# rtsp_url = rtsp_url
rtsp_url = "rtsp://desarrollo:Password123.@192.168.6.31:554/Streaming/Channels/601"
model_name = os.getenv("MODEL_NAME")

# Crear carpeta para guardar capturas y archivo de registro
output_folder = path = os.getenv("path")
logs_folder = "detections/logs"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(logs_folder, exist_ok=True)

# Función para obtener el nombre del archivo de log basado en la fecha actual


def get_log_file_path():
    current_date = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(logs_folder, f"detection_{current_date}.log")


# Cargar modelo YOLO
yolo_model = YOLO(model_name)


class DetectionTask:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.running = False

    def calculate_iou(self, box1, box2):
        """Calcula el Intersection-over-Union (IoU) de dos cajas delimitadoras."""
        x1, y1, x2, y2 = box1
        px1, py1, px2, py2 = box2
        inter_x1 = max(x1, px1)
        inter_y1 = max(y1, py1)
        inter_x2 = min(x2, px2)
        inter_y2 = min(y2, py2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (px2 - px1) * (py2 - py1)
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def is_new_person(self, bbox, current_time, detected_persons, threshold_seconds=70, iou_threshold=0.7):
        """Determina si una persona detectada es nueva o ya fue registrada."""
        x1, y1, x2, y2 = bbox
        for (px1, py1, px2, py2), last_time in list(detected_persons.items()):
            if self.calculate_iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.5:
                if (current_time - last_time).total_seconds() > threshold_seconds:
                    detected_persons[(px1, py1, px2, py2)] = current_time
                    return True
                return False
        detected_persons[(x1, y1, x2, y2)] = current_time
        return True

    def start(self):
        self.running = True
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            error = f"Error: No se pudo abrir el flujo RTSP o la cámara en {rtsp_url}."
            print(error)
            return

        detected_persons = {}
        last_capture_time = datetime.min  # Inicializa con una fecha muy antigua
        capture_interval = timedelta(seconds=30)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame del flujo RTSP.")
                break

            current_time = datetime.now()

            # Revisa si ha pasado suficiente tiempo desde la última captura
            if (current_time - last_capture_time) < capture_interval:
                continue

            results = yolo_model(frame, conf=0.5, verbose=False)
            persons = [
                box for box in results[0].boxes.data if int(box[-1]) == 0]

            for person in persons:
                x1, y1, x2, y2, conf, cls = map(int, person.tolist())
                current_time = datetime.now()

                if self.is_new_person((x1, y1, x2, y2), current_time, detected_persons, threshold_seconds=70, iou_threshold=0.7):
                    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                    imagen = f"person_{timestamp}.jpg"
                    full_image_path = os.path.join(output_folder, imagen)
                    # Save image with compression (lower quality for smaller file size)
                    # 60 is the compression quality
                    cv2.imwrite(full_image_path, frame, [
                                cv2.IMWRITE_JPEG_QUALITY, 60])

                    gender, description = analyze_person(full_image_path)

                    # Generate the log file path for the current day
                    log_file = get_log_file_path()

                    log_entry = f"[{timestamp}] Nueva persona detectada.\n  Género: {gender}.\n  Descripción: {description}\n"
                    with open(log_file, "a") as log:
                        log.write(log_entry + "\n")

                    # Insert the detection event into the database
                    try:
                        conn = get_db_connection()
                        insert_into_database(
                            conn, gender, timestamp, description, imagen)
                    except Exception as e:
                        print(f"Error al insertar en la base de datos: {e}")
                    finally:
                        if conn and conn.is_connected():
                            conn.close()

        cap.release()

    def stop(self):
        self.running = False

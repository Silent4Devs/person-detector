import cv2
import sys
import os
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.detections import analyze_person, device
from transformers import BlipForConditionalGeneration

load_dotenv()

# Configuración de variables de entorno
rtsp_url = os.getenv("rtsp_url")

# Crear carpeta para guardar capturas y archivo de registro
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)
log_file = os.path.join(output_folder, "detections.log")

# Cargar modelo YOLO
yolo_model = YOLO("yolov8s.pt")

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
        #pass

    def is_new_person(self, bbox, current_time, detected_persons, threshold_seconds=70, iou_threshold=0.7):
        """Determina si una persona detectada es nueva o ya fue registrada."""
        #global detected_persons
        x1, y1, x2, y2 = bbox
        for (px1, py1, px2, py2), last_time in list(detected_persons.items()):
            if self.calculate_iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.5:
                if (current_time - last_time).total_seconds() > threshold_seconds:
                    detected_persons[(px1, py1, px2, py2)] = current_time
                    return True
                return False
        detected_persons[(x1, y1, x2, y2)] = current_time
        return True
        #pass

    def start(self):
        self.running = True
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print("Error: No se pudo abrir el flujo RTSP.")
            return

        detected_persons = {}

        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame del flujo RTSP.")
                break

            results = yolo_model(frame, conf=0.5, verbose=False)
            persons = [box for box in results[0].boxes.data if int(box[-1]) == 0]

            for person in persons:
                x1, y1, x2, y2, conf, cls = map(int, person.tolist())
                current_time = datetime.now()

                if self.is_new_person((x1, y1, x2, y2), current_time, detected_persons, threshold_seconds=70, iou_threshold=0.7):
                    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                    full_image_path = os.path.join(output_folder, f"person_{timestamp}.jpg")
                    cv2.imwrite(full_image_path, frame)

                    gender, description = analyze_person(full_image_path)

                    log_entry = f"[{timestamp}] Nueva persona detectada.\n  Género: {gender}.\n  Descripción: {description}\n"
                    with open(log_file, "a") as log:
                        log.write(log_entry + "\n")

        cap.release()

    def stop(self):
        self.running = False


#
# detected_persons = {}
#
# def calculate_iou(box1, box2):
#     """Calcula el Intersection-over-Union (IoU) de dos cajas delimitadoras."""
#     x1, y1, x2, y2 = box1
#     px1, py1, px2, py2 = box2
#     inter_x1 = max(x1, px1)
#     inter_y1 = max(y1, py1)
#     inter_x2 = min(x2, px2)
#     inter_y2 = min(y2, py2)
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (px2 - px1) * (py2 - py1)
#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou
#
# def is_new_person(bbox, current_time, threshold_seconds=5):
#     """Determina si una persona detectada es nueva o ya fue registrada."""
#     global detected_persons
#     x1, y1, x2, y2 = bbox
#     for (px1, py1, px2, py2), last_time in list(detected_persons.items()):
#         if calculate_iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.5:
#             if (current_time - last_time).total_seconds() > threshold_seconds:
#                 detected_persons[(px1, py1, px2, py2)] = current_time
#                 return True
#             return False
#     detected_persons[(x1, y1, x2, y2)] = current_time
#     return True
#
# # Inicializar captura de video
# cap = cv2.VideoCapture(rtsp_url)
# if not cap.isOpened():
#     print("Error: No se pudo abrir el flujo RTSP.")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: No se pudo leer el frame del flujo RTSP.")
#         break
#
#     # Realizar detecciones con YOLO
#     results = yolo_model(frame, conf=0.5, verbose=False)
#     persons = [box for box in results[0].boxes.data if int(box[-1]) == 0]
#
#     for person in persons:
#         x1, y1, x2, y2, conf, cls = map(int, person.tolist())
#         current_time = datetime.now()
#
#         if is_new_person((x1, y1, x2, y2), current_time):
#             # Guardar el frame completo
#             timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
#             full_image_path = os.path.join(output_folder, f"person_{timestamp}.jpg")
#             cv2.imwrite(full_image_path, frame)
#
#             # Analizar la persona detectada
#             gender, description = analyze_person(full_image_path)
#
#             # Registrar en el log
#             log_entry = f"[{timestamp}] Nueva persona detectada.\n  Género: {gender}.\n  Descripción: {description}\n"
#             with open(log_file, "a") as log:
#                 log.write(log_entry + "\n")
#
#     # Detener el proceso después de un tiempo para evitar ejecución infinita
#     # Descomenta esta línea si deseas limitar la ejecución: break
#
# # Liberar recursos
# cap.release()
# print("Proceso finalizado. Las capturas se han guardado en la carpeta detections.")

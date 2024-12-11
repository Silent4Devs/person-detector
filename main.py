import cv2
import os
import torch
import requests
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv
from PIL import Image

# Cargar variables de entorno
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Configurar BLIP (Image-to-Text)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Función para generar descripciones de imágenes
def generate_image_description(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Configurar YOLOv8n
yolo_model = YOLO("yolov8n.pt")  # Modelo preentrenado
PERSON_CLASS = 0
PACKAGE_CLASSES = [22, 23]  # IDs de las clases de paquetes (por ejemplo, cajas y sobres)

# Configurar DeepFace para análisis facial
def analyze_face(image):
    """Analiza género de un rostro usando DeepFace."""
    try:
        analysis = DeepFace.analyze(img_path=image, actions=['gender'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        gender_scores = analysis.get("gender", {})
        return "Femenina" if gender_scores.get("Woman", 0) > gender_scores.get("Man", 0) else "Masculino"
    except Exception:
        return "Unknown"

# Detección de personas previas
detected_persons = {}

def calculate_iou(box1, box2):
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

def is_new_person(bbox, current_time, threshold_seconds=5):
    """Determina si una persona detectada es nueva."""
    global detected_persons
    x1, y1, x2, y2 = bbox
    for (px1, py1, px2, py2), last_time in list(detected_persons.items()):
        if calculate_iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.5:
            if (current_time - last_time).total_seconds() > threshold_seconds:
                detected_persons[(px1, py1, px2, py2)] = current_time
                return True
            else:
                return False
    detected_persons[(x1, y1, x2, y2)] = current_time
    return True

# Configuración de salida
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)
log_file = os.path.join(output_folder, "detections.log")

# Abrir RTSP cámara
rtsp_url = "rtsp://desarrollo:Password123.@192.168.6.31:554/Streaming/Channels/202"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar objetos usando YOLO
    results = yolo_model(frame, conf=0.5, verbose=False)
    boxes = results[0].boxes

    persons = []
    packages = []

    for box in boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if cls == PERSON_CLASS:
            persons.append((x1, y1, x2, y2, conf))
        elif cls in PACKAGE_CLASSES:
            packages.append((x1, y1, x2, y2, conf, cls))

    # Procesar detección de personas
    for person in persons:
        x1, y1, x2, y2, conf = person
        current_time = datetime.now()

        if is_new_person((x1, y1, x2, y2), current_time):
            # Guardar rostro detectado
            face_crop = frame[y1:y2, x1:x2]
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            face_path = os.path.join(output_folder, f"person_{timestamp}.jpg")
            cv2.imwrite(face_path, face_crop)

            # Generar descripción e información adicional
            image_description = generate_image_description(face_path)
            gender = analyze_face(face_path)

            # Registrar evento
            with open(log_file, "a") as log:
                log.write(f"[{timestamp}] New Person Detected. Gender: {gender}. Description: {image_description}\n")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Procesar detección de paquetes
    for package in packages:
        x1, y1, x2, y2, conf, cls = package
        label = "Box" if cls == 22 else "Envelope"

        # Dibujar rectángulo y etiqueta
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Registrar evento
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as log:
            log.write(f"[{timestamp}] {label} Detected. Confidence: {conf:.2f}\n")

    # Mostrar frame
    cv2.imshow("Person & Package Detection", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

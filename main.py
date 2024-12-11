import cv2
import os
import torch
from datetime import datetime, timedelta
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv
from PIL import Image

# Cargar variables de entorno
load_dotenv()

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

# Detección de rostros previos con tiempo de espera
detected_faces = {}

def should_process_person(face_id, current_time, wait_time=60):
    """Determina si debe procesarse una persona según el tiempo de espera."""
    last_time = detected_faces.get(face_id)
    if last_time is None or (current_time - last_time).total_seconds() > wait_time:
        detected_faces[face_id] = current_time
        return True
    return False

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

    for box in boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if cls == PERSON_CLASS:
            persons.append((x1, y1, x2, y2, conf))

    # Procesar detección de personas
    for person in persons:
        x1, y1, x2, y2, conf = person
        current_time = datetime.now()
        face_id = (x1, y1, x2, y2)  # Identificador basado en la caja delimitadora

        if should_process_person(face_id, current_time):
            # Guardar imagen completa
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            image_path = os.path.join(output_folder, f"frame_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)

            # Generar descripción e información adicional
            image_description = generate_image_description(image_path)
            gender = analyze_face(image_path)

            # Registrar evento
            with open(log_file, "a") as log:
                log.write(f"[{timestamp}] New Person Detected. Gender: {gender}. Description: {image_description}\n")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar frame
    cv2.imshow("Person Detection", frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

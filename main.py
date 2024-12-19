import cv2
import os
import torch
#import requests
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")  # URL de tu servidor IA
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")  # Modelo IA en tu servidor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_image_description(image_path):
    """Genera una descripción de la imagen usando BLIP."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# def analyze_with_server(image_path):
#     """Envía la imagen al servidor IA para su análisis."""
#     try:
#         with open(image_path, 'rb') as image_file:
#             files = {'file': image_file}
#             data = {'model': OLLAMA_MODEL}
#             response = requests.post(OLLAMA_URL, files=files, data=data, timeout=10)
#             if response.status_code == 200:
#                 return response.json().get("result", "Unknown Analysis")
#             else:
#                 return f"Llama Server Error: {response.status_code}"
#     except Exception as e:
#         return f"Error: {str(e)}"

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

output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)
log_file = os.path.join(output_folder, "detections.log")

yolo_model = YOLO("yolov8n.pt")
detected_persons = {}

def is_new_person(bbox, current_time, threshold_seconds=5):
    """Determina si una persona detectada es nueva o ya fue registrada."""
    global detected_persons
    x1, y1, x2, y2 = bbox
    for (px1, py1, px2, py2), last_time in list(detected_persons.items()):
        if calculate_iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.5:
            if (current_time - last_time).total_seconds() > threshold_seconds:
                detected_persons[(px1, py1, px2, py2)] = current_time
                return True
            return False
    detected_persons[(x1, y1, x2, y2)] = current_time
    return True

def analyze_person(image_path):
    """Analiza la persona detectada usando DeepFace, BLIP y el servidor IA."""
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['gender'], enforce_detection=False)
        gender_scores = analysis[0].get("gender", {}) if isinstance(analysis, list) else analysis.get("gender", {})
        gender = "Feminine" if gender_scores.get("Woman", 0) > gender_scores.get("Man", 0) else "Masculine"
    except Exception:
        gender = "Unknown"

    description = generate_image_description(image_path)

#    server_response = analyze_with_server(image_path)

    return gender, description,

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame, conf=0.5, verbose=False)
    persons = [box for box in results[0].boxes.data if int(box[-1]) == 0]

    for person in persons:
        x1, y1, x2, y2, conf, cls = map(int, person.tolist())
        current_time = datetime.now()

        if is_new_person((x1, y1, x2, y2), current_time):
            face_crop = frame[y1:y2, x1:x2]
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            face_path = os.path.join(output_folder, f"person_{timestamp}.jpg")
            cv2.imwrite(face_path, face_crop)

            gender, description, = analyze_person(face_path)

            log_entry = f"[{timestamp}] New Person Detected. \n  Gender: {gender}. \n Description: {description}"
            with open(log_file, "a") as log:
                log.write(log_entry + "\n")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{gender}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

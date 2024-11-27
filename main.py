import cv2
import os
import torch
from datetime import datetime, timedelta
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# Load BLIP (Image-to-Text) model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def generate_image_description(image_path):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

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

yolo_model = YOLO("yolov8n.pt")  # Ajustar según el modelo deseado (n, s, m, etc.)

detected_persons = {}

def is_new_person(bbox, current_time, threshold_seconds=5):
    """Determina si una persona detectada es nueva o ya fue registrada."""
    global detected_persons
    x1, y1, x2, y2 = bbox

    for (px1, py1, px2, py2), last_time in list(detected_persons.items()):
        iou = calculate_iou((x1, y1, x2, y2), (px1, py1, px2, py2))
        if iou > 0.5:  # Considera misma persona si IoU > 0.5
            if (current_time - last_time).total_seconds() > threshold_seconds:
                detected_persons[(px1, py1, px2, py2)] = current_time
                return True
            else:
                return False

    detected_persons[(x1, y1, x2, y2)] = current_time
    return True

def analyze_face(image):
    """Realiza un análisis facial usando DeepFace y devuelve el género predominante."""
    try:
        analysis = DeepFace.analyze(img_path=image, actions=['gender'], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = analysis[0]
        
        gender_scores = analysis.get("gender", {})
        if not gender_scores:
            return "Unknown"
        
        # Seleccionar el género predominante
        return "Feminine" if gender_scores.get("Woman", 0) > gender_scores.get("Man", 0) else "Masculine"
    except Exception as e:
        return "Unknown"

# Abrir cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar personas usando YOLO
    results = yolo_model(frame, conf=0.5, verbose=False)
    persons = [box for box in results[0].boxes.data if int(box[-1]) == 0]  # Clase 0 corresponde a 'person'

    for person in persons:
        x1, y1, x2, y2, conf, cls = map(int, person.tolist())
        current_time = datetime.now()

        # Verifica si es una nueva persona
        if is_new_person((x1, y1, x2, y2), current_time):
            # Recortar la región del rostro
            face_crop = frame[y1:y2, x1:x2]
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            face_path = os.path.join(output_folder, f"person_{timestamp}.jpg")
            cv2.imwrite(face_path, face_crop)

            # Analizar género de la persona
            gender = analyze_face(face_path)
        else:
            # Para una persona ya registrada, usa el género de la última vez
            gender = gender

        # Dibujar rectángulo y género en el cuadro
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{gender}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostrar el feed de video
    cv2.imshow('Person Detection', frame)

    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# import cv2
# import os
# from datetime import datetime, timedelta
# import requests
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch
# from ultralytics import YOLO
# from dotenv import load_dotenv

# load_dotenv()

# OLLAMA_URL = os.getenv("OLLAMA_URL")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

# # Load BLIP (Image-to-Text) model and processor
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


# def generate_image_description(image_path):
#     from PIL import Image
#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(image, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs)
#     caption = processor.decode(outputs[0], skip_special_tokens=True)
#     return caption

# # Function to communicate with LLaMA via Ollama API
# def analyze_with_llama(text):
#     try:
#         payload = {
#             "model": OLLAMA_MODEL,  # Specify the model name
#             "input": f"Analyze this description: {text}"
#         }
#         response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
#         if response.status_code == 200:
#             return response.json().get('response', 'No response')
#         else:
#             return f"Ollama Error: {response.status_code}"
#     except Exception as e:
#         return f"Error communicating with Ollama: {str(e)}"

# # Create an output folder for detections
# output_folder = "detections"
# os.makedirs(output_folder, exist_ok=True)

# # Log file for detections
# log_file = os.path.join(output_folder, "detections.log")

# # Load YOLOv8 model
# yolo_model = YOLO("yolov8n.pt")  # Replace with 'yolov8s.pt', 'yolov8m.pt', etc., for better accuracy

# # Tracking detected persons
# detected_persons = {}  # Store unique persons with last detected time

# def is_new_person(bbox, current_time, threshold_seconds=300):
#     """Check if the detected person is new or already recorded within the threshold time."""
#     global detected_persons
#     x1, y1, x2, y2 = bbox

#     # Check if a similar bounding box exists in the record
#     for (px1, py1, px2, py2), last_time in detected_persons.items():
#         # Use an intersection-over-union (IoU) threshold to determine if it’s the same person
#         iou = calculate_iou((x1, y1, x2, y2), (px1, py1, px2, py2))
#         if iou > 0.5:  # Consider it the same person
#             if (current_time - last_time).seconds > threshold_seconds:
#                 # Update the timestamp for this person
#                 detected_persons[(px1, py1, px2, py2)] = current_time
#                 return True  # Same person but beyond threshold time
#             else:
#                 return False  # Same person within threshold time

#     # If no matching person is found, add a new entry
#     detected_persons[(x1, y1, x2, y2)] = current_time
#     return True

# def calculate_iou(box1, box2):
#     """Calculate the Intersection-over-Union (IoU) of two bounding boxes."""
#     x1, y1, x2, y2 = box1
#     px1, py1, px2, py2 = box2

#     # Determine the coordinates of the intersection rectangle
#     inter_x1 = max(x1, px1)
#     inter_y1 = max(y1, py1)
#     inter_x2 = min(x2, px2)
#     inter_y2 = min(y2, py2)

#     # Compute the area of intersection
#     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

#     # Compute the area of both boxes
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (px2 - px1) * (py2 - py1)

#     # Compute the IoU
#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou

# # Open camera
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Perform person detection using YOLOv8
#     results = yolo_model(frame, conf=0.5)
#     persons = [box for box in results[0].boxes.data if int(box[-1]) == 0]  # Class 0 corresponds to 'person'

#     for person in persons:
#         x1, y1, x2, y2, conf, cls = person.tolist()
#         current_time = datetime.now()

#         # Check if the person is new
#         if is_new_person((int(x1), int(y1), int(x2), int(y2)), current_time):
#             # Save image
#             timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
#             image_path = os.path.join(output_folder, f"person_{timestamp}.jpg")
#             cv2.imwrite(image_path, frame)

#             # Generate image description
#             image_description = generate_image_description(image_path)

#             # Analyze description with LLaMA
#             llama_analysis = analyze_with_llama(image_description)

#             # Log event
#             with open(log_file, "a") as log:
#                 log.write(f"Person detected at {timestamp}.\n")
#                 log.write(f"Image Description: {image_description}\n")
#                 log.write(f"LLaMA Analysis: {llama_analysis}\n")

#         # Draw rectangles around detected persons
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#         cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Display the video feed
#     cv2.imshow('Person Detection', frame)

#     # Break loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

import cv2
import sys
import os
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace
from dotenv import load_dotenv
from models.detections import analyze_person, device
from transformers import BlipForConditionalGeneration

load_dotenv()

# OLLAMA_URL = os.getenv("OLLAMA_URL")  # URL de tu servidor IA
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")  # Modelo IA en tu servidor

rtsp_url=os.getenv("rtsp_url")

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

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

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

            log_entry = f"[{timestamp}] New Person Detected. \n  Gender: {gender}.\n Description: {description}"
            with open(log_file, "a") as log:
                log.write(log_entry + "\n")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{gender}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    cv2.imshow('Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        breakmodel = [BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
            ).to(device)]


cap.release()
cv2.destroyAllWindows()

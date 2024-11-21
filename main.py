import cv2
import os
from datetime import datetime
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from ultralytics import YOLO  # YOLOv8 import

# Load BLIP (Image-to-Text) model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Function to generate image descriptions using BLIP
def generate_image_description(image_path):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function to communicate with LLaMA via Ollama API
def analyze_with_llama(text):
    try:
        ollama_url = "http://localhost:11434"
        payload = {
            "model": "llama-3.1-8b",  # Specify the model name
            "input": f"Analyze this description: {text}"
        }
        response = requests.post(f"{ollama_url}/api/generate", json=payload)
        if response.status_code == 200:
            return response.json().get('response', 'No response')
        else:
            return f"Ollama Error: {response.status_code}"
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

# Create an output folder for detections
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)

# Log file for detections
log_file = os.path.join(output_folder, "detections.log")

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Replace with 'yolov8s.pt', 'yolov8m.pt', etc., for better accuracy

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform person detection using YOLOv8
    results = yolo_model(frame, conf=0.5)
    persons = [box for box in results[0].boxes.data if int(box[-1]) == 0]  # Class 0 corresponds to 'person'

    if len(persons) > 0:
        # Save image
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        image_path = os.path.join(output_folder, f"person_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)

        # Generate image description
        image_description = generate_image_description(image_path)

        # Analyze description with LLaMA
        llama_analysis = analyze_with_llama(image_description)

        # Log event
        with open(log_file, "a") as log:
            log.write(f"Person(s) detected at {timestamp}: {len(persons)} person(s) found.\n")
            log.write(f"Image Description: {image_description}\n")
            log.write(f"LLaMA Analysis: {llama_analysis}\n")

    # Draw rectangles around detected persons
    for person in persons:
        x1, y1, x2, y2, conf, cls = person.tolist()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow('Person Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

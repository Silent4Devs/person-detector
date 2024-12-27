import torch
#import requests
from deepface import DeepFace
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

def generate_image_description(image_path):
    """Genera una descripción de la imagen usando BLIP."""
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def analyze_person(image_path):
    """
    Analiza la persona detectada usando DeepFace, BLIP
    y el servidor IA.
    """
    try:
        analysis = DeepFace.analyze(
            img_path=image_path, actions=['gender'], enforce_detection=False
            )
        gender_scores = analysis[0].get("gender", {}) if isinstance(
            analysis, list
            ) else analysis.get("gender", {})
        gender = "Feminine" if gender_scores.get(
            "Woman", 0
            ) > gender_scores.get(
                "Man", 0
                ) else "Masculine"
    except Exception:
        gender = "Unknown"

    description = generate_image_description(image_path)
#    server_response = analyze_with_server(image_path)
    return gender, description

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
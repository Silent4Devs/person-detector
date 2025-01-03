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
rtsp_url = os.getenv("rtsp_url")
#rtsp_url = "rtsp://desarrollo:Password123.@192.168.6.31:554/Streaming/Channels/602"
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
        self.max_retries = 30
        self.retry_delay = 10  # seconds

    def verify_stream_format(self):
        """Verify stream format using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                self.rtsp_url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            codec = result.stdout.strip()
            print(f"Detected codec: {codec}")
            return codec
        except Exception as e:
            print(f"Error checking stream format: {e}")
            return None

    def connect_to_stream(self):
        # Set OpenCV backend options for HEVC
        os.environ[
            "OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp,tcp,rtsp|rtsp_transport;tcp|fflags;nobuffer|max_delay;0"

        for attempt in range(self.max_retries):
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                print(f"Connection attempt {attempt + 1} failed, retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
                continue

            # Configure stream parameters
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H265'))

            # Verify connection with test frame read
            for _ in range(5):  # Multiple read attempts
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Successfully connected to RTSP stream on attempt {attempt + 1}")
                    return cap

            cap.release()

        return None

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

        # Verify stream codec
        codec = self.verify_stream_format()
        if codec and codec.lower() == 'hevc':
            print("HEVC stream detected, using appropriate configuration")

        cap = self.connect_to_stream()
        if cap is None:
            print(f"Error: Failed to connect to RTSP stream at {self.rtsp_url} after {self.max_retries} attempts.")
            return

        detected_persons = {}
        last_capture_time = datetime.min
        capture_interval = timedelta(seconds=30)
        consecutive_failures = 0
        max_consecutive_failures = 30

        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    print(f"Failed to read frame. Attempt {consecutive_failures} of {max_consecutive_failures}")

                    if consecutive_failures >= max_consecutive_failures:
                        print("Attempting to reconnect to stream...")
                        cap.release()
                        cap = self.connect_to_stream()
                        if cap is None:
                            print("Reconnection failed. Stopping detection task.")
                            break
                        consecutive_failures = 0
                    continue

                consecutive_failures = 0  # Reset counter on successful frame read
                current_time = datetime.now()

                if (current_time - last_capture_time) < capture_interval:
                    continue

                # Rest of your existing detection code...
                results = yolo_model(frame, conf=0.5, verbose=False)
                persons = [box for box in results[0].boxes.data if int(box[-1]) == 0]

                for person in persons:
                    # Your existing person detection code...
                    pass

            except Exception as e:
                print(f"Error during frame processing: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("Too many consecutive errors. Attempting to reconnect...")
                    cap.release()
                    cap = self.connect_to_stream()
                    if cap is None:
                        print("Reconnection failed. Stopping detection task.")
                        break
                    consecutive_failures = 0

        if cap is not None:
            cap.release()

    def stop(self):
        self.running = False

"""

Este script lee información del archivo detections.log y las imágenes asociadas
en la carpeta detections para insertarlas en una base de datos MySQL.
"""

import os
from dotenv import load_dotenv
import mysql.connector

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Recuperar las credenciales de conexión
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

def connect_to_db():
    """
    Conecta a la base de datos MySQL usando las credenciales del archivo .env.
    """
    try:
        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        if conn.is_connected():
            print("Conexión exitosa a la base de datos")
        return conn
    except mysql.connector.Error as err:
        print(f"Error de conexión a la base de datos: {err}")
        raise

def parse_log_file(log_file):
    """
    Parsea el archivo detections.log y extrae los datos de detección.
    Maneja registros distribuidos en múltiples líneas.
    """
    detections = []
    current_detection = {}

    with open(log_file, "r") as file:
        for line in file:
            line = line.strip()

            if line.startswith("["):
                if current_detection:
                    detections.append(current_detection)
                current_detection = {"timestamp": line[1:].split("]")[0]}

            elif line.startswith("Gender: "):
                current_detection["gender"] = line.split("Gender: ")[1]

            elif line.startswith("Description: "):
                current_detection["description"] = line.split("Description: ")[1]

        # Agregar la última detección si existe
        if current_detection:
            detections.append(current_detection)

    # Formatear detecciones en la salida esperada
    formatted_detections = [
        (
            det.get("timestamp", "Unknown"),
            det.get("gender", "Unknown"),
            det.get("description", "No description"))
        for det in detections
    ]

    return formatted_detections

def insert_into_database(connection, detections, image_folder):
    """
    Inserta las detecciones y sus imágenes asociadas en la base de datos.
    """
    cursor = connection.cursor()
    insert_query = """
        INSERT INTO detections (gender_detected, datetime_detected, photo, photo_context)
        VALUES (%s, %s, %s, %s)
    """
    for timestamp, gender, description in detections:
        # Crear el nombre de archivo para la imagen
        image_name = f"person_{timestamp.replace(':', '-').replace(' ', '_')}.jpg"
        image_path = os.path.join(image_folder, image_name)

        # Verifica si la imagen existe antes de insertar
        if os.path.exists(image_path):
            try:
                # Valores para insertar
                values = (
                    gender,             # gender_detected
                    timestamp,          # datetime_detected
                    image_path,         # photo
                    description         # photo_context
                )
                cursor.execute(insert_query, values)
            except mysql.connector.Error as err:
                print(f"Error al insertar datos: {err}")
        else:
            print(f"Imagen no encontrada para {timestamp}: {image_path}")

    connection.commit()
    cursor.close()

if __name__ == "__main__":
    log_file = "detections/detections.log"
    image_folder = "detections"

    # Leer y parsear el archivo detections.log
    detections = parse_log_file(log_file)

    # Conectar a la base de datos
    conn = None
    try:
        conn = connect_to_db()
        insert_into_database(conn, detections, image_folder)
        print("Datos insertados correctamente en la base de datos.")
    except Exception as e:
        print(f"Error en el script: {e}")
    finally:
        # Cerrar la conexión a la base de datos
        if conn and conn.is_connected():
            conn.close()

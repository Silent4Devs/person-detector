"""

Este script lee información del archivo detections.log y las imágenes asociadas
en la carpeta detections para insertarlas en una base de datos MySQL.
"""

import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

# Recuperar las credenciales de conexión
db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")

def get_db_connection():
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

def create_detections_table(connection):
    """
    Crea la tabla 'detections' en la base de datos si no existe.
    """
    cursor = connection.cursor()
    create_table_query = """
    CREATE TABLE IF NOT EXISTS detections (
        id INT AUTO_INCREMENT PRIMARY KEY,
        gender_detected VARCHAR(50),
        datetime_detected DATETIME,
        photo VARCHAR(255),
        photo_context TEXT
    );
    """
    try:
        cursor.execute(create_table_query)
        connection.commit()
        print("Tabla 'detections' creada o ya existe.")
    except mysql.connector.Error as err:
        print(f"Error al crear la tabla: {err}")
    finally:
        cursor.close()

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

def insert_into_database(connection, gender, timestamp, description, image_path):
    """
    Inserta los detalles de la detección en la base de datos.
    """
    cursor = connection.cursor()
    insert_query = """
        INSERT INTO detections (gender_detected, datetime_detected, photo, photo_context)
        VALUES (%s, %s, %s, %s)
    """
    try:
        values = (gender, timestamp, image_path, description)
        cursor.execute(insert_query, values)
        connection.commit()
    except mysql.connector.Error as err:
        print(f"Error al insertar datos: {err}")
    finally:
        cursor.close()

if __name__ == "__main__":
    log_file = "detections/detections.log"
    image_folder = "detections"

    # Leer y parsear el archivo detections.log
    detections = parse_log_file(log_file)

    # Conectar a la base de datos
    conn = None
    try:
        conn = get_db_connection()
        insert_into_database(conn, detections, image_folder)
        print("Datos insertados correctamente en la base de datos.")
    except Exception as e:
        print(f"Error en el script: {e}")
    finally:
        # Cerrar la conexión a la base de datos
        if conn and conn.is_connected():
            conn.close()

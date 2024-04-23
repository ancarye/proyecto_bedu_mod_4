import cv2
import os

# Cargar el modelo preentrenado para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para realizar el reconocimiento facial en una imagen
def reconocimiento_facial(image_path):
    # Leer la imagen
    image = cv2.imread(image_path)
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(25, 25))
    # Dibujar un rectángulo alrededor de cada rostro detectado
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Guardar la imagen con los rostros detectados en la carpeta imagenes
    output_path = os.path.join('imagenes', os.path.basename(image_path).split('.')[0] + '_detectado.jpg')
    cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # Guardar como JPG con calidad del 90%

# Llamar a la función para realizar el reconocimiento facial
image_path = input("Por favor, escribe la ubicación de tu archivo de imagen: ")
reconocimiento_facial(image_path)
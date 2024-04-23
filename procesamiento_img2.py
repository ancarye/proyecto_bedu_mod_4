import cv2
import os

# Cargar el modelo preentrenado para detección de rostros utilizando dnn (Deep Neural Networks)
face_detector = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Función para realizar el reconocimiento facial en una imagen
def reconocimiento_facial(image_path):
    # Leer la imagen
    image = cv2.imread(image_path)

    # Obtener dimensiones de la imagen
    (h, w) = image.shape[:2]

    # Preprocesar la imagen para la detección de rostros
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pasar el blob a través de la red neuronal para la detección de rostros
    face_detector.setInput(blob)
    detections = face_detector.forward()

    # Dibujar un rectángulo alrededor de cada rostro detectado
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Umbral de confianza ajustable
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)

    # Guardar la imagen con los rostros detectados en la carpeta imagenes
    output_path = os.path.join('imagenes', os.path.basename(image_path).split('.')[0] + '_detectado.jpg')
    cv2.imwrite(output_path, image)

# Llamar a la función para realizar el reconocimiento facial
image_path = input("Por favor, escribe la ubicación de tu archivo de imagen: ")
reconocimiento_facial(image_path)

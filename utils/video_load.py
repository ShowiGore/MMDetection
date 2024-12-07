import cv2
from mmdet.apis import init_detector, inference_detector

# Specify the configuration file for the model.
config_file = 'fine_tuning/fine_tuning.py'

# Specify the pre-trained model checkpoint file.
checkpoint_file = 'fine_tuning/epoch_50.pth'

# Inicializa el modelo
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # Usa 'cpu' si no tienes GPU

# Obtén los nombres de las clases desde la metadata del modelo
class_names = model.dataset_meta['classes']

# Ruta del video de entrada y salida
input_video_path = 'data/video/balonmano.mp4'
output_video_path = 'fine_tuning/output.mp4'

# Captura del video de entrada
cap = cv2.VideoCapture(input_video_path)

# Propiedades del video de entrada
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el video de salida

# Inicializa el video de salida
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define el umbral de confianza para mostrar detecciones
score_thr = 0.46

# Procesa el video frame por frame
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Se alcanzó el final del video o no se pudo leer el frame.")
            break

        # Realiza la inferencia en el frame actual
        result = inference_detector(model, frame)

        # Extrae las detecciones del resultado
        detections = result.pred_instances  # Accede a las predicciones de instancias
        bboxes = detections.bboxes.cpu().numpy()  # Cajas delimitadoras
        scores = detections.scores.cpu().numpy()  # Puntuaciones
        labels = detections.labels.cpu().numpy()  # Índices de clases

        # Visualiza las detecciones en el frame
        for bbox, score, label in zip(bboxes, scores, labels):
            if score >= score_thr:
                x1, y1, x2, y2 = bbox
                class_name = class_names[label]
                # Dibuja la caja delimitadora
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Muestra la etiqueta de la clase y la puntuación
                label_text = f"{class_name} {score:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_y = max(int(y1) - 10, 0)  # Asegura que el texto no se salga del frame
                cv2.rectangle(frame, (int(x1), label_y - label_size[1]),
                              (int(x1) + label_size[0], label_y), (0, 255, 0), -1)
                cv2.putText(frame, label_text, (int(x1), label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Escribe el frame procesado en el video de salida
        out.write(frame)

    print(f"El video procesado se ha guardado en {output_video_path}")
finally:
    # Libera recursos al terminar
    cap.release()
    out.release()
    cv2.destroyAllWindows()

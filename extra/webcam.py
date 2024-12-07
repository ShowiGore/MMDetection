import cv2
from mmdet.apis import init_detector, inference_detector
import time  # Para medir el tiempo entre frames

# Specify the configuration file for the model.
config_file = '../configs/rtmdet_tiny_8xb32-300e_coco.py'
#config_file = '../fine_tuning/fine_tuning.py'

# Specify the pre-trained model checkpoint file.
checkpoint_file = '../configs/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
#checkpoint_file = '../fine_tuning/epoch_50.pth'

# Inicializa el modelo
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # Usa 'cpu' si no tienes GPU

# Obtén los nombres de las clases desde la metadata del modelo
class_names = model.dataset_meta['classes']

# Captura de video de la webcam
cap = cv2.VideoCapture(4)  # Usa 0 para la webcam predeterminada

# Define el umbral de confianza para mostrar detecciones
score_thr = 0.4

# Nombre único y fijo para la ventana
window_name = 'Detección en tiempo real'
cv2.namedWindow(window_name)  # Crea una única ventana al inicio

# Inicializa la variable de tiempo para calcular FPS
prev_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la webcam")
            break

        # Marca el inicio del tiempo para calcular FPS
        start_time = time.time()

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

        # Calcula FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Muestra los FPS en la esquina superior izquierda del frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Muestra el frame en la ventana única
        cv2.imshow(window_name, frame)

        # Verifica si la ventana ha sido cerrada
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Libera recursos y cierra todas las ventanas al salir
    cap.release()
    cv2.destroyAllWindows()

import cv2


def list_webcams(max_index=10):
    """Lista todas las cámaras disponibles en el sistema."""
    available_cams = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cams.append(index)
            cap.release()

    return available_cams


def select_webcam():
    """Permite al usuario seleccionar una webcam de las disponibles."""
    webcams = list_webcams()
    if not webcams:
        print("No se encontraron cámaras disponibles.")
        return None

    print("Cámaras disponibles:")
    for i, cam_index in enumerate(webcams):
        print(f"{i}: Cámara {cam_index}")

    while True:
        try:
            selection = int(input(f"Selecciona una cámara (0-{len(webcams) - 1}): "))
            if 0 <= selection < len(webcams):
                print(f"Has seleccionado la cámara {webcams[selection]}")
                return webcams[selection]
            else:
                print("Selección no válida. Inténtalo de nuevo.")
        except ValueError:
            print("Por favor, introduce un número válido.")


# Ejecuta el script para listar y seleccionar una webcam
if __name__ == "__main__":
    webcam_index = select_webcam()
    if webcam_index is not None:
        print(f"Puedes usar la cámara seleccionada con el índice: {webcam_index}")

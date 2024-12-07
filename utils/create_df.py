import os
import json
import shutil
from sklearn.model_selection import train_test_split

# Configuración
frames_folder = "data/frames/"  # Carpeta con los frames extraídos
annotation_file = "data/annotations.json"  # Archivo JSON original
output_folder = "data/dataset/"  # Carpeta donde se guardarán los conjuntos divididos

os.makedirs(output_folder, exist_ok=True)

# Crear subcarpetas
for subset in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, subset), exist_ok=True)

# Leer las anotaciones originales
with open(annotation_file, "r") as f:
    annotations = json.load(f)

# Obtener los nombres de las imágenes
image_files = [img["file_name"] for img in annotations["images"]]
image_ids = [img["id"] for img in annotations["images"]]

# Dividir los datos en 80% train, 10% val, 10% test
train_files, temp_files, train_ids, temp_ids = train_test_split(
    image_files, image_ids, test_size=0.2, random_state=42
)
val_files, test_files, val_ids, test_ids = train_test_split(
    temp_files, temp_ids, test_size=0.5, random_state=42
)

# Función para crear un nuevo JSON para un subconjunto
def create_subset_json(subset_files, subset_ids, subset_name):
    subset_images = [img for img in annotations["images"] if img["id"] in subset_ids]
    subset_annotations = [
        ann for ann in annotations["annotations"] if ann["image_id"] in subset_ids
    ]
    subset_data = {
        "images": subset_images,
        "annotations": subset_annotations,
        "categories": annotations["categories"],
    }

    # Guardar el JSON
    with open(os.path.join(output_folder, f"annotation_{subset_name}.json"), "w") as f:
        json.dump(subset_data, f, indent=4)

    # Mover las imágenes correspondientes
    for file_name in subset_files:
        src = os.path.join(frames_folder, file_name)
        dst = os.path.join(output_folder, subset_name, file_name)
        shutil.copy(src, dst)

# Crear conjuntos de entrenamiento, validación y prueba
create_subset_json(train_files, train_ids, "train")
create_subset_json(val_files, val_ids, "val")
create_subset_json(test_files, test_ids, "test")

print("División completada. Los conjuntos están en", output_folder)

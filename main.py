import onnxruntime as ort
import numpy as np
from PIL import Image
import ijson
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python main.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Erreur : Le fichier {image_path} n'existe pas.")
    sys.exit(1)

with open('hierarchy_map.json', 'rb') as f:
    ordres = set()
    familles = set()
    genres = set()
    especes = set()
    for key, value in ijson.kvitems(f, 'hierarchy_map'):
        ordres.add(value['ordre'])
        familles.add(value['famille'])
        genres.add(value['genre'])
        especes.add(value['espece'])

ordre_classes = sorted(list(ordres))
famille_classes = sorted(list(familles))
genre_classes = sorted(list(genres))
espece_classes = sorted(list(especes))

session = ort.InferenceSession("insect_model.onnx")

image = Image.open(image_path).convert('RGB').resize((224, 224))

# Convertir en numpy array et prétraiter
image_array = np.array(image).astype(np.float32) / 255.0  # Scale to 0-1
image_array = image_array.transpose(2, 0, 1)  # HWC to CHW

# Normaliser avec les mêmes valeurs que ImageNet
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
image_array = (image_array - mean) / std

input_tensor = image_array[np.newaxis, ...]  # Add batch dim

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

outputs = session.run(output_names, {input_name: input_tensor})

for name, output in zip(output_names, outputs):
    predicted = np.argmax(output, axis=1)[0]
    classes = globals()[f"{name}_classes"]
    print(f"{name}: {classes[predicted]}")

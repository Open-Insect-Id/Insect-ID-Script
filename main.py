import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import ijson

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

image_path = "image.jpg"
image = Image.open(image_path).convert('RGB')

# Appliquer les mêmes transformations que pour l’entraînement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(image).unsqueeze(0).numpy()

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]

outputs = session.run(output_names, {input_name: input_tensor})

for name, output in zip(output_names, outputs):
    predicted = np.argmax(output, axis=1)[0]
    classes = globals()[f"{name}_classes"]
    print(f"{name}: {classes[predicted]}")

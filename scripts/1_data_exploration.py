import os
import matplotlib.pyplot as plt
import random
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ruta base del dataset
DATA_DIR = Path("data/raw/chest_xray")

# Clases
sets = ['train', 'val', 'test']
categories = ['NORMAL', 'PNEUMONIA']

# Mostrar cantidad de imágenes por clase
def contar_imagenes():
    print("Conteo de imágenes por clase:\n")
    for set_name in sets:
        print(f"Conjunto: {set_name.upper()}")
        for category in categories:
            dir_path = DATA_DIR / set_name / category
            count = len(list(dir_path.glob("*.jpeg")))
            print(f" - {category}: {count} imágenes")
        print()

# Visualizar imágenes aleatorias
def mostrar_imagenes(set_name='train', samples=3):
    print(f"\nMostrando {samples} imágenes de cada clase en el set {set_name.upper()}:")
    plt.figure(figsize=(12, 4))
    for i, category in enumerate(categories):
        dir_path = DATA_DIR / set_name / category
        images = list(dir_path.glob("*.jpeg"))
        for j in range(samples):
            img_path = random.choice(images)
            img = load_img(img_path, target_size=(150, 150))
            plt.subplot(2, samples, i * samples + j + 1)
            plt.imshow(img)
            plt.title(f"{category}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    contar_imagenes()
    mostrar_imagenes()

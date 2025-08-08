<!-- # Fase 1: Preparación de datos

## 📦 Dataset
- Origen: [Chest X-Ray Pneumonia (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## 🧱 Estructura de carpetas
📁 chest_xray/ <br>
├── train/<br>
│ ├── NORMAL/<br>
│ └── PNEUMONIA/<br>
├── val/<br>
├── test/<br>


## 🧹 Preprocesamiento
- Redimensionamiento de imágenes: 150x150
- Conversión a escala de grises (si aplica)
- Normalización de píxeles: `[0, 1]`

## 📊 Exploración inicial
- Cantidad de imágenes por clase
- Imágenes de muestra por clase -->


# 📂 Fase 1: Carga y Preprocesamiento de Datos

## 🎯 Objetivo
Preparar los datos de radiografías para el entrenamiento del modelo de detección de neumonía, incluyendo la descarga, estructura y preprocesamiento.

## 🗂️ Estructura del Dataset

El dataset está organizado en la carpeta `data/` con la siguiente jerarquía:

data/
│
├── raw/ # Radiografías originales
│ ├── train/
│ ├── val/
│ └── test/
│
└── processed/ # Radiografías redimensionadas y normalizadas
├── train/
├── val/
└── test/


## 🧾 Script utilizado

```bash
python scripts/download_dataset.py
python scripts/preprocesamiento_datos.py

```

## 🛠️ Preprocesamiento
  * Redimensión de imágenes a 150x150 píxeles
  * Normalización de valores (0–255 → 0–1)
  * Separación en conjuntos: train, val, test

## 🧪 Generadores
Se crearon generadores con ImageDataGenerator para alimentar el modelo con batches:
```bash
train_generator, val_generator, test_generator = crear_generadores(...)
```




# Diagnóstico Automatizado de Neumonía con Deep Learning

Este proyecto implementa una red neuronal convolucional (CNN) para la detección de neumonía a partir de radiografías de tórax, utilizando el dataset Chest X-Ray Pneumonia de Kaggle.

---

## 🔍 Objetivo del Proyecto

Desarrollar un modelo de Deep Learning capaz de diferenciar entre radiografías de tórax normales y con neumonía, brindando una herramienta de apoyo al diagnóstico médico en entornos clínicos.
---

## 📚 Navegación del Proyecto

| Documento                                             | Descripción                                              |
| ----------------------------------------------------- | -------------------------------------------------------- |
| [docs/01\_datos.md](docs/01_datos.md)                 | Carga de datos y configuración de generadores            |
| [docs/02\_modelo.md](docs/02_modelo.md)               | Arquitectura del modelo CNN                              |
| [docs/03\_entrenamiento.md](docs/03_entrenamiento.md) | Proceso de entrenamiento del modelo                      |
| [docs/04\_evaluacion.md](docs/04_evaluacion.md)       | Evaluación del modelo con métricas                       |
| [docs/05\_predicciones.md](docs/05_predicciones.md)   | Generación de predicciones y visualización de resultados |
| [docs/06\_despliegue.md](docs/06_despliegue.md)       | Interfaz y despliegue con Streamlit o Flask              |


---

## 🗂️ Estructura del Proyecto

pneumonia-dx/
│
├── 📁 data/              # Datasets
│   ├── raw/             # Radiografías originales
│   └── processed/       # Imágenes preprocesadas y aumentadas
│
├── 📁 models/            # Modelos entrenados
│   ├── mejor_modelo.h5
│   └── modelo_final.h5
│
├── 📁 notebooks/         # Exploración y análisis (EDA)
│   └── eda_model_exploration.ipynb
│
├── 📁 reports/           # Resultados: métricas, gráficas
│   ├── confusion_matrix.png
│   └── entrenamiento_vs_validacion.png
│
├── 📁 src/               # Código fuente del proyecto
│   ├── cargar_datos.py
│   ├── modelo_cnn.py
│   └── entrena_modelo.py
│
├── 📁 docs/              # Documentación técnica de cada módulo
│   ├── 01_carga_dataset.md
│   ├── 02_preprocesamiento.md
│   ├── 03_modelo_cnn.md
│   ├── 04_entrenamiento.md
│   ├── 05_visualizacion.md
│   └── 06_interfaz_despliegue.md
│
├── 📁 scripts/           # Automatizaciones
│   └── download_dataset.py
│
├── 📄 requirements.txt   # Dependencias
├── 📄 README.md          # Visión general del proyecto
└── 📄 LICENSE            # Licencia de uso




# Guías por Módulo

| Archivo                     | Descripción                                         |
| --------------------------- | --------------------------------------------------- |
| `01_carga_dataset.md`       | Carga del dataset desde Kaggle                      |
| `02_preprocesamiento.md`    | Procesamiento de imágenes y creación de generadores |
| `03_modelo_cnn.md`          | Construcción del modelo CNN                         |
| `04_entrenamiento.md`       | Entrenamiento del modelo y early stopping           |
| `05_visualizacion.md`       | Visualización de métricas y análisis de resultados  |
| `06_interfaz_despliegue.md` | Despliegue de interfaz web con Streamlit (opcional) |






---

# 🧪 Fases del Proyecto

| Etapa | Descripción                                         | Estado         |
| ----- | --------------------------------------------------- | -------------- |
| 1️⃣   | Carga y exploración del dataset                     | ✅ Completado   |
| 2️⃣   | Preprocesamiento de imágenes y `ImageDataGenerator` | ✅ Completado   |
| 3️⃣   | Construcción del modelo CNN                         | ✅ Completado   |
| 4️⃣   | Entrenamiento con Early Stopping y validación       | ✅ Completado   |
| 5️⃣   | Evaluación y visualización de métricas              | 🔄 En progreso |
| 6️⃣   | Despliegue web con Streamlit o Flask                | ⏳ Pendiente    |


---

## 📦 Dataset Utilizado

**Chest X-Ray Pneumonia**:  
- Fuente: Kaggle  
- Clases: `NORMAL`, `PNEUMONIA`  
- División: `/train`, `/test`, `/val` (usamos `validation_split` para separar val)

---

## ⚙️ Requisitos del Entorno

```bash
Python 3.10+
TensorFlow >= 2.10
matplotlib
numpy
pandas


## 👇 Instalación Rapida
pip install -r requirements.txt


## ✍️ Autor
Brayan Garzón
Ingeniero en Informática - Esp. En Seguridad Informatica
Proyecto para publicación académica y demostración de IA aplicada al sector salud.
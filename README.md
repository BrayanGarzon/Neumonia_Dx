# Diagnóstico Automatizado de Neumonía con Deep Learning

Este proyecto implementa una red neuronal convolucional (CNN) para la detección de neumonía a partir de radiografías de tórax, utilizando el dataset **Chest X-Ray Pneumonia** de Kaggle.

---

## 🔍 Objetivo del Proyecto

Construir un modelo de Deep Learning capaz de diferenciar entre radiografías de tórax normales y con neumonía, permitiendo una ayuda al diagnóstico clínico en entornos hospitalarios.

---

## 🗂️ Estructura del Proyecto

pneumonia-dx/
│
├── .venv/                        # Virtual environment
│
├── 📁 data/                      # Datasets
│   ├── raw/                     # Raw data (original chest X-rays)
│   └── processed/               # Preprocessed and augmented data
│
├── 📁 notebooks/                 # Jupyter notebooks for analysis
│   └── eda_model_exploration.ipynb
│
├── 📁 models/                    # Trained models (.h5, .pt, etc.)
│   └── best_model.h5
│
├── 📁 reports/                   # Evaluation results, plots, confusion matrices
│   └── confusion_matrix.png
│
├── 📁 src/                       # Source code
│   ├── data_loader/             # Data loading and preprocessing
│   │   └── dataset.py
│   ├── training/                # Training loop, metrics, validation
│   │   └── train.py
│   ├── utils/                   # Utility functions (visualization, helpers)
│   │   └── helpers.py
│   └── interface/               # Interface code (e.g., Streamlit, Flask)
│       └── app.py
│
├── 📁 scripts/                   # Automation scripts
│   └── download_dataset.py      # Script to download Kaggle dataset
│
├── 📄 requirements.txt          # Project dependencies
├── 📄 README.md                 # Documentation and project overview
└── 📄 LICENSE                   # License file



# Traduction of leguage 🌐

| Español            | Inglés técnico sugerido             |
| ------------------ | ----------------------------------- |
| Interfaz gráfica   | `interface/` o `ui/`                |
| Script             | `main.py`, `run.py`, `predictor.py` |
| Modelo entrenado   | `models/`                           |
| Datos procesados   | `data/processed/`                   |
| Datos sin procesar | `data/raw/`                         |
| Cuaderno Jupyter   | `notebooks/`                        |
| Funciones comunes  | `utils/`                            |
| Pruebas            | `tests/`                            |





---

## 📌 Fases del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1️⃣   | **Carga y exploración del dataset** | ✅ Completado |
| 2️⃣   | **Preprocesamiento de imágenes y creación de generadores (`ImageDataGenerator`)** | ✅ Completado |
| 3️⃣   | **Definición del modelo CNN (`modelo_cnn.py`)** | 🔄 En progreso |
| 4️⃣   | Entrenamiento del modelo con los datos | ⏳ Pendiente |
| 5️⃣   | Evaluación del modelo con datos de prueba | ⏳ Pendiente |
| 6️⃣   | Visualización de métricas (matriz de confusión, curvas ROC/AUC) | ⏳ Pendiente |
| 7️⃣   | Exportación y despliegue (opcional con Streamlit/Flask) | ⏳ Pendiente |

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
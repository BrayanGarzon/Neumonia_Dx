# Diagnóstico Automatizado de Neumonía con Deep Learning

Este proyecto implementa una red neuronal convolucional (CNN) para la detección de neumonía a partir de radiografías de tórax, utilizando el dataset Chest X-Ray Pneumonia de Kaggle.

---

## 🔍 Objetivo del Proyecto

Desarrollar un modelo de Deep Learning capaz de diferenciar entre radiografías de tórax normales y con neumonía, brindando una herramienta de apoyo al diagnóstico médico en entornos clínicos.
---

## 📊 Rendimiento del Modelo

- **Precisión global (Accuracy):** 94.8%
- **Sensibilidad (Recall) para neumonía:** 99.3%  
  _Prácticamente no se escapan casos positivos._
- **Especificidad para casos normales:** 88.5%
- **F1-Score global:** 94.7%
- **Matriz de confusión en test set:**  
  _(ver imagen o sección correspondiente)_

> 🔍 **Estrategia adoptada:**  
> Se prioriza la detección de neumonía, permitiendo falsos positivos que luego serán revisados en una segunda valoración médica.

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

pneumonia-dx

├── 📁 data/              # Dataset  
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
└── 📄 LICENSE            #




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


---

## 🧪 Fases del Proyecto

| Etapa | Descripción                                              | Estado         |
| ----- | -------------------------------------------------------- | -------------- |
| 1️⃣   | Carga y exploración del dataset                          | ✅ Completado   |
| 2️⃣   | Preprocesamiento de imágenes y `ImageDataGenerator`      | ✅ Completado   |
| 3️⃣   | Construcción del modelo CNN                              | ✅ Completado   |
| 4️⃣   | Entrenamiento con Early Stopping y validación            | ✅ Completado   |
| 5️⃣   | Evaluación inicial y análisis de métricas                | ✅ Completado   |
| 6️⃣   | Ajuste de umbral y estrategia para falsos positivos      | ✅ Completado   |
| 7️⃣   | Despliegue web con interfaz visual y heatmap explicativo | 🔄 En progreso |


---

## 📦 Dataset Utilizado

**Chest X-Ray Pneumonia**:  
- Fuente: Kaggle  
- Clases: `NORMAL`, `PNEUMONIA`  
- División: `/train`, `/test`, `/val` (o `validation_split` en código)

---

## ⚙️ Requisitos del Entorno

```bash
Python 3.10+
TensorFlow >= 2.10
matplotlib
numpy
pandas

---

## 📦 Dataset Utilizado

**Chest X-Ray Pneumonia**:  
- Fuente: Kaggle  
- Clases: `NORMAL`, `PNEUMONIA`  
- División: `/train`, `/test`, `/val` (usamos `validation_split` para separar val)

---


## 👇 Instalación Rapida
pip install -r requirements.txt


## 🚀 Próximas Mejoras

### Mejorar balance de clases
  - Ajustar umbral de decisión para reducir falsos positivos.
  - Usar `class_weight` y aumento de datos de la clase NORMAL.

### Aumentar robustez
  - Entrenar con más variedad de imágenes.
  - Aplicar data augmentation avanzada.
  - Probar arquitecturas como EfficientNet o DenseNet.

### Interpretabilidad
  - Integrar Grad-CAM para resaltar áreas de interés en radiografías.

### Integración clínica
  - Interfaz web clara para carga de imágenes y reporte automático.
  - Opción de exportar reportes PDF.
  - Doble validación médica.

### Seguridad y regulaciones
  - Registro de predicciones.
  - Cumplimiento



## ✍️ Autor
Brayan Garzón
Ingeniero en Informática - Esp. En Seguridad Informatica
Proyecto para publicación académica y demostración de IA aplicada al sector salud.
# evaluar_modelo_test.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# 1. Configuración
# ==============================
MODEL_PATH = "models/modelo_final.keras"
TEST_DIR = "data/raw/chest_xray/test"
RESULTS_DIR = "results"
THRESHOLD = 0.3  # 🔹 Cambia este valor para ajustar el umbral

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# 2. Cargar modelo
# ==============================
print("🔹 Cargando modelo entrenado...")
model = load_model(MODEL_PATH)

# ==============================
# 3. Generador para test
# ==============================
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# ==============================
# 4. Predicciones con umbral personalizado
# ==============================
print("🔹 Generando predicciones...")
predictions = model.predict(test_generator)
predicted_classes = (predictions > THRESHOLD).astype("int32").ravel()
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# ==============================
# 5. Reporte de clasificación
# ==============================
report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)

# Guardar reporte en TXT
report_path = os.path.join(RESULTS_DIR, "classification_report_test.txt")
with open(report_path, "w") as f:
    from sklearn.metrics import classification_report
    f.write(classification_report(true_classes, predicted_classes, target_names=class_labels))

print(f"📄 Reporte de clasificación guardado en: {report_path}")

# ==============================
# 6. Matriz de confusión
# ==============================
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_test.png"))
plt.close()

# # ==============================
# # 7. Guardar gráficas de métricas
# # ==============================
# metrics_names = ["precision", "recall", "f1-score"]
# metrics_values = [report["PNEUMONIA"][m] for m in metrics_names]  # Cambiar nombre de clase si es distinto

# plt.bar(metrics_names, metrics_values, color=["#4CAF50", "#2196F3", "#FF5722"])
# plt.ylim(0, 1)
# plt.title(f"Métricas de Rendimiento - Neumonía (Umbral={THRESHOLD})")
# plt.savefig(os.path.join(RESULTS_DIR, "metrics_bar_test.png"))
# plt.close()

# ==============================
# 7. Guardar gráficas de métricas
# ==============================
metrics_names = ["precision", "recall", "f1-score"]

# Usar índice correcto de la clase PNEUMONIA desde class_labels
if "PNEUMONIA" in class_labels:
    pneumonia_label = "PNEUMONIA"
else:
    pneumonia_label = class_labels[1]  # asume que la segunda clase es PNEUMONIA

metrics_values = [report[pneumonia_label][m] for m in metrics_names]

plt.bar(metrics_names, metrics_values, color=["#4CAF50", "#2196F3", "#FF5722"])
plt.ylim(0, 1)
plt.title(f"Métricas de Rendimiento - {pneumonia_label} (Umbral={THRESHOLD})")
plt.savefig(os.path.join(RESULTS_DIR, f"metrics_bar_{pneumonia_label}.png"))
plt.close()


print("✅ Evaluación completada con umbral personalizado. Resultados guardados en carpeta 'results/'")

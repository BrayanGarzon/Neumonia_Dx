import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from load_data import val_generator
import pickle

# Crear carpeta de resultados si no existe
os.makedirs("results", exist_ok=True)

# =============================
# 1. Cargar el modelo entrenado
# =============================
modelo = load_model("models/mejor_modelo.keras")


# =============================
# 2. Evaluación
# =============================
loss, accuracy = modelo.evaluate(val_generator)
print(f"📊 Pérdida (loss): {loss:.4f}")
print(f"✅ Precisión (accuracy): {accuracy:.4f}")

# =============================
# 3. Predicciones
# =============================
y_pred_probs = modelo.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

# Nombres de las clases
class_names = list(val_generator.class_indices.keys())

# =============================
# 4. Reporte de clasificación
# =============================
reporte = classification_report(y_true, y_pred, target_names=class_names)
print("📄 Reporte de clasificación:")
print(reporte)

# Guardar el reporte en un archivo .txt
with open("results/reporte_clasificacion.txt", "w") as f:
    f.write(reporte)

# =============================
# 5. Matriz de confusión
# =============================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# =============================
# 6. Gráfica de accuracy y pérdida
# =============================
try:
    with open('models/historial_entrenamiento.pkl', 'rb') as f:
        historial = pickle.load(f)

    acc = historial.history['accuracy']
    val_acc = historial.history['val_accuracy']
    loss = historial.history['loss']
    val_loss = historial.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Entrenamiento')
    plt.plot(epochs, val_acc, 'go-', label='Validación')
    plt.title('Precisión por Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Entrenamiento')
    plt.plot(epochs, val_loss, 'go-', label='Validación')
    plt.title('Pérdida por Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/metricas_entrenamiento.png")
    plt.close()

except FileNotFoundError:
    print("⚠ No se encontró el historial de entrenamiento (.pkl). Solo se generará el reporte y la matriz de confusión.")

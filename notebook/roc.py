import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- 1️⃣ Cargar datos de test ---
test_dir = 'data/raw/chest_xray/test'  # ajusta si está en otra carpeta
img_size = (150, 150)  # tamaño que usaste en el entrenamiento
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# --- 2️⃣ Cargar el modelo entrenado ---
model = tf.keras.models.load_model('models/mejor_modelo.keras')

# --- 3️⃣ Obtener y_test y y_pred_proba ---
y_test = test_generator.classes
y_pred_proba = model.predict(test_generator)[:, 0]  # Probabilidad de PNEUMONIA

# --- 4️⃣ Calcular curva ROC ---
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - CNN Neumonía')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# --- 5️⃣ Guardar en reports ---
plt.savefig('reports/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# threshold_sweep.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# Config (ajusta si hace falta)
# -------------------------
MODEL_PATH = "models/mejor_modelo.keras"   # <- pon aqui 'mejor_modelo.keras' o 'modelo_final.keras'
TEST_DIR = "data/raw/chest_xray/test"      # <- tu carpeta test
RESULTS_DIR = "results"
TARGET_RECALL = 0.98   # objetivo mínimo de recall para PNEUMONIA (ajusta si quieres)
BINS = 51              # número de umbrales a evaluar (51 => 0.00..1.00 step 0.02)
BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------
# Carga modelo y datos
# -------------------------
print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(150,150),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

y_true = test_gen.classes
print("Found", len(y_true), "test images. Classes:", test_gen.class_indices)

# -------------------------
# Predict probs
# -------------------------
print("Predicting probabilities on test set...")
y_prob = model.predict(test_gen).ravel()  # shape (N,)

# -------------------------
# Sweep thresholds
# -------------------------
thresholds = np.linspace(0.0, 1.0, BINS)
precisions = []
recalls = []
f1s = []
false_positives = []
specificities = []

for t in thresholds:
    y_pred = (y_prob > t).astype(int)
    precisions.append(precision_score(y_true, y_pred, zero_division=0))
    recalls.append(recall_score(y_true, y_pred, zero_division=0))
    f1s.append(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)  # [[TN, FP],[FN, TP]]
    if cm.shape == (2,2):
        fp = int(cm[0,1])
        tn = int(cm[0,0])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        fp = 0
        specificity = 0.0
    false_positives.append(fp)
    specificities.append(specificity)

# -------------------------
# Save curves
# -------------------------
plt.figure(figsize=(8,6))
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, f1s, label='F1')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision / Recall / F1 vs Threshold')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'threshold_metrics.png'))
plt.close()

plt.figure(figsize=(6,4))
plt.plot(thresholds, false_positives, marker='o')
plt.xlabel('Threshold')
plt.ylabel('False Positives (NORMAL->PNEUMONIA)')
plt.title('False positives vs Threshold')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'threshold_false_positives.png'))
plt.close()

plt.figure(figsize=(6,4))
plt.plot(thresholds, specificities, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Specificity (TN / (TN+FP))')
plt.title('Specificity vs Threshold')
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'threshold_specificity.png'))
plt.close()

# -------------------------
# Suggest best threshold keeping recall >= TARGET_RECALL
# -------------------------
candidates = []
for t,p,r,f in zip(thresholds, precisions, recalls, false_positives):
    if r >= TARGET_RECALL:
        candidates.append((t,p,r,f))

if candidates:
    best = max(candidates, key=lambda x: x[1])  # maximize precision among candidates
    print("\nCandidates found with recall >= {:.2f}: {}".format(TARGET_RECALL, len(candidates)))
    print("Suggested threshold (maximize precision while keeping recall >= {:.2f}):".format(TARGET_RECALL))
    print("  threshold = {:.3f}, precision = {:.3f}, recall = {:.3f}, false_positives = {}".format(*best))
else:
    print("\nNo threshold keeps recall >= {:.2f}. Consider lowering TARGET_RECALL or retraining.".format(TARGET_RECALL))

# Print top few thresholds table
print("\nTop thresholds by F1 (top 5):")
rows = [(t,p,r,f) for t,p,r,f in zip(thresholds, precisions, recalls, false_positives)]
rows = sorted(rows, key=lambda x: x[2]*x[1], reverse=True)  # heuristica
for r in rows[:5]:
    print("  th={:.3f}  prec={:.3f}  recall={:.3f}  FP={}".format(*r))

# ROC AUC
try:
    auc = roc_auc_score(y_true, y_prob)
    print("\nROC AUC:", auc)
except Exception as e:
    print("ROC AUC error:", e)

print("\nSaved: threshold_metrics.png, threshold_false_positives.png, threshold_specificity.png in", RESULTS_DIR)

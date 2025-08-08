# 🚀 Fase 3: Entrenamiento del Modelo

## 🎯 Objetivo
Entrenar el modelo CNN utilizando los datos preprocesados para que aprenda a distinguir entre radiografías normales y con neumonía.

## 🧠 Configuración de entrenamiento

- 📊 **Epochs:** hasta 30
- ⏹️ **EarlyStopping:** monitorea `val_loss`, con paciencia de 5 épocas
- 💾 **Checkpoint:** guarda el mejor modelo como `mejor_modelo.h5`
- ✅ Se utiliza `model.fit()` con `train_generator` y `val_generator`

## 🧾 Código del entrenamiento

```python
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modelo_cnn import construir_modelo_cnn
from cargar_datos import train_generator, val_generator

os.makedirs("models", exist_ok=True)

modelo = construir_modelo_cnn(input_shape=(150, 150, 3))

checkpoint_cb = ModelCheckpoint("models/mejor_modelo.h5", 
                                monitor="val_loss", 
                                save_best_only=True, 
                                verbose=1)

earlystop_cb = EarlyStopping(monitor="val_loss", 
                             patience=5, 
                             restore_best_weights=True, 
                             verbose=1)

historial = modelo.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint_cb, earlystop_cb]
)

modelo.save("models/modelo_final.h5")
```

## 📦 Archivo involucrado
    * entrena_modelo.py
import os
import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modelo_cnn import construir_modelo_cnn
from cargar_datos import train_generator, val_generator 

# Crear carpeta para modelos si no existe
os.makedirs("models", exist_ok=True)

# Crear el modelo
modelo = construir_modelo_cnn(input_shape=(150, 150, 3))

# Callbacks
checkpoint_cb = ModelCheckpoint("models/mejor_modelo.keras", 
                                monitor="val_loss", 
                                save_best_only=True, 
                                verbose=1)

earlystop_cb = EarlyStopping(monitor="val_loss", 
                             patience=5, 
                             restore_best_weights=True, 
                             verbose=1)

# Entrenamiento
historial = modelo.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Guardar el modelo final
modelo.save("models/modelo_final.keras")

# Guardar historial de entrenamiento
with open("models/historial_entrenamiento.json", "w") as f:
    json.dump(historial.history, f)

print("✅ Entrenamiento finalizado y modelo guardado.")

# 🧠 Fase 2: Construcción del Modelo CNN

## 🎯 Objetivo
Diseñar una red neuronal convolucional (CNN) capaz de clasificar radiografías de tórax como normales o con neumonía.

## 🏗️ Arquitectura

El modelo fue construido con Keras utilizando `Sequential`. La arquitectura es la siguiente:

- 📥 Input: 150x150x3
- 🧱 3 bloques de convolución + max pooling
- 🧠 Capa densa oculta de 128 unidades
- 🔚 Capa de salida con activación `sigmoid` para clasificación binaria

## 🧾 Código del Modelo

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def construir_modelo_cnn(input_shape=(150, 150, 3)):
    modelo = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    
    modelo.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
    
    return modelo
```

## 📦 Archivo involucrado
    - modelo_cnn.py
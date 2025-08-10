#Explicación rápida del modelo:
    #Conv2D + MaxPooling2D: detecta patrones espaciales en las imágenes.
    #BatchNormalization: estabiliza el aprendizaje.
    #Dropout (0.5): ayuda a evitar el sobreajuste.
    #Sigmoid: salida binaria para NORMAL vs PNEUMONIA


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def construir_modelo_cnn(input_shape=(150, 150, 3)):
    modelo = Sequential()

    # Bloque 1
    modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloque 2
    modelo.add(Conv2D(64, (3, 3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloque 3
    modelo.add(Conv2D(128, (3, 3), activation='relu'))
    modelo.add(BatchNormalization())
    modelo.add(MaxPooling2D(pool_size=(2, 2)))

    # Capa densa final
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dropout(0.5))
    modelo.add(Dense(1, activation='sigmoid'))  # Clasificación binaria

    # Compilación del modelo
    modelo.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    return modelo



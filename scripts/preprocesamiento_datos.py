from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parámetros base
image_size = (150, 150)
batch_size = 32

# Generador con aumento de datos para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # usamos una parte como validación
)

# Generador solo con reescalado para validación y test
test_val_datagen = ImageDataGenerator(rescale=1./255)

# Ruta base del dataset
dataset_path = "data/raw/chest_xray"  # Ajusta esto a tu path real

# Generador de entrenamiento
train_generator = train_datagen.flow_from_directory(
    directory=dataset_path + "/train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Generador de validación
val_generator = train_datagen.flow_from_directory(
    directory=dataset_path + "/train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Generador de prueba (test)
test_generator = test_val_datagen.flow_from_directory(
    directory=dataset_path + "/test",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)




import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Ruta del archivo kaggle.json
KAGGLE_JSON_PATH = os.path.join(os.path.dirname(__file__), '..', 'kaggle', 'kaggle.json')

# Asegurar que la variable de entorno esté configurada
os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kaggle'))

# Inicializar API
api = KaggleApi()
api.authenticate()

# Nombre del dataset en Kaggle
dataset_name = 'paultimothymooney/chest-xray-pneumonia'

# Ruta destino
destination_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Descargar
print("Descargando dataset desde Kaggle...")
api.dataset_download_files(dataset_name, path=destination_path, unzip=True)
print("✅ Descarga completada.")

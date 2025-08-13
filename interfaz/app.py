# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# --- Configuración de la página ---
st.set_page_config(
    page_title="Detección de Neumonía",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 Detección de Neumonía con IA")
st.write("Este sistema utiliza una red neuronal convolucional para ayudar en la detección de neumonía a partir de radiografías de tórax.")

# --- Cargar modelo ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/mejor_modelo.keras")
    return model

model = load_model()

# --- Subida de imagen ---
uploaded_file = st.file_uploader("Sube una radiografía en formato JPG o PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiografía cargada", use_column_width=True)

    # --- Preprocesar imagen ---
    img_size = (150, 150)  # tamaño usado en entrenamiento
    img_array = image.resize(img_size)
    img_array = np.array(img_array) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Predicción ---
    pred_prob = model.predict(img_array)[0][0]
    pred_class = "PNEUMONIA" if pred_prob >= 0.5 else "NORMAL"

    st.subheader("Resultado del análisis")
    st.metric(label="Diagnóstico", value=pred_class, delta=f"{pred_prob*100:.2f}%")

    # --- Mensajes de alerta ---
    if pred_class == "PNEUMONIA":
        if pred_prob >= 0.9:
            st.error("⚠ Alta probabilidad de Neumonía")
        elif pred_prob >= 0.7:
            st.warning("Posible Neumonía - requiere revisión médica")
        else:
            st.info("Caso leve - verificar con especialista")

    # --- Placeholder para Heatmap ---
    st.subheader("Mapa de calor (interpretabilidad)")
    st.write("Aquí se mostrará el heatmap de la radiografía resaltando las zonas más relevantes para el modelo.")
    # TODO: Agregar función de Grad-CAM

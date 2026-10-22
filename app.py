
import json
import os

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model
import paho.mqtt.client as mqtt

# ---------------- CONFIGURACIÓN ----------------
st.set_page_config(
    page_title="Cerradura Inteligente",
    page_icon="🔐",
    layout="centered",
)

BROKER = "broker.mqttdashboard.com"
PORT = 1883
TOPIC = "IMIASCG"
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
UMBRAL = 0.85

# ---------------- INTERFAZ ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #101827, #182b3a);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🔐 Cerradura Inteligente")
st.caption("Control de acceso mediante reconocimiento de gestos")

st.info(
    "Toma una foto del gesto entrenado para evaluar si corresponde "
    "a la clase autorizada."
)

# ---------------- CARGA DEL MODELO ----------------
@st.cache_resource
def cargar_modelo():
    return load_model(MODEL_PATH, compile=False)

@st.cache_resource
def cargar_etiquetas():
    with open(LABELS_PATH, "r", encoding="utf-8") as archivo:
        return [line.strip() for line in archivo if line.strip()]

try:
    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encuentra {MODEL_PATH}.")
        st.stop()

    if not os.path.exists(LABELS_PATH):
        st.error(f"No se encuentra {LABELS_PATH}.")
        st.stop()

    modelo = cargar_modelo()
    etiquetas = cargar_etiquetas()

except Exception as error:
    st.error(f"No fue posible cargar el modelo: {error}")
    st.stop()

if len(etiquetas) != 2:
    st.error("Se esperaban exactamente dos clases en labels.txt.")
    st.stop()

st.markdown("### Gestos configurados")
col1, col2 = st.columns(2)

with col1:
    st.success("✅ Abre cerradura")

with col2:
    st.warning("⛔ No abre cerradura")

# ---------------- CONEXIÓN MQTT ----------------
def enviar_orden(gesto):
    cliente = mqtt.Client(
        client_id="APP_Deteccion_SCG",
        protocol=mqtt.MQTTv311,
    )
    try:
        cliente.connect(BROKER, PORT, keepalive=30)
        cliente.loop_start()

        info = cliente.publish(
            TOPIC,
            json.dumps({"gesto": gesto}),
            qos=0,
            retain=False,
        )
        info.wait_for_publish(timeout=5)
        resultado = info.is_published()

        cliente.loop_stop()
        cliente.disconnect()
        return resultado, None

    except Exception as error:
        try:
            cliente.loop_stop()
            cliente.disconnect()
        except Exception:
            pass
        return False, str(error)

# ---------------- CÁMARA Y PREDICCIÓN ----------------
st.markdown("### 📸 Verificar gesto")
foto = st.camera_input("Toma una foto con el gesto que deseas verificar")

if foto is not None:
    try:
        imagen = Image.open(foto).convert("RGB")
        imagen = ImageOps.fit(imagen, (224, 224))

        arreglo = np.asarray(imagen, dtype=np.float32)
        arreglo = (arreglo / 127.5) - 1.0
        entrada = np.expand_dims(arreglo, axis=0)

        prediccion = modelo.predict(entrada, verbose=0)[0]
        indice = int(np.argmax(prediccion))
        confianza = float(prediccion[indice])

        etiqueta = etiquetas[indice]
        st.metric("Confianza del modelo", f"{confianza:.1%}")
        st.progress(max(0.0, min(1.0, confianza)))

        # Solo la clase 0 autoriza abrir.
        if indice == 0 and confianza >= UMBRAL:
            st.success("✅ Gesto autorizado: abrir cerradura")

            if st.button("🔓 Enviar orden de apertura"):
                with st.spinner("Enviando orden por MQTT..."):
                    ok, error = enviar_orden("Abre")

                if ok:
                    st.success("Orden enviada al broker MQTT.")
                    st.caption(
                        "Verifica que el ESP32 de Wokwi esté conectado "
                        "al mismo broker y tema."
                    )
                else:
                    st.error(f"No se pudo enviar la orden: {error}")

        elif indice == 1 and confianza >= UMBRAL:
            st.error("⛔ Gesto no autorizado")
            st.caption("No se envió ninguna orden a la cerradura.")

        else:
            st.warning(
                "Gesto incierto. No se envió ninguna orden. "
                "Intenta tomar otra foto con mejor iluminación."
            )

        with st.expander("Detalles de la predicción"):
            for i, nombre in enumerate(etiquetas):
                st.write(f"{nombre}: {float(prediccion[i]):.1%}")

    except Exception as error:
        st.error(f"Error al procesar la imagen: {error}")

st.divider()
st.caption(
    "Proyecto académico · Teachable Machine · TensorFlow/Keras · MQTT"
)


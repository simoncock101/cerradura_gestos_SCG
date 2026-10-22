
import json
import os
import uuid

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model
import paho.mqtt.client as mqtt

# ---------- CONFIGURACIÓN ----------
BROKER = "broker.mqttdashboard.com"
PORT = 1883
TOPIC = "IMIASCG"

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
UMBRAL = 0.85

st.set_page_config(
    page_title="Cerradura Inteligente",
    page_icon="🔐",
    layout="centered",
)

# ---------- INTERFAZ ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #203747);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🔐 Cerradura Inteligente")
st.write("Reconocimiento de gestos con Teachable Machine")
st.caption("Streamlit · TensorFlow/Keras · MQTT · ESP32")

# ---------- CARGAR MODELO ----------
@st.cache_resource
def cargar_modelo():
    return load_model(MODEL_PATH, compile=False)

@st.cache_data
def cargar_etiquetas():
    with open(LABELS_PATH, "r", encoding="utf-8-sig") as f:
        etiquetas = []

        for linea in f:
            linea = linea.strip()
            if not linea:
                continue

            # Admite etiquetas como "0 Abre cerradura"
            partes = linea.split(maxsplit=1)
            if len(partes) == 2 and partes[0].isdigit():
                etiquetas.append(partes[1])
            else:
                etiquetas.append(linea)

    return etiquetas

if not os.path.exists(MODEL_PATH):
    st.error("No se encuentra keras_model.h5 en el repositorio.")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error("No se encuentra labels.txt en el repositorio.")
    st.stop()

try:
    model = cargar_modelo()
    labels = cargar_etiquetas()
except Exception as e:
    st.error(f"Error cargando el modelo o las etiquetas: {e}")
    st.stop()

if len(labels) != 2:
    st.error("El modelo debe tener las dos clases de tu entrenamiento.")
    st.stop()

# ---------- ESTADO ----------
st.markdown("### Clases entrenadas")

c1, c2 = st.columns(2)
c1.success("🔓 Abre cerradura")
c2.warning("🚫 No abre cerradura")

st.info(
    "Toma una foto del gesto de apertura. Solo se enviará la orden "
    "si la primera clase supera el umbral de confianza."
)

# ---------- MQTT ----------
def enviar_mqtt(gesto):
    client = mqtt.Client(
        client_id="Streamlit_" + uuid.uuid4().hex[:12],
        protocol=mqtt.MQTTv311,
    )

    try:
        client.connect(BROKER, PORT, keepalive=30)
        client.loop_start()

        info = client.publish(
            TOPIC,
            json.dumps({"gesto": gesto}),
            qos=0,
            retain=False,
        )

        info.wait_for_publish(timeout=8)
        publicado = info.is_published()

        client.loop_stop()
        client.disconnect()

        if not publicado:
            return False, "No se confirmó el envío al cliente MQTT."

        return True, None

    except Exception as e:
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass

        return False, str(e)

# ---------- DETECCIÓN ----------
st.markdown("### 📷 Detectar gesto")
foto = st.camera_input("Toma una foto con tu cámara")

if foto is not None:
    try:
        imagen = Image.open(foto).convert("RGB")

        # Preprocesamiento utilizado por Teachable Machine
        imagen = ImageOps.fit(imagen, (224, 224))
        arr = np.asarray(imagen, dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        entrada = np.expand_dims(arr, axis=0)

        pred = model.predict(entrada, verbose=0)[0]
        indice = int(np.argmax(pred))
        confianza = float(pred[indice])

        st.subheader("Resultado de detección")
        st.write(f"**Gesto:** {labels[indice]}")
        st.metric("Confianza", f"{confianza:.1%}")
        st.progress(max(0.0, min(1.0, confianza)))

        with st.expander("Ver resultados de ambas clases"):
            for i, nombre in enumerate(labels):
                st.write(f"{nombre}: {float(pred[i]):.1%}")

        if indice == 0 and confianza >= UMBRAL:
            st.success("Gesto de apertura reconocido.")

            if st.button("🔓 Enviar orden para abrir"):
                with st.spinner("Enviando orden al ESP32 por MQTT..."):
                    ok, error = enviar_mqtt("Abre")

                if ok:
                    st.success("Orden publicada en el tema IMIASCG.")
                    st.write(
                        "Comprueba la consola serial de Wokwi "
                        "para confirmar que el ESP32 la recibió."
                    )
                else:
                    st.error(f"No se pudo enviar la orden: {error}")

        elif indice == 1 and confianza >= UMBRAL:
            st.warning(
                "El modelo detectó «No abre cerradura». "
                "No se envió ninguna orden."
            )

        else:
            st.warning(
                "La predicción no es suficientemente confiable. "
                "Intenta de nuevo con mejor iluminación."
            )

    except Exception as e:
        st.error(f"No se pudo procesar la imagen: {e}")

st.divider()
st.caption("Prototipo académico de cerradura inteligente.")

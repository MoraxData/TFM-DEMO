import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pandas as pd

# Clases de predicción para la CNN
class_names = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Cargamos el archivo de las enfermedades
file = r'C:\Users\morad\OneDrive\Documentos\Master IA\TFM\enfermedades_tomates.xlsx'
enfermedades = pd.read_excel(file)

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model("C:\\Users\\morad\\OneDrive\\Documentos\\Master IA\\TFM\\models\\modelo_9_mobile.h5")

# Función para predicción basada en imágenes cargadas
def model_predicion(img):
    # Convertir la imagen a formato compatible
    img = img.resize((256, 256))  # Asegurarse de que el tamaño sea 256x256
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convertir a array numpy
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para batch
    img_array /= 255.0  # Normalizar de la misma manera que en el entrenamiento

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probability = predictions[0][predicted_class] * 100  # Convertir a porcentaje
    return predicted_class, predicted_probability

# Función para predicción desde la cámara
def model_predicion_camera(img):
    img = cv2.resize(img, (256, 256))  # Redimensionar la imagen
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expandir dimensiones para batch
    img_array /= 255.0  # Normalizar

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probability = predictions[0][predicted_class] * 100  # Convertir a porcentaje
    return predicted_class, predicted_probability

# Sidebar de navegación
app_mode = st.sidebar.selectbox("Navegación de páginas", ["Inicio", "Prueba del modelo", "Capturar imagen", "Contacto"])

# Página de inicio
if app_mode == "Inicio":
    st.markdown("# Sistema de detección de enfermedades en hojas de tomate")
    img_ruta = "C:\\Users\\morad\\OneDrive\\Documentos\\Master IA\\TFM\\image_web.png"
    st.image(img_ruta, use_column_width=True)
    st.markdown("""
   

Esta aplicación ha sido desarrollada para demostrar la capacidad de las **Redes Neuronales Convolucionales (CNN)** en la clasificación automática de imágenes. Utilizando un modelo de deep learning entrenado, esta herramienta permite a los usuarios cargar imágenes y obtener predicciones precisas en tiempo real.

## ¿Cómo funciona?
- La aplicación utiliza una **CNN entrenada** para predecir enfermedades en hojas de tomate.
- Los usuarios pueden cargar una imagen a través de la interfaz, y el modelo analizará la imagen para proporcionar una **predicción** basada en las características visuales aprendidas durante el entrenamiento.
- El resultado incluye la clase predicha y un grado de **confianza** en la predicción así como información las causas, síntomas, tratamientos y comentarios de la enfermedad capturada

## ¿Cómo utilizar la aplicación?
1. **Carga o captura una imagen**: Usa la ventana de _Prueba del modelo_  para seleccionar una imagen o la ventana _Capturar imagen_ para tomar una fotografía desde tu dispositivo
2. **Procesamiento**: Una vez cargada, la imagen será procesada por el modelo CNN en segundo plano.
3. **Predicción**: La aplicación mostrará la predicción devuelta modelo por el modelo, junto a su probabilidad asociada a la predicción.
4. **Información**: La aplicación aportará información sobre la enfermedad capturada como las causas, tratamientos, comentarios...



## Sobre el Proyecto
Este trabajo se ha desarrollado como **Trabajo de Fin de Máster** para finalizar los estudios del Master en Inteligencia Artificial de la Univerisad Internacional de Valencia. Se presenta esta aplicación como apoyo a este trabajo y poner a pruebla el modelo mediante uso externo de otros usuarios.

- Autor: Morad Charchaoui Oilad Ali
- Fecha última actualización: 30/09/2024
""")

# Página de prueba del modelo
elif app_mode == "Prueba del modelo":
    st.header("Prueba del modelo")

    # Subir una imagen para la predicción
    imagen_test = st.file_uploader("Elija una imagen para realizar la predicción", type=["jpg", "jpeg", "png"])

    if imagen_test is not None:
        # Convertir la imagen a un formato PIL
        imagen = Image.open(imagen_test)
        st.image(imagen, use_column_width=True)

        if st.button("Predecir"):
            pred, prob = model_predicion(imagen)
            st.success(f"El modelo ha predicho que se trata de {class_names[pred]} con una confianza del {prob:.2f}%")
            if class_names[pred]!= 'Tomato___healthy':
                filtro = enfermedades[enfermedades['Enfermedad'] == class_names[pred]]
                st.markdown("""
## Síntomas
""")
                st.write(filtro["Síntomas"].values[0])
                st.markdown("""
## Causas
""")
                st.write(filtro["Causas"].values[0])
                st.markdown("""
## Tratamiento
""")
                st.write(filtro["Tratamiento"].values[0])
                st.markdown("""
## Comentarios
""")
                st.write(filtro["Comentarios"].values[0])
                st.markdown("""
## [+ Info]
[PlantViallge\Tomato](https://plantvillage.psu.edu/topics/tomato/infos)
                            
""")

# Página de captura de imagen (cámara)
elif app_mode == "Capturar imagen":

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.snapshot = None

        def transform(self, frame):
            img_ = frame.to_ndarray(format="bgr24")
            self.snapshot = img_  # Guardar la imagen capturada
            return img_

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    if webrtc_ctx.video_transformer and webrtc_ctx.video_transformer.snapshot is not None:
        if st.button("Predecir"):
            img_ = webrtc_ctx.video_transformer.snapshot
            pred, prob = model_predicion_camera(img_)
            st.success(f"El modelo ha predicho que se trata de {class_names[pred]} con una confianza del {prob:.2f}%")


elif app_mode == "-Contacto":
    st.header("Contacto")
    st.markdown("""
<a href="https://www.linkedin.com/in/morad-c-25b976202" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width:30px; height:30px;">
</a>
""", unsafe_allow_html=True)
    st.markdown("""
<a href="mailto:morad11jr@gmail.com" target="_blank">
    <img src="https://th.bing.com/th/id/R.2630f8f80aa6f55c2f9f775db5c7de96?rik=gkOt8nyE4z239Q&pid=ImgRaw&r=0" alt="Gmail" style="width:40px; height:30px;">
</a>
""", unsafe_allow_html=True)

elif app_mode == "Contacto":
    st.title("Contacto")
    st.markdown("## ¡Conecta conmigo!")

    # Crear columnas para una mejor organización
    col1, col2 = st.columns([1, 4])

    # Colocar el logo de LinkedIn en la primera columna y el enlace en la segunda columna
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png", width=40)

    with col2:
        st.markdown("""
        ### [Perfil en LinkedIn](https://www.linkedin.com/in/morad-c-25b976202)
        """, unsafe_allow_html=True)

    # Espaciado entre secciones
    st.markdown("---")

    # Crear columnas para el logo de Gmail y el correo electrónico
    col1, col2 = st.columns([1, 4])

    # Colocar el logo de Gmail en la primera columna y el correo en la segunda columna
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg", width=40)

    with col2:
        st.markdown("""
        ### [morad11jr@gmail.com](mailto:morad11jr@gmail.com)
        """, unsafe_allow_html=True)

    # Espaciado entre secciones
    st.markdown("---")

    # Mensaje de bienvenida para la interacción
    st.markdown("""
    Si tienes alguna pregunta no dudes en ponerte en contacto. ¡Quedo atento tu mensaje! 😊
    """)


    # Pie de página atractivo
    st.markdown("<hr style='border: 2px solid #f63366;'>", unsafe_allow_html=True)
    st.markdown("Gracias por visitar mi página de contacto.", unsafe_allow_html=True)
    st.markdown(""" 
    Almería, España 
    """)

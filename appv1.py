import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# Config de la página
st.set_page_config(page_title="Sistema de detección", page_icon="🍃", layout="centered")

# Clases de nuestro problema
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Informe Excel de los resultados a devolver
file = 'enfermedades_tomates.xlsx'
enfermedades = pd.read_excel(file)

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model("models/modelo_9_mobile.h5")

# Función para predicción basada en imágenes cargadas
def model_prediccion(img):
    # Preprocesamiento
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalizar

    # Generar predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_probability = predictions[0][predicted_class] * 100
    return predicted_class, predicted_probability

# Página de navegación
app_mode = st.sidebar.radio("Navegación de páginas", ["Inicio", "Prueba del modelo", "Contacto"])

# Página de inicio
if app_mode == "Inicio":
    st.markdown("# Sistema de detección de enfermedades en hojas de tomate")
    img_ruta = "image_web.png"
    st.image(img_ruta, use_column_width=True)
    st.markdown("""
Esta aplicación ha sido desarrollada para demostrar la capacidad de las **Redes Neuronales Convolucionales (CNN)** en la clasificación automática de imágenes. Utiliza un modelo de deep learning entrenado que permite a los usuarios cargar imágenes y obtener predicciones precisas en tiempo real.

## ¿Cómo funciona?
- La aplicación utiliza una **CNN entrenada** para predecir enfermedades en hojas de tomate.
- Los usuarios pueden cargar una imagen a través de la interfaz, y el modelo analizará la imagen para proporcionar una **predicción** basada en las características visuales aprendidas durante el entrenamiento.
- El resultado incluye la clase predicha y un grado de **confianza** en la predicción, así como información sobre las causas, síntomas, tratamientos y comentarios de la enfermedad detectada.

## ¿Cómo utilizar la aplicación?
1. **Carga una imagen**: Usa la pestaña _Prueba del modelo_ para seleccionar una imagen.
2. **Procesamiento**: Una vez cargada, la imagen será procesada por el modelo CNN en segundo plano.
3. **Predicción**: La aplicación mostrará la predicción devuelta por el modelo, junto a su probabilidad asociada.
4. **Información**: La aplicación proporcionará información detallada sobre la enfermedad detectada.

## Sobre el Proyecto
Este trabajo se ha desarrollado como **Trabajo de Fin de Máster** para finalizar los estudios del Máster en Inteligencia Artificial de la Universidad Internacional de Valencia. Se presenta esta aplicación como apoyo a este trabajo y para poner a prueba el modelo mediante el uso externo de otros usuarios.

- Autor: Morad Charchaoui Oilad Ali
- Fecha última actualización: 05/10/2024
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
            pred, prob = model_prediccion(imagen)
            st.success(f"El modelo ha predicho que se trata de {class_names[pred]} con una confianza del {prob:.2f}%")
            if class_names[pred] != 'Tomato___healthy':
                filtro = enfermedades[enfermedades['Enfermedad'] == class_names[pred]]
                st.markdown("## Síntomas")
                st.write(filtro["Síntomas"].values[0])
                st.markdown("## Causas")
                st.write(filtro["Causas"].values[0])
                st.markdown("## Tratamiento")
                st.write(filtro["Tratamiento"].values[0])
                st.markdown("## Comentarios")
                st.write(filtro["Comentarios"].values[0])
                st.markdown("""
## [+ Info]
[PlantVillage - Tomato](https://plantvillage.psu.edu/topics/tomato/infos)
""")

# Página de contacto
elif app_mode == "Contacto":
    st.title("Contacto")
    # Crear columnas para separar secciones
    col1, col2 = st.columns([1, 4])

    # st.markdown("<hr style='border: 2px solid #00913f;'>", unsafe_allow_html=True)
    # st.markdown("---")

    col1, col2 = st.columns([1, 4])

    # Logo de correo electrónico
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg", width=40)

    with col2:
        st.markdown("### [Correo electrónico](mailto:morad11jr@gmail.com)", unsafe_allow_html=True)  # Enlace actualizado o marcador


    st.markdown("Si tienes alguna pregunta, no dudes en ponerte en contacto. 😊")

    st.markdown("<hr style='border: 2px solid #00913f;'>", unsafe_allow_html=True)
    st.markdown("Almería, España")

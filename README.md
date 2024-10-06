# Resumen del proyecto

La agricultura de precisión, impulsada por tecnologías emergentes como la inteligencia artificial (IA), ha transformado la forma en que los agricultores gestionan sus cultivos, permitiéndoles tomar decisiones más informadas y optimizar el uso de recursos como el agua, los fertilizantes y la tierra. Este Trabajo de Fin de Máster se enmarca en este contexto, proponiendo una solución basada en redes neuronales convolucionales (CNN) para la identificación automática de enfermedades en hojas de tomate.


El modelo desarrollado utiliza MobileNet, una arquitectura ligera y eficiente, ideal para su implementación en dispositivos móviles, permitiendo su uso en campo con una alta precisión del $95.63$\%. El entrenamiento del modelo se llevó a cabo en Google Colaboratory, aprovechando técnicas avanzadas de aumento de datos para mejorar la capacidad de generalización del modelo, especialmente en clases con menos representaciones. Además, se utilizó Keras Tuner para optimizar los hiperparámetros clave, como la tasa de aprendizaje y el tamaño del lote, lo que resultó en un rendimiento óptimo durante el proceso de entrenamiento.


A lo largo del desarrollo, se observaron algunos desafíos en la clasificación de clases minoritarias, lo que subraya la necesidad de un balance adecuado de los datos. El modelo fue evaluado utilizando herramientas como TensorBoard, que permitieron monitorear de forma detallada la evolución del error y la precisión durante las distintas fases del entrenamiento.


El modelo final fue desplegado en una aplicación web interactiva a través de Streamlit, proporcionando una solución práctica para los agricultores. Esta herramienta permite cargar imágenes de hojas de tomate y obtener diagnósticos en tiempo real sobre posibles enfermedades, facilitando así la toma de decisiones rápidas y efectivas en la gestión de cultivos.


Este trabajo demuestra la efectividad de las redes neuronales en la agricultura de precisión y ofrece una solución escalable y accesible que puede mejorar significativamente la detección temprana de enfermedades, optimizando los recursos y reduciendo las pérdidas en la producción agrícola.

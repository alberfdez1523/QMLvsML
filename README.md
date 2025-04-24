QMLvsML: Comparativa entre Quantum Machine Learning y Machine Learning Clásico
Este repositorio contiene una colección de casos prácticos que comparan el rendimiento entre algoritmos de Machine Learning Cuántico (QML) y Machine Learning Clásico (ML) en diferentes dominios y aplicaciones.

Estructura del Repositorio
El repositorio está organizado en varias carpetas principales, cada una enfocada en un tipo específico de aplicación:

1. Caminos
Rutas: Comparación de algoritmos para encontrar rutas óptimas en grafos.
quantum.txt: Resultados de ejecución de algoritmos de optimización de rutas.
2. Galaxy Detection using Quantum Machine Learning
Clasificación de imágenes de galaxias utilizando algoritmos de ML clásico y cuántico.

Galaxy_Detection_using_QNN.ipynb: Notebook principal del proyecto.
ScriptQNN.py: Implementación del modelo QNN.
ScriptClassic.py: Implementación del modelo clásico.
ImageRead.py: Utilidades para lectura de imágenes.
Datasets y Modelos: Incluye conjuntos de datos y modelos entrenados para:
🌌 Galaxias
🦁 Animales
👆 Dedos
🔢 Números
🚢 Barcos
3. Híbridos
Modelos híbridos que combinan componentes cuánticos y clásicos para problemas de regresión.

model_celsius_4/model_celsius_6: Modelos para conversión de temperatura (Fahrenheit a Celsius).
model_electricity_4/model_electricity_6: Modelos para predicción de consumo eléctrico.
model_milllas_4/model_milllas_6: Modelos para conversión de kilómetros a millas.
lightyear_ibm: Modelo híbrido implementado usando IBM Quantum.
4. RegresionMultivalual
Implementación de modelos de regresión multivaluados.

Características Principales
🧠 Algoritmos Implementados
Modelos Clásicos: Redes neuronales, MLPs, y otros algoritmos clásicos.
Modelos Cuánticos: Utilizando frameworks como PennyLane para implementar QNNs (Quantum Neural Networks).
Modelos Híbridos: Combinación de capas clásicas y cuánticas.
📊 Métricas de Comparación
⏱️ Tiempo de entrenamiento y ejecución
🎯 Precisión de los modelos
🛣️ Longitud y calidad de los caminos encontrados (para problemas de rutas)
📉 Pérdida (loss) durante el entrenamiento

Requisitos
Python 3.7+
PyTorch
Numpy
Matplotlib
PennyLane (para modelos cuánticos)
Scikit-learn
Seaborn (para visualizaciones)
Cómo utilizar
Clona el repositorio:

Explora los diferentes directorios para ejecutar los casos de uso específicos.

Contacto
Para más información o colaboraciones, no dudes en abrir un issue en este repositorio.

Este repositorio forma parte de un estudio comparativo entre técnicas de Machine Learning Clásico y Cuántico para diversos problemas prácticos.
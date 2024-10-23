# Importar las bibliotecas necesarias
import tensorflow as tf # Para construir y entrenar modelos de aprendizaje automático
import matplotlib.pyplot as plt # Para la visualización de datos
import seaborn as sn # Para visualizaciones estadísticas
import numpy as np # Para álgebra lineal y manipulación de matrices
import pandas as pd # Para procesamiento de datos, principalmente CSV
import math # Para funciones matemáticas
import datetime # Para trabajar con fechas y horas
import platform # Para obtener información sobre el sistema operativo
from sklearn.model_selection import train_test_split # Para dividir los datos en conjuntos de entrenamiento y validación
from sklearn.manifold import TSNE # Para reducción de dimensionalidad, útil para visualización

# Cargar los datos de entrenamiento y prueba desde archivos CSV
train = pd.read_csv('train.csv') # Cargar datos de entrenamiento
test = pd.read_csv('test.csv') # Cargar datos de prueba

# Preparar las características (X) y las etiquetas (y) desde el conjunto de entrenamiento
X = train.iloc[:, 1:785] # Obtener las columnas que representan las imágenes (28x28 = 784 píxeles)
y = train.iloc[:, 0] # Obtener la columna que representa el dígito (la etiqueta)
X_test = test.iloc[:, 0:784] # Obtener las columnas del conjunto de prueba (784 píxeles)

# (Código comentado que imprime la forma de los conjuntos de datos)

# Dividir los datos en conjuntos de entrenamiento y validación (80% entrenamiento, 20% validación)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state = 1212)

# (Código comentado que imprime la forma de los conjuntos de entrenamiento y validación)

# Reestructurar los datos para que sean adecuados para la entrada del modelo
x_train_re = X_train.to_numpy().reshape(33600, 28, 28) # Convertir X_train a un arreglo y redimensionarlo
y_train_re = y_train.values # Obtener las etiquetas como un arreglo
x_validation_re = X_validation.to_numpy().reshape(8400, 28, 28) # Hacer lo mismo para el conjunto de validación
y_validation_re = y_validation.values # Obtener las etiquetas de validación
x_test_re = test.to_numpy().reshape(28000, 28, 28) # Hacer lo mismo para el conjunto de prueba

# Guardar los parámetros de las imágenes en constantes que se usarán más tarde
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train_re.shape # Obtener las dimensiones de las imágenes
IMAGE_CHANNELS = 1 # Número de canales en la imagen (grayscale)

# Imprimir dimensiones de las imágenes
print('IMAGE_WIDTH:', IMAGE_WIDTH)
print('IMAGE_HEIGHT:', IMAGE_HEIGHT)
print('IMAGE_CHANNELS:', IMAGE_CHANNELS)

# Mostrar una de las imágenes de entrenamiento
pd.DataFrame(x_train_re[0]) # Crear un DataFrame para una visualización más legible (opcional)

# Mostrar la primera imagen del conjunto de entrenamiento
plt.imshow(x_train_re[0], cmap=plt.cm.binary) # Mostrar la imagen en grayscale
plt.show()

# Visualizar más ejemplos de los datos de entrenamiento
numbers_to_display = 25 # Número de ejemplos a mostrar
num_cells = math.ceil(math.sqrt(numbers_to_display)) # Calcular la cantidad de celdas necesarias para mostrar los ejemplos
plt.figure(figsize=(10,10)) # Configurar el tamaño de la figura
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i+1) # Crear un subplot para cada imagen
    plt.xticks([]) # Quitar las marcas del eje x
    plt.yticks([]) # Quitar las marcas del eje y
    plt.grid(False) # Quitar la cuadrícula
    plt.imshow(x_train_re[i], cmap=plt.cm.binary) # Mostrar la imagen
    plt.xlabel(y_train_re[i]) # Etiquetar la imagen con su dígito correspondiente
plt.show() # Mostrar todas las imágenes


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Mostrar la imagen de la universidad
st.image("https://github.com/lRevenz/prediccion-acero-/blob/main/logo%20URP.jpg", caption="Proyecto realizado por estudiantes de la Universidad Ricardo Palma", use_column_width=True)

# Solicitar el nombre del usuario
nombre_usuario = st.text_input("¿Cuál es tu nombre?", "")

# Mostrar el mensaje sobre el proyecto
if nombre_usuario:
    st.write(f"¡Hola {nombre_usuario}! Este es un proyecto realizado por estudiantes de la Universidad Ricardo Palma.")

# Generación de datos ficticios (usados para entrenar el modelo)
np.random.seed(42)

fy = np.random.uniform(2800, 4200, size=100)  # Resistencia del acero (kg/cm²)
fc = np.random.uniform(200, 400, size=100)  # Resistencia del concreto (kg/cm²)
b = np.random.uniform(20, 40, size=100)  # Ancho de la sección (cm)
d = np.random.uniform(30, 60, size=100)  # Altura útil de la sección (cm)
Mu = np.random.uniform(5000, 10000, size=100)  # Momento flector (kg·cm)

# Fórmula simplificada para calcular el área de acero (As)
As = (Mu * fy) / (fc * b * d)  # Relación simplificada para obtener As

# Crear un DataFrame con los datos generados
data = pd.DataFrame({
    'fy': fy,
    'fc': fc,
    'b': b,
    'd': d,
    'Mu': Mu,
    'As': As  # Área de acero (salida)
})

# Definir las características (X) y la salida (y)
X = data[['fy', 'fc', 'b', 'd', 'Mu']]  # Entradas
y = data['As']  # Área de acero (salida)

# Entrenamiento del modelo
model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X, y)

# Entrada de datos por parte del usuario
st.write("Introduce los valores para calcular el área de acero:")

fy_input = st.number_input('Resistencia del acero (fy) en kg/cm²', min_value=0, max_value=5000, value=3500)
fc_input = st.number_input("Resistencia del concreto (f'c) en kg/cm²", min_value=0, max_value=500, value=300)
b_input = st.number_input('Ancho de la sección (b) en cm', min_value=0, max_value=100, value=30)
d_input = st.number_input('Altura útil de la sección (d) en cm', min_value=0, max_value=100, value=50)
Mu_input = st.number_input('Momento flector (Mu) en kg·cm', min_value=0, max_value=10000, value=7000)

# Botón para realizar la predicción
if st.button('Calcular Área de Acero'):
    nuevo_parametro = np.array([[fy_input, fc_input, b_input, d_input, Mu_input]])
    prediccion = model.predict(nuevo_parametro)
    
    # Mostrar la predicción del área de acero
    st.write(f"Área de acero predicha (As): {prediccion[0]:.2f} cm²")
    
    # Recomendación de tipo de acero basado en el área de acero
    if prediccion[0] < 100:
        st.write("Te recomendamos usar acero de 3/8.")
    elif prediccion[0] < 200:
        st.write("Te recomendamos usar acero de 1/2.")
    elif prediccion[0] < 300:
        st.write("Te recomendamos usar acero de 5/8.")
    else:
        st.write("Te recomendamos usar acero de 3/4.")

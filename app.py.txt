import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Generación de datos ficticios
np.random.seed(42)

# Generar datos aleatorios para fy, f'c, b, d, Mu
n_samples = 100  # Número de muestras
fy = np.random.uniform(2800, 4200, size=n_samples)  # Resistencia del acero (kg/cm²)
fc = np.random.uniform(200, 400, size=n_samples)  # Resistencia del concreto (kg/cm²)
b = np.random.uniform(20, 40, size=n_samples)  # Ancho de la sección (cm)
d = np.random.uniform(30, 60, size=n_samples)  # Altura útil de la sección (cm)
Mu = np.random.uniform(5000, 10000, size=n_samples)  # Momento flector (kg·cm)

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

# Entrenamiento de la red neuronal
model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X, y)

# Crear la interfaz con Streamlit
st.title('Predicción del Área de Acero por Flexión')
st.write("Ingrese los valores de los parámetros para predecir el área de acero:")

# Entrada de datos por parte del usuario
fy_input = st.number_input('Resistencia del acero (fy) en kg/cm²', min_value=0, max_value=5000, value=3500)
fc_input = st.number_input("Resistencia del concreto (f'c) en kg/cm²", min_value=0, max_value=500, value=300)
b_input = st.number_input('Ancho de la sección (b) en cm', min_value=0, max_value=100, value=30)
d_input = st.number_input('Altura útil de la sección (d) en cm', min_value=0, max_value=100, value=50)
Mu_input = st.number_input('Momento flector (Mu) en kg·cm', min_value=0, max_value=10000, value=7000)

# Realizar la predicción
if st.button('Predecir Área de Acero'):
    nuevo_parametro = np.array([[fy_input, fc_input, b_input, d_input, Mu_input]])
    prediccion = model.predict(nuevo_parametro)
    
    # Mostrar la predicción del área de acero
    st.write(f"Área de acero predicha (As): {prediccion[0]:.2f} cm²")

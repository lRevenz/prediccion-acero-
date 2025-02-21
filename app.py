import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Título y descripción del proyecto
st.title("Proyecto de Cálculo de Acero de Refuerzo")
st.image("https://github.com/lRevenz/prediccion-acero-/blob/main/logo%20URP.jpg?raw=true", caption="Proyecto realizado por estudiantes de la Universidad Ricardo Palma", use_container_width=True)

# Creación de pestañas
tab = st.radio("Selecciona una pestaña:", ["Inicio", "Cálculo de Acero", "Procedimiento"])

# Pestaña de "Inicio"
if tab == "Inicio":
    st.subheader("Bienvenido al proyecto")
    st.write("Este proyecto ha sido desarrollado por estudiantes de la Universidad Ricardo Palma.")
    st.write("Aquí podrás calcular el área de acero de refuerzo de una zapata, según los parámetros que ingreses.")

# Pestaña de "Cálculo de Acero"
elif tab == "Cálculo de Acero":
    st.subheader("Introduce los parámetros para calcular el área de acero:")

    # Solicitar los parámetros del usuario
    fy_input = st.number_input('Resistencia del acero (fy) en kg/cm²', min_value=0, max_value=5000, value=3500)
    fc_input = st.number_input("Resistencia del concreto (f'c) en kg/cm²", min_value=0, max_value=500, value=300)
    b_input = st.number_input('Ancho de la sección (b) en cm', min_value=0, max_value=100, value=30)
    d_input = st.number_input('Altura útil de la sección (d) en cm', min_value=0, max_value=100, value=50)
    Mu_input = st.number_input('Momento flector (Mu) en kg·cm', min_value=0, max_value=10000, value=7000)

    # Botón para realizar la predicción
    if st.button('Calcular Área de Acero'):
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

        # Calcular la predicción
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

# Pestaña de "Procedimiento"
elif tab == "Procedimiento":
    st.subheader("Procedimiento para calcular el área de acero")
    st.write("""
    1. **Definición de Parámetros**: Se utilizan los siguientes parámetros para el cálculo:
       - `fy`: Resistencia del acero.
       - `f'c`: Resistencia del concreto.
       - `b`: Ancho de la sección.
       - `d`: Altura útil de la sección.
       - `Mu`: Momento flector.
    
    2. **Cálculo del Área de Acero**: El área de acero se calcula utilizando una fórmula simplificada que relaciona el momento flector, la resistencia del acero y el concreto, el ancho de la sección y la altura útil.

    3. **Recomendación del Tipo de Acero**: Según el área de acero calculada, se recomienda usar un tipo de acero específico (3/8, 1/2, 5/8, 3/4) en función de las necesidades del proyecto.
    """)


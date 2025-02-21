import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Título y descripción del proyecto
st.title("Proyecto de Cálculo de Acero de Refuerzo")
st.image("https://github.com/lRevenz/prediccion-acero-/blob/main/logo%20URP.jpg?raw=true", caption="Proyecto realizado por estudiantes de la Universidad Ricardo Palma", use_container_width=True)

# Manejo del flujo entre pasos
if "step" not in st.session_state:
    st.session_state.step = 1  # Paso inicial

# Paso 1: Ingresar el nombre del usuario
if st.session_state.step == 1:
    st.subheader("Paso 1: Ingrese su nombre")
    nombre_usuario = st.text_input("¿Cuál es tu nombre?", "")
    
    if nombre_usuario:
        st.session_state.nombre_usuario = nombre_usuario
        st.session_state.step = 2  # Mover a la siguiente etapa cuando el nombre se ingresa

# Paso 2: Ingresar los datos para el cálculo del acero
if st.session_state.step == 2:
    st.subheader("Paso 2: Introduzca los parámetros para calcular el área de acero")
    
    # Crear un formulario para que el usuario ingrese los datos
    with st.form(key="input_form"):
        fy_input = st.number_input('Resistencia del acero (fy) en kg/cm²', min_value=0, max_value=5000, value=3500)
        fc_input = st.number_input("Resistencia del concreto (f'c) en kg/cm²", min_value=0, max_value=500, value=300)
        b_input = st.number_input('Ancho de la sección (b) en cm', min_value=0, max_value=100, value=30)
        d_input = st.number_input('Altura útil de la sección (d) en cm', min_value=0, max_value=100, value=50)
        Mu_input = st.number_input('Momento flector (Mu) en kg·cm', min_value=0, max_value=10000, value=7000)

        submit_button = st.form_submit_button(label="Calcular Área de Acero")
    
    if submit_button:
        # Realizar el cálculo cuando el usuario presiona el botón de "Calcular"
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
        
        # Guardar el resultado en el estado de la sesión
        st.session_state.prediccion = prediccion[0]
        st.session_state.step = 3  # Mover al siguiente paso

# Paso 3: Mostrar el procedimiento y la respuesta
if st.session_state.step == 3:
    st.subheader("Paso 3: Procedimiento y resultados")
    
    # Procedimiento de cálculo
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

    # Mostrar el área de acero calculada y la recomendación
    st.write(f"Área de acero predicha (As): {st.session_state.prediccion:.2f} cm²")

    if st.session_state.prediccion < 100:
        st.write("Te recomendamos usar acero de 3/8.")
    elif st.session_state.prediccion < 200:
        st.write("Te recomendamos usar acero de 1/2.")
    elif st.session_state.prediccion < 300:
        st.write("Te recomendamos usar acero de 5/8.")
    else:
        st.write("Te recomendamos usar acero de 3/4.")

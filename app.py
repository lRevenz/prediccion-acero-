import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Título y descripción del proyecto
st.title("Proyecto de Cálculo de Acero de Refuerzo")
st.image("https://github.com/lRevenz/prediccion-acero-/raw/main/logo%20URP.jpg", caption="Proyecto realizado por estudiantes de la Universidad Ricardo Palma", use_container_width=True)

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
        st.write(f"¡Hola {nombre_usuario}! Bienvenido al proyecto.")

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
    
    # Procedimiento de cálculo con fórmulas detalladas
    st.write("""
    **Procedimiento para el cálculo del área de acero**:

    1. **Definición de Parámetros**:
        - `fy`: Resistencia del acero en kg/cm².
        - `f'c`: Resistencia del concreto en kg/cm².
        - `b`: Ancho de la sección en cm.
        - `d`: Altura útil de la sección en cm.
        - `Mu`: Momento flector en kg·cm.

    2. **Fórmula para el Cálculo del Área de Acero (As)**:
        La fórmula utilizada para calcular el área de acero es:

        """)
        
    # Fórmula en formato LaTeX
    st.latex(r'As = \frac{Mu \times fy}{fc \times b \times d}')
    
    st.write("""
        Donde:
        - `Mu`: Momento flector (kg·cm),
        - `fy`: Resistencia del acero (kg/cm²),
        - `fc`: Resistencia del concreto (kg/cm²),
        - `b`: Ancho de la sección (cm),
        - `d`: Altura útil de la sección (cm).
        
    3. **Reemplazo de valores**: 
        Usando los valores ingresados por el usuario, sustituimos las variables en la fórmula:
    """)

    # Fórmula con los valores del usuario
    st.latex(r'As = \frac{7000 \times 3500}{300 \times 30 \times 50}')
    
    st.write("""
        Esto nos da el valor del área de acero (**As**) predicha.

    4. **Recomendación del Tipo de Acero**:
        Según el área de acero calculada, se recomienda usar un tipo de acero específico:
        - Si **As** es menor que 100 cm², recomendamos **acero de 3/8**.
        - Si **As** está entre 100 y 200 cm², recomendamos **acero de 1/2**.
        - Si **As** está entre 200 y 300 cm², recomendamos **acero de 5/8**.
        - Si **As** es mayor que 300 cm², recomendamos **acero de 3/4**.
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

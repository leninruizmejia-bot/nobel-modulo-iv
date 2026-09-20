import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


st.write("# Predicción de categoría de Premio Nobel")


st.header("Texto")


def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    palabras = texto.split()
    return ' '.join(palabras)


def user_input_features():
    texto = st.text_input("Introduce el texto a evaluar")

    user_input_data = {'Text': texto}

    features = pd.DataFrame(user_input_data, index=[0])

    return features


df = user_input_features()


# Cargar dataset consolidado
nobel = pd.read_csv(
    'nobel_consolidado.csv'
)


# Preparar texto
nobel['Motivation_clean'] = nobel['Motivation'].apply(limpiar_texto)


X = nobel['Motivation_clean']
y = nobel['Category']


# Vectorización TF-IDF
vect = TfidfVectorizer()

X_dtm = vect.fit_transform(X)


# Entrenamiento del modelo
modelo = LogisticRegression(max_iter=1000)

modelo.fit(X_dtm, y)


# Preparar texto introducido por el usuario
texto_usuario = limpiar_texto(df['Text'].iloc[0])

df_dtm = vect.transform([texto_usuario])


# Realizar predicción
prediction = modelo.predict(df_dtm)


st.subheader("Predicción")


if df['Text'].iloc[0].strip() == "":
    st.write("Introduce un texto para realizar la predicción.")
else:
    st.write(prediction[0])
#Imports
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('indicadores_statusinvest.csv')
df = df.set_index('TICKER')
modelo = load_model('modelo.h5')

st.set_page_config(
     page_title="Keras - Ferramenta de investimento",
     page_icon="💸",
     layout="wide",
     initial_sidebar_state="expanded",
 )

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    image = Image.open('transferir.png')
    st.image(image, caption='Invista com inteligência. Invista com Keras')

with col3:
    st.write(' ')

st.title('Seu parceiro de investimentos inteligentes')



def ajustar_scaler(tabela_original):
    scaler = StandardScaler()
    tabela_auxiliar = tabela_original

    tabela_auxiliar = pd.DataFrame(scaler.fit_transform(tabela_auxiliar), tabela_auxiliar.index,
                                   tabela_auxiliar.columns)
    return tabela_auxiliar

with st.form("my_form"):
    st.write("Preencha o formulário")
    empresa = st.text_input('Digite o ticker da sua empresa:').upper()
    submitted = st.form_submit_button("Submit")

    if submitted:
        escolha = df.loc[empresa]
        st.write('Os fundamentos da empresa escolhida são:')
        fundamentos = escolha.T
        st.write(fundamentos)
        dataframe = ajustar_scaler(df)
        escolha = dataframe.loc[empresa]
        escolha = pd.DataFrame(escolha).T
        x_teste = np.array(escolha)
        prev = modelo.predict(x_teste)
        df = pd.DataFrame(prev, columns=['Não compra', 'Compra'])
        st.write('Decisão da I.A.:')
        st.write(df)

st.write('Feito por:Thiago Pedroso, Pedro Fernandes, Alberto Lucas, Davi de Jesus')


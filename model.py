#Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras

dataset = pd.read_csv('base_dados_ano.csv')

dataset = dataset.drop(columns=['Date', 'Adj Close', 'IBOV', 'Resultado', 'Empresa', 'LC', 'P/FCO'])

def ajustar_scaler(tabela_original):
    scaler = StandardScaler()
    tabela_auxiliar = tabela_original.drop("Decisao", axis=1)

    tabela_auxiliar = pd.DataFrame(scaler.fit_transform(tabela_auxiliar), tabela_auxiliar.index,
                                   tabela_auxiliar.columns)
    tabela_auxiliar["Decisao"] = tabela_original["Decisao"]
    return tabela_auxiliar


nova_base_dados = ajustar_scaler(dataset)

x_treino = nova_base_dados.drop("Decisao", axis=1)
y_treino = nova_base_dados["Decisao"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x_treino, y_treino, random_state=1)

x_treino =  np.array(x_treino)
y_treino = np.array(y_treino)
x_teste = np.array(x_teste)
y_teste = np.array(y_teste)

#callback = tf.keras.callbacks.EarlyStopping(patience=2)

modelo = keras.Sequential([
    keras.layers.Dense(units = 10, activation = "relu"), #Entrada
    keras.layers.Dense(units = 18, activation = "sigmoid"),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(units = 32, activation = "relu"), #Processamento
    #keras.layers.Dropout(0.1),
    #keras.layers.LeakyReLU(alpha=0.05),
    keras.layers.Dense(units = 2, activation = "softmax") #Saída
])

modelo.compile(optimizer = "Adam", loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

historico = modelo.fit(x_treino, y_treino, epochs = 180, validation_split = 0.2, shuffle=True)

modelo.save('modelo.h5')

print(x_teste)
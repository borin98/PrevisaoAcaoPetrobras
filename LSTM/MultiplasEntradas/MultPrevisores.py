import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler

def montaGraficos ( preco_real_teste, previsoes ) :

    plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
    plt.plot(previsoes, color = 'blue', label = 'Previsões')
    plt.title('Previsão preço das ações')
    plt.xlabel('Tempo')
    plt.ylabel('Valor Yahoo')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("GraficoPredicao.png")

def baseTeste ( dfTreinamento, normalizador ) :
    """
    Monta a base de teste
    normalizada

    :return:
    """
    dfTeste = pd.read_csv ( "petr4-teste.csv" )
    dfTeste = dfTeste.dropna (  )

    precoRealTeste = dfTeste.iloc [ :, 1:2 ].values

    frames = [ dfTreinamento, dfTeste ]

    dfCompleto = pd.concat ( frames )
    dfCompleto = dfCompleto.drop ( "Date", axis = 1 )
    #dfCompleto = dfCompleto.drop("Date",axis = 1)

    entradas = dfCompleto [ len ( dfCompleto ) - len ( dfTeste ) - 90 : ].values
    entradas = normalizador.transform ( entradas )

    xTeste = []

    for i in range ( 90, len ( entradas ) ) :

        xTeste.append ( entradas [ i - 90:i, 0:6 ] )

    xTeste = np.array ( xTeste )

    return xTeste, precoRealTeste

def MontaRede ( previsao, quantVariaveis = 1 ) :
    """
    Função que monta a rede LSTM
    que a retorna

    :return:
    """

    Lstm = Sequential ( )

    Lstm.add ( LSTM (
        units = 100,
        return_sequences = True,
        input_shape = ( ( previsao.shape[1], quantVariaveis ) )
    ) )
    Lstm.add ( Dropout ( 0.3 ) )

    Lstm.add ( LSTM (
        units = 50,
        return_sequences = True
    ) )
    Lstm.add ( Dropout ( 0.3 ) )

    Lstm.add ( LSTM (
        units = 50,
        return_sequences = True
    ) )
    Lstm.add ( Dropout ( 0.3 ) )

    Lstm.add ( LSTM (
        units = 50
    ) )
    Lstm.add ( Dropout ( 0.3 ) )

    Lstm.add ( Dense (
        units = 1,
        activation = "sigmoid"
    ) )

    Lstm.compile (
        optimizer = "adam",
        loss = "mean_squared_error",
        metrics = ["mae"]
    )

    Es = EarlyStopping (
        monitor = "loss",
        min_delta = 1e-10,
        patience = 10,
        verbose = 1
    )

    Rlr = ReduceLROnPlateau (
        monitor = "loss",
        factor = 0.2,
        patience = 5,

    )

    Mcp = ModelCheckpoint (
        filepath = "pesos.h5",
        monitor = "loss",
        save_best_only = True
    )

    return Lstm, Es, Rlr, Mcp

def basesTreinamento ( df ) :
    """
    Função que cria a bases
    de treinamento normalizado

    :param df:
    :return:
    """

    dfTreinamento = df.iloc [ :, 1:7 ].values

    valor = dfTreinamento

    normalizador = MinMaxScaler (
        feature_range = ( 0, 1 )
    )

    dfTreinamento = normalizador.fit_transform ( dfTreinamento )

    precoPrevisao, precoReal = [], []

    for i in range(90, len(dfTreinamento)):
        precoPrevisao.append(dfTreinamento[i - 90: i, 0:6])
        precoReal.append(dfTreinamento[i, 0])


    precoReal, precoPrevisao = np.array(precoReal), np.array(precoPrevisao)

    return precoReal, precoPrevisao, len(dfTreinamento[0][:]), normalizador, valor

def main (  ) :

    """

    Programa que usa dois parâmetros de entrada para treinamento
    e faz a predição do preço da ação de entrada de uma ação da
    petrobrás

    :return:
    """

    df = pd.read_csv ( "petr4-treinamento.csv" )
    df = df.dropna()

    normalizadorTeste = MinMaxScaler ( feature_range = ( 0, 1 ) )

    baseTreinamento, previsorTeste, quantVariaveis, normalizador, dfTreinamento = basesTreinamento ( df )

    dfTeste, precoRealTeste = baseTeste (
        dfTreinamento = df,
        normalizador = normalizador
    )

    normalizadorTeste.fit_transform ( dfTreinamento [ :, 0:1 ] )

    Lstm, Es, Rlr, Mcp = MontaRede ( previsorTeste, quantVariaveis = quantVariaveis )

    Lstm.fit (
        previsorTeste,
        baseTreinamento,
        epochs = 1000,
        callbacks = [ Es, Rlr, Mcp ]
    )

    resultado = Lstm.predict ( dfTeste )
    resultado = normalizadorTeste.inverse_transform ( resultado )

    montaGraficos (
        preco_real_teste = precoRealTeste,
        previsoes = resultado
    )

main()
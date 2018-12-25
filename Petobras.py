import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

def montaGraficos ( preco_real_teste, previsoes ) :

    plt.figure(0)
    plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
    plt.plot(previsoes, color = 'blue', label = 'Previsões')
    plt.title('Previsão preço das ações')
    plt.xlabel('Tempo')
    plt.ylabel('Valor Yahoo')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("GraficoPredicao.png")

def montaRedeNeural ( previsao, quantVariaveis ) :
    """
    Função que monta o modelo de rede
    neural para regressão

    """

    Net = Sequential ( )

    Net.add(LSTM(
        units = 100,
        return_sequences = True,
        input_shape = ( previsao.shape[1], 1 )
    ) )

    Net.add ( Dropout ( 0.3 ) )

    Net.add( LSTM (
        units = 50,
        return_sequences = True
    ) )

    Net.add ( Dropout ( 0.3 ) )

    Net.add( LSTM (
        units = 50,
        return_sequences = True
    ) )

    Net.add ( Dropout ( 0.3 ) )

    Net.add( LSTM (
        units = 50
    ) )

    Net.add ( Dropout ( 0.3 ) )

    Net.add ( Dense (
        units = quantVariaveis,
        activation = "linear"
    ) ) # podemos utilizar a sigmoide, caso os dados estejam normalizados

    Net.compile(
        optimizer = "rmsprop",
        loss = "mean_squared_error",
        metrics = ["mean_absolute_error", "acc"]
    )

    return Net

def preProcessingDataframe (  ) :
    """
    Função que faz o pre-procemento
    do dataframe de treinamento e teste , retirando os valores nulos e normalizanod
    seus valores

    """

    df = pd.read_csv ( "petr4-treinamento.csv" )
    df = df.dropna()

    dfTeste = pd.read_csv ( "petr4-teste.csv" )

    # separando os valores para predição em um numpy array
    dfTreinamento = df.iloc[ :, 1:2 ].values
    precoRealTeste = dfTeste.iloc[:, 1:2].values

    dfCompleto = pd.concat ( ( df["Open"], dfTeste["Open"] ), axis = 0 )

    entradaTeste = dfCompleto[ len(dfCompleto) - len(dfTeste) - 90 : ].values
    entradaTeste = entradaTeste.reshape ( -1, 1 )   # transpondo o vetor para nx1

    # normalizando os dados de treinamento
    normalizador = MinMaxScaler ( feature_range = (0, 1) )
    dfTreinamento = normalizador.fit_transform ( dfTreinamento )
    entradaTeste = normalizador.transform ( entradaTeste )

    # iremos fazer a previsão baseado
    # nos 90 dias anteriores
    precoPrevisao, precoReal, previsaoTeste = [], [], []

    for i in range ( 90, len ( dfTreinamento ) ) :

        precoPrevisao.append ( dfTreinamento [ i - 90 : i, 0 ] )
        precoReal.append ( dfTreinamento [ i, 0 ] )

    for i in range ( 90, len ( entradaTeste ) ) :

        previsaoTeste.append ( entradaTeste[ i - 90 : i, 0 ] )

    precoReal, precoPrevisao, previsaoTeste = np.array ( precoReal ), np.array ( precoPrevisao ), np.array ( previsaoTeste )

    # o último valor, é a quantidade de colunas que iremos utiizar para fazer a previsão dos dados
    precoPrevisao = np.reshape (
        precoPrevisao,
        ( precoPrevisao.shape[0],
        precoPrevisao.shape[1],
        len ( dfTreinamento[0][:] ) )
    )

    previsaoTeste = np.reshape (
        previsaoTeste,
        ( previsaoTeste.shape[0],
        previsaoTeste.shape[1],
        len ( dfTreinamento[0][:] )
    ) )

    return precoReal, precoPrevisao, previsaoTeste , precoRealTeste ,len ( dfTreinamento[0][:] ), normalizador

def main (  ) :

    precoRealTreinamento, precoPrevisaoTreinamento, precoPrevisaoTeste,precoRealPrevisao, quantVariaveis, normalizador = preProcessingDataframe()

    Net = montaRedeNeural ( precoPrevisaoTreinamento, quantVariaveis)
    Net.fit ( precoPrevisaoTreinamento, precoRealTreinamento, epochs = 100, batch_size = 32 )

    resultado = Net.predict ( precoPrevisaoTeste )
    resultado = normalizador.inverse_transform ( resultado )

    print(len ( precoPrevisaoTeste[0][:] ))
    print(len ( resultado[0][:] ))
    montaGraficos ( precoRealPrevisao, resultado )

if __name__ == '__main__':
    main()

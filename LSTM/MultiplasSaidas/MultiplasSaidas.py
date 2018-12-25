import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers  import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

def montaGraficos ( resultado, precoAbertura, precoAlta  ) :

    plt.plot ( precoAbertura, color = 'red', label = 'Preço Abertura real')
    plt.plot ( resultado[ :, 0], color = 'black', label = 'Previsões')
    plt.title('Previsão preço de abertura das ações')
    plt.xlabel('Tempo')
    plt.ylabel('Valor Yahoo')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("GraficoPredicaoAbertura.png")

    plt.plot(precoAlta, color='red', label='Preço alta real')
    plt.plot(resultado[:, 1], color='black', label='Previsões')
    plt.title('Previsão preço de alta das ações')
    plt.xlabel('Tempo')
    plt.ylabel('Valor Yahoo')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("GraficoPredicaoAlta.png")

    return

def preProcessingTestBase ( dfTreinamento, normalizador ) :

    dfTeste = pd.read_csv ( "petr4-teste.csv" )
    dfTeste = dfTeste.dropna()

    precoRealTesteAbertura = dfTeste.iloc [ :, 1:2 ].values # preço de entrada da bolsa

    precoRealTesteAlta = dfTeste.iloc [ :, 2:3 ].values  # preço de alta da ação

    dfCompleto = pd.concat ( ( dfTreinamento["Open"], dfTeste["Open"] ), axis = 0 )

    entradas = dfCompleto[len(dfCompleto) - len(dfTeste) - 90:].values
    entradas = np.reshape ( entradas, ( -1, 1 ) )   # transformando em um vetor unidimencional
    entradas = normalizador.transform(entradas)

    xTeste = []

    for i in range(90, len(entradas)):
        xTeste.append(entradas[i - 90:i, 0:6])

    xTeste = np.array(xTeste)
    xTeste = np.reshape (
        xTeste,
        (

            xTeste.shape[0],
            xTeste.shape[1],
            1

        )
    )

    return xTeste, precoRealTesteAbertura, precoRealTesteAlta

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
        units = 2,
        activation = "linear"
    ) )

    Lstm.compile (
        optimizer = "rmsprop",
        loss = "mean_squared_error",
        metrics = ["mae"]
    )

    return Lstm

def preProcessingTrainDatabase (  ) :

    df = pd.read_csv ( "petr4-treinamento.csv" )
    df = df.dropna()

    normalizadorTreinamento = MinMaxScaler ( feature_range = ( 0, 1 ) )

    dfTreinamento = df.iloc[:, 1:2].values # valores de abertura da bolsa

    dfValorMaximo = df.iloc[:, 2:3].values # valores de máximo da bolsa

    dfTreinamento = normalizadorTreinamento.fit_transform ( dfTreinamento )
    dfValorMaximo = normalizadorTreinamento.fit_transform ( dfValorMaximo )

    paramEntrada, precoAberturaReal, precoSaidaReal = [], [], []

    for i in range ( 90, len ( dfTreinamento ) ) :

        paramEntrada.append ( dfTreinamento [ i - 90: i, 0] )
        precoAberturaReal.append ( dfTreinamento [ i, 0 ] )
        precoSaidaReal.append ( dfValorMaximo [ i, 0 ] )

    paramEntrada, precoAberturaReal, precoSaidaReal = np.array ( paramEntrada ), np.array ( precoAberturaReal ), np.array ( precoSaidaReal )

    paramEntrada = np.reshape (
        paramEntrada,
        (

          paramEntrada.shape[0],
          paramEntrada.shape[1],
          1

        ) )

    precoRealConcat = np.column_stack ( ( precoAberturaReal, precoSaidaReal ) )

    return paramEntrada,precoRealConcat, normalizadorTreinamento, 1, df

def main (  ) :
    """

    Programa que usa apenas um parâmetro como entrada da rede
    e faz a projeção do preço de abertura e máximo de uma ação
    da petrobrás

    :return:
    """
    paramEntrada, precoRealConcat, normalizadorTreinamento, quantVariaveis, dfTreinamento = preProcessingTrainDatabase()

    TesteEntrada, precoRealAbertura, precoRealAlta = preProcessingTestBase ( dfTreinamento, normalizadorTreinamento )

    Lstm = MontaRede ( paramEntrada, quantVariaveis = quantVariaveis )
    Lstm.fit (
        paramEntrada,
        precoRealConcat,
        epochs = 1000,
        batch_size = 32
    )

    resultado = Lstm.predict ( TesteEntrada )
    resultado = normalizadorTreinamento.inverse_transform ( resultado )

    montaGraficos ( resultado, precoRealAbertura, precoRealAlta )

if __name__ == '__main__':
    main()
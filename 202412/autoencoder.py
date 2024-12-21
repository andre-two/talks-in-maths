import numpy as np
from tensorflow import keras 
from matplotlib import pyplot as plt
import pandas as pd


"""
**Metodos para treinar e calcular score de cross validação**

nao encontrei uma forma padrao de fazer cross validacao direto do keras, como existe do sklearn, entao criei uma funcao manual para isso.

é melhor criar um método apenas para o treinamento, pois é chamado várias vezes dentro da cross validacao e durante otimizacao de hiperparametros

**Atencao:** esses metodos foram escritos tendo em mente o problema de criacao de autoencoder. se for utilizar para outras redes neurais, algumas adaptacoes podem ser necessarias, e.g., se for criar uma rede para classificacao, no metodo de treinamento devemos mudar a loss estabelecida para a loss apropriada ( keras.losses.BinaryCrossentropy() ) e talvez acrescentar mais uma metrica para acompanhamento. No caso de acrescentar mais uma metrica, o metodo de treinamento retornara dois valores ao inves de um e é necessario uma adaptacao no outro metodo de cross validacao, para tratar essa duplicidade 
"""

def train_evaluate_model(model, X_train, y_train, X_test, y_test, nb_epoch, learn_rate, verbose) :
    optimizer = keras.optimizers.Adam(learning_rate=learn_rate)

    model.compile(optimizer = optimizer,
            loss = 'mean_squared_error',
            # metrics = ['mean_squared_error']
            )
    

    history = model.fit(X_train, y_train,
                    epochs = nb_epoch,
                    validation_data = (X_test, y_test),
                    verbose = verbose
                    ).history

    return model.evaluate(X_test, y_test, verbose = verbose), history


def cross_val_score_keras(model, X, y, cv = 5, nb_epoch = 50, learn_rate = 1e-3, verbose = 1):
    from sklearn.model_selection import KFold
    import numpy as np

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    cvscores = []

    i = 1
    for train, test in kfold.split(X,y):
        if verbose >0:
            print(format(f' Fitting fold {i:02d}/{cv:02d} ', '-^80'))

        model = model

        if verbose > 1:
            verbose_train = 1
        else:
            verbose_train = 0
        

        score, _ = train_evaluate_model(model, X[train], y[train], X[test], y[test], nb_epoch, learn_rate, verbose_train)

        cvscores.append(score)

        if verbose>0:
            print(f'Score {score:.4f}')

        i = i+1

    return np.asarray(cvscores)



"""
**Método para criar o autoencoder selecionando os seguintes parâmetros:**

qtd de neurônios de input;\
qtd de neurônios na camada latente;\
qtd de camadas entre a input e a latente;\
a forma de decaimento da qtd de neuronios entre o input e a camada latente:\
-- 'linear' cria uma PA entre a qtd de neuronios de input e a camada latente com a qtd de camadas intermediarias configurada\
-- 'geometric' cria uma PG entre a qtd de neuronios de input e a camada latente com a qtd de camadas intermediarias configurada

além disso também recebe parâmetros l1 e l2 para regularização dos pesos nas camadas intermediárias


A criacao desse método/funcao tornou-se necessario como uma forma de otimizar a topologia escolhida para o autoencoder
"""


def create_autoencoder(n_input, n_latent_dims, decay, n_hidden = 1, l1 = 1e-4, l2 = 1e-4):

    n_dims = 3 + 2*n_hidden
    array_dims = np.zeros(n_dims, dtype='int')
    array_dims[int((n_dims-1)/2)] = n_latent_dims
    array_dims[0] = n_input
    array_dims[-1] = n_input

    if n_hidden>0:
        if decay == 'linear':
            r = (n_latent_dims - n_input) / (n_hidden +1 )

            for i in range(n_hidden):
                array_dims[i+1] = int(n_input + (i+1)*r)
                array_dims[-1 - (i+1)] = int(n_input + (i+1)*r)

        elif decay == 'geometric':
            r = (n_latent_dims/n_input) ** (1/(n_hidden+1))
            
            for i in range(n_hidden):
                array_dims[i+1] = int(n_input * r **(i+1))
                array_dims[-1 - (i+1)] = int(n_input * r **(i+1))

        elif isinstance(decay, list):
            for i in range(n_hidden):
                array_dims[i+1] = decay[i]
                array_dims[-1 - (i+1)] = decay[i]


    for i in range(n_dims):
        if i==0:
            input_ = keras.layers.Input(shape=[array_dims[i]], name='input')
        elif i==1:
            x = keras.layers.Dense(array_dims[i], activation='tanh', kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2) )(input_)
        elif i==n_dims-1:
            output_ = keras.layers.Dense(array_dims[i], activation='relu', kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2), name='output')(x)
        else:
            x = keras.layers.Dense(array_dims[i], activation='tanh', kernel_regularizer=keras.regularizers.L1L2(l1=l1, l2=l2))(x)

    model = keras.models.Model(inputs = input_, outputs = output_)

    
    return model

# create_autoencoder(164, 6, 'geometric', 0)

def reconstruction_error(model, X):

    train_x_predictions = model.predict(X, verbose = 0)
    mse = np.mean(np.power(X - train_x_predictions, 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse})
    print(error_df.describe())

    # return error_df['Reconstruction_error'].mean(), error_df['Reconstruction_error'].median()
    return None




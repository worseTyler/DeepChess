from keras import models, layers, Model, Input, losses
import numpy as np
import random


def setup_data():
    bit_strings = np.load("./data/bit_strings_2.npy")
    labels = np.load("./data/labels_2.npy")
    white_win_indices = np.where(labels == 1)[0]
    black_win_indices = np.where(labels == 0)[0]

    white_wins = bit_strings[white_win_indices]
    black_wins = bit_strings[black_win_indices]
    
    print(bit_strings.shape)
    print(white_wins.shape)
    print(black_wins.shape)

    return white_wins, black_wins

def getDeepChessModel():
    DBN = models.load_model('./models/deepBeliefNetwork_relu_tanh')
    # DBN.compile(optimizer='adam', loss='mse')
    # DBN.summary()

    inputLayer1 = layers.Input(shape=(773,))
    inputLayer2 = layers.Input(shape=(773,))
    DBN1 = DBN(inputLayer1)
    DBN2 = DBN(inputLayer2)

    joinedDBN = layers.Concatenate()([DBN1, DBN2])
    layer_400 = layers.Dense(400, activation="relu")(joinedDBN)
    layer_200 = layers.Dense(200, activation="relu")(layer_400)
    layer_100 = layers.Dense(100, activation="relu")(layer_200)
    output = layers.Dense(2, activation="softmax")(layer_100)

    deepChess = Model(inputs=[inputLayer1, inputLayer2], outputs=output)
    deepChess.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics = ["accuracy"])
    
    deepChess.summary()

    return deepChess

white_wins, white_loses = setup_data()

def get_data(size):
    trainOne = []
    trainTwo = []
    label = []
    for i in range(size):
        white_win = random.choice(white_wins)
        white_loss = random.choice(white_loses)
        num = random.randint(0,1)
        if num == 0:
            trainOne.append(white_win)
            trainTwo.append(white_loss)
            label.append(np.array([1,0]))
        else: 
            trainOne.append(white_loss)
            trainTwo.append(white_win)
            label.append(np.array([0,1]))

    trainOne = np.array(trainOne)
    trainTwo = np.array(trainTwo)
    label = np.array(label)
    return trainOne, trainTwo, label

trainOne, trainTwo, label = get_data(1000000)
valOne, valTwo, labelVal = get_data(500000)

model = getDeepChessModel()
model.fit([trainOne, trainTwo], label, epochs=40, batch_size=50000, validation_data=([valOne, valTwo], labelVal))
model.save("./models/deepChess")
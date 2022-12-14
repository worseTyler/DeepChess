from keras import models, layers, Model, Input, losses, utils
import numpy as np
import random


def setup_data():
    white_wins = np.load("./data/white_wins.npy")
    white_losses = np.load("./data/white_losses.npy")
    
    np.random.shuffle(white_wins)
    np.random.shuffle(white_losses)

    print(white_wins.shape)
    print(white_losses.shape)

    return white_wins, white_losses

def getDeepChessModel():
    DBN = models.load_model('./models/deepBeliefNetwork_new_try')
    # DBN.compile(optimizer='adam', loss='mse')
    # DBN.summary()

    inputLayer1 = layers.Input(shape=(773,))
    inputLayer2 = layers.Input(shape=(773,))
    DBN1 = DBN(inputLayer1)
    DBN2 = DBN(inputLayer2)

    joinedDBN = layers.Concatenate()([DBN1, DBN2])
    layer_400 = layers.Dense(400, activation=layers.LeakyReLU(alpha=0.3))(joinedDBN)
    layer_200 = layers.Dense(200, activation=layers.LeakyReLU(alpha=0.3))(layer_400)
    layer_100 = layers.Dense(100, activation=layers.LeakyReLU(alpha=0.3))(layer_200)
    output = layers.Dense(2, activation="softmax")(layer_100)

    deepChess = Model(inputs=[inputLayer1, inputLayer2], outputs=output)
    deepChess.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics = ["accuracy"])
    
    deepChess.summary()

    return deepChess

white_wins, white_losses = setup_data()

class Generator(utils.Sequence):
    # https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    # This allows for me to change the training data combinations after every epoch
    
    def __init__(self, white_wins, white_losses, num_batches, batch_size):
        self.white_wins = white_wins
        self.white_losses = white_losses
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __getitem__(self, index):
        # Generates one batch of data
        trainOne = []
        trainTwo = []
        label = []
        for i in range(self.batch_size):
            white_win = random.choice(self.white_wins)
            white_loss = random.choice(self.white_losses)
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
        return [trainOne, trainTwo], label

    def __len__(self):
        # Number of Batches Per Epoch
        return self.num_batches

dataGenerator = Generator(white_wins, white_losses, 20, 50000)

model = getDeepChessModel()
model.fit(dataGenerator, epochs=30)
# validation_data=([valOne, valTwo], labelVal)
model.save("./models/deepChess_new_data", save_format="h5")
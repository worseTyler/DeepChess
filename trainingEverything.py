import tables as tb
from keras import layers, Model, Input, utils, losses, callbacks
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import time
import random

DBN_EPOCHS = 80
DBN_BATCH_SIZE = 5000
DBN_TRAIN_SIZE = 750000
DBN_VALIDATION_SIZE = 75000

DEEP_CHESS_EPOCHS = 250
DEEP_CHESS_NUM_BATCHES = 50
DEEP_CHESS_BATCH_SIZE = 50000

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

def create_autoencoder(input_size, latent_size, train, validation):
    input = Input(shape=(input_size,))
    encoded = layers.Dense(latent_size, activation="relu")(input)
    # tanh worked SIGNIFICANTLY better than sigmoid
    decoded = layers.Dense(input_size, activation='tanh')(encoded)

    autoencoder = Model(input, decoded)
    encoder = Model(input, encoded)

    csv_logger = callbacks.CSVLogger(f'./logs/DBN_{input_size}_{latent_size}.txt', "\n")
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(train, train, epochs=DBN_EPOCHS, batch_size=DBN_BATCH_SIZE, shuffle=True, validation_data=(validation, validation), callbacks=[csv_logger])
    return encoder

def generate_model(autoencoder_layers, training_data):
    encoders = []
    train, validation = train_test_split(training_data, train_size=DBN_TRAIN_SIZE, test_size=DBN_VALIDATION_SIZE, shuffle=True)

    encoder_train = train
    encoder_validation = validation
    for input_size, latent_size in autoencoder_layers:
        encoder = create_autoencoder(input_size, latent_size, encoder_train, encoder_validation)
        encoder.summary()
        encoders.append(encoder)

        # transform training data to match the latent space
        # so that it can be used to train the next encoder
        encoder_train = encoder.predict(encoder_train)
        encoder_validation = encoder.predict(encoder_validation)

    input = encoders[0].input
    output = encoders[0].output
    for encoder in encoders[1:]:
        output = encoder(output)

    DBN = Model(input, output)
    DBN.summary()
    DBN.save("./models/deepBeliefNetwork.h5", save_format="h5")
    return DBN

def getDeepChessModel(DBN):
    # DBN = models.load_model('./models/deepBeliefNetwork.h5')
    # DBN.summary()

    inputLayer1 = layers.Input(shape=(773,))
    inputLayer2 = layers.Input(shape=(773,))
    DBN1 = DBN(inputLayer1)
    DBN2 = DBN(inputLayer2)

    # Create Siamese Network through concatenation
    joinedDBN = layers.Concatenate()([DBN1, DBN2])

    # Add dense layers
    layer_400 = layers.Dense(400, activation=layers.LeakyReLU(alpha=0.3))(joinedDBN)
    layer_200 = layers.Dense(200, activation=layers.LeakyReLU(alpha=0.3))(layer_400)
    layer_100 = layers.Dense(100, activation=layers.LeakyReLU(alpha=0.3))(layer_200)
    output = layers.Dense(2, activation="softmax")(layer_100)

    deepChess = Model(inputs=[inputLayer1, inputLayer2], outputs=output)
    deepChess.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics = ["accuracy"])
    
    deepChess.summary()

    return deepChess

def load_data(win_amount, loss_amount):
    winTable = tb.open_file("./data/winTable.h5", mode='r')
    lossTable = tb.open_file("./data/lossTable.h5", mode='r')
    white_wins = winTable.root.bitStrings[:win_amount]
    white_losses = lossTable.root.bitStrings[:loss_amount]

    print(f"Win Shape: {white_wins.shape}")
    print(f"Loss Shape: {white_losses.shape}")
    return white_wins, white_losses



if __name__ == "__main__":

    start = time.perf_counter()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    ### Start Training Deep Belief Network
    start_dbn = time.perf_counter()
    # Load Data for DBN
    white_wins, white_losses = load_data((DBN_TRAIN_SIZE+DBN_VALIDATION_SIZE)/2, (DBN_TRAIN_SIZE+DBN_VALIDATION_SIZE)/2)
    
    # Put wins and losses into single collection of data
    training_data = np.append(white_wins, white_losses, axis=0)
    
    # Define input size and latent size for autoencoder layers
    autoencoder_layers = [(773,600), (600,400), (400,200), (200,100)]
    
    # Generate and train deep belief network
    DBN = generate_model(autoencoder_layers, training_data)
    
    end_dbn = time.perf_counter()
    print(f"Time to train DBN: {end_dbn-start_dbn} seconds")
    ### End Training Deep Belief Network

    ### Start Training Deep Chess
    # Generate model structure
    deepChess = getDeepChessModel(DBN)
    
    # Get training data
    white_wins, white_losses = load_data(-1, -1)

    # Define data generator and callback to write logs to file
    dataGenerator = Generator(white_wins, white_losses, DEEP_CHESS_NUM_BATCHES, DEEP_CHESS_BATCH_SIZE)
    csv_logger = callbacks.CSVLogger(f'./logs/deepChess.txt', "\n")

    # Train the model
    deepChess.fit(dataGenerator, epochs=DEEP_CHESS_EPOCHS, callbacks=[csv_logger])

    deepChess.save("./models/deepChess.h5", save_format="h5")
    end = time.perf_counter()
    ### End Training Deep Chess

    print(f"Total time to train everything: {end - start} seconds")
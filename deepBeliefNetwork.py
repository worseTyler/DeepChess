from keras import layers, Model, Input
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
bit_strings = np.load("./data/bit_strings.npy")
# bit_strings = np.load("./data/bit_strings_2.npy")


autoencoder_layers = [773, 600, 400, 200, 100]

def create_autoencoder(input_size, latent_size, train, test):
    input = Input(shape=(input_size,))
    encoded = layers.Dense(latent_size, activation='relu')(input)
    # tanh worked SIGNIFICANTLY better than sigmoid
    decoded = layers.Dense(input_size, activation='tanh')(encoded)

    autoencoder = Model(input, decoded)
    encoder = Model(input, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(train, train, epochs=50, batch_size=50000, shuffle=True, validation_data=(test, test))
    return encoder


train, test = train_test_split(bit_strings, train_size=1000000, test_size=100000)
encoder = create_autoencoder(773, 600, train, test)
encoder.summary()

new_train = encoder.predict(train)
new_test = encoder.predict(test)

encoder_2 = create_autoencoder(600, 400, new_train, new_test)
new_train = encoder_2.predict(new_train)
new_test = encoder_2.predict(new_test)

encoder_3 = create_autoencoder(400, 200, new_train, new_test)
new_train = encoder_3.predict(new_train)
new_test = encoder_3.predict(new_test)

encoder_4 = create_autoencoder(200, 100, new_train, new_test)

input = encoder.input
stacked = Model(input, encoder_4(encoder_3(encoder_2(encoder.output))))

stacked.summary()
stacked.save("./models/deepBeliefNetwork_relu_tanh")
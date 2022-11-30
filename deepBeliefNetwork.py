from keras import layers, Model, Input
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# bit_strings = np.load("./data/bit_strings.npy")
bit_strings = np.load("./data/bit_strings_2.npy")


autoencoder_layers = [773, 600, 400, 200, 100]

def create_autoencoder(input_size, latent_size, train, test):
    input = Input(shape=(input_size,))
    encoded = layers.Dense(latent_size, activation='relu')(input)
    decoded = layers.Dense(input_size, activation='sigmoid')(encoded)

    autoencoder = Model(input, decoded)
    encoder = Model(input, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(train, train, epochs=20, batch_size=20000, shuffle=True, validation_data=(test, test))
    return encoder


train, test = train_test_split(bit_strings, train_size=20000, test_size=10000)
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

# x = encoder_2(encoder.get_layer('input_1'))
# x = encoder_3(x)
# x = encoder_4(x)
input = encoder.input

# input = Input(shape=(773,))
# x = layers.Dense(600, activation='relu')(input)
# x = layers.Dense(400, activation='relu')(x)
# x = layers.Dense(200, activation='relu')(x)
# x = layers.Dense(100, activation='relu')(x)
stacked = Model(input, encoder_4(encoder_3(encoder_2(encoder.output))))
# encoder_4(encoder_3(encoder_2(encoder.output)))
stacked.summary()

# for layer in stacked.layers:
#     print(layer.get_config())
#     print(layer.get_weights())

# full_model_prediction = stacked.predict(train)

# x = encoder.predict(train)
# x = encoder_2.predict(x)
# x = encoder_3.predict(x)
# separate_encoder_prediction = encoder_4.predict(x)

# comparison = separate_encoder_prediction == full_model_prediction
# print(comparison.all())
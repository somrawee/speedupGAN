import keras
from keras import layers


encoding_dim = 32
input_img = keras.Input(shape=(128,128,3))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
)
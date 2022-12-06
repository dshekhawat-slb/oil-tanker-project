from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import Input



def build_encoder(latent_dimension):
    '''returns an encoder model, of output_shape equals to latent_dimension'''
    encoder = Sequential(name="Encoder")
    
    encoder.add(Conv2D(8, (2,2), input_shape=(28, 28, 1), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(16, (2, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(32, (2, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))     

    encoder.add(Flatten())
    encoder.add(Dense(latent_dimension, activation='tanh',name="latent_space"))
    
    return encoder


def build_decoder(latent_dimension):
    # $CHALLENGIFY_BEGIN
    decoder = Sequential(name="Decoder")
    
    decoder.add(Dense(7*7*8, activation='tanh', input_shape=(latent_dimension,)))
    decoder.add(Reshape((7, 7, 8)))  # no batch axis here
    decoder.add(Conv2DTranspose(8, (2, 2), strides=2, padding='same', activation='relu'))

    decoder.add(Conv2DTranspose(1, (2, 2), strides=2, padding='same', activation='relu'))
    return decoder




def build_autoencoder():
    
    inp = Input((28, 28,1),name="Input")
    encoded = build_encoder(16)(inp)

    classif = Dense(10,activation='softmax',name="Classification")(encoded)    
    decoded = build_decoder(16)(encoded)
    
    autoencoder = Model(inp, [classif,decoded])
    return autoencoder
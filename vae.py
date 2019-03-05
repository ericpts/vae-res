import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

from config import rootk


class VAE(tf.keras.Model):
    def __init__(self, latent_dim: int, name: str) -> None:
        super(VAE, self).__init__(name=name)

        self.latent_dim = latent_dim

        self.encoder = self.encoder_network(self.latent_dim)
        self.decoder = self.decoder_network(self.latent_dim)

        print(self.encoder.summary())
        print(self.decoder.summary())

    def convolutional_layers(self, transp: bool):
        if transp:
            conv_layer = keras.layers.Conv2DTranspose
        else:
            conv_layer = keras.layers.Conv2D

        def fn(inputs):
            X = inputs

            base_filters = np.array([8, 8, 32])

            filters = base_filters
            # X = convolutional(filters, 2, transp=transp)(X)
            # X = identitiy(filters, transp=transp)(X)

            X = conv_layer(16, kernel_size=3, strides=2, padding='same')(X)
            X = keras.layers.BatchNormalization(axis=3)(X)
            X = keras.layers.Activation('relu')(X)

            if transp: X = keras.layers.Cropping2D( cropping=((1, 0), (1, 0)))(X)

            X = conv_layer(32, kernel_size=3, strides=2, padding='same')(X)
            X = keras.layers.BatchNormalization(axis=3)(X)
            X = keras.layers.Activation('relu')(X)

            X = conv_layer(64, kernel_size=3, strides=2, padding='same')(X)
            X = keras.layers.BatchNormalization(axis=3)(X)
            X = keras.layers.Activation('relu')(X)

            X = conv_layer(128, kernel_size=3, strides=2, padding='same')(X)

            filters = 2 * base_filters
            # X = convolutional(filters, 2, transp=transp)(X)
            # X = identitiy(filters, transp=transp)(X)

            filters = 4 * base_filters
            # X = convolutional(filters, 2, transp=transp)(X)
            # X = identitiy(filters, transp=transp)(X)

            return X

        return fn

    def encoder_network(self, latent_dim: int) -> tf.keras.Model:
        inputs = keras.Input(shape=(28 * rootk, 28 * rootk, 1))

        X = inputs

        X = self.convolutional_layers(False)(X)

        X = keras.layers.Flatten()(X)

        # Here we use 2 * latent_dim, because each of the cells will represent a Gaussian dist.
        X = keras.layers.Dense(latent_dim + latent_dim)(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='Encoder')
        return model

    def decoder_network(self, latent_dim: int) -> tf.keras.Model:
        first_shape = self.encoder.layers[-3].output_shape[1:]

        inputs = keras.Input(shape=(latent_dim, ))
        X = inputs
        X = keras.layers.Dense(np.prod(first_shape), activation='relu')(X)
        X = keras.layers.Reshape(first_shape)(X)

        X = self.convolutional_layers(True)(X)

        X = keras.layers.Conv2DTranspose(filters=1,
                kernel_size=3,
                activation='sigmoid',
                padding='same')(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='Decoder')
        return model

    def sample(self, eps):
        return self.decode(eps)

    def encode(self, x):
        latent_var = self.encoder(x)
        (mean, logvar) = tf.split(latent_var, num_or_size_splits=2, axis=1)
        return (mean, logvar)

    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(1/2 * logvar) + mean

    def decode(self, z):
        return self.decoder(z)

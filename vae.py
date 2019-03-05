import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

from config import *


class VAE(tf.keras.Model):
    def __init__(self, latent_dim: int, name: str = 'VAE') -> None:
        super(VAE, self).__init__(name=name)

        self.latent_dim = latent_dim

        self.encoder = self.encoder_network(self.latent_dim)
        self.decoder = self.decoder_network(self.latent_dim)

    def convolutional_layers(self, transp: bool):
        if transp:
            conv_layer = keras.layers.Conv2DTranspose
        else:
            conv_layer = keras.layers.Conv2D

        def fn(inputs):
            X = inputs

            nlayers = 3
            for i in range(nlayers):
                layer_name = 'layer-{}'.format(i)

                X = conv_layer(16 * 2**i, kernel_size=3, strides=2, padding='same')(X)
                X = keras.layers.BatchNormalization(axis=3)(X)
                X = keras.layers.Activation('relu', name=layer_name)(X)

                if not transp:
                    continue

                transp_layer_name = 'layer-{}'.format(nlayers - i - 1)
                X_transp = self.encoder.get_layer(transp_layer_name)


                desired_shape = X_transp.output_shape[1:]
                cur_shape = X.shape[1:]

                if desired_shape != cur_shape:
                    dx = cur_shape[0] - desired_shape[0]
                    dy = cur_shape[1] - desired_shape[1]

                    X = keras.layers.Cropping2D(
                            cropping=((dx, 0), (dy, 0)))(X)

            X = conv_layer(32 * 2**nlayers, kernel_size=3, strides=2, padding='same')(X)

            return X

        return fn

    def encoder_network(self, latent_dim: int) -> tf.keras.Model:
        inputs = keras.Input(
                shape=(28 * expand_per_height, 28 * expand_per_width, 1))

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

    def compute_kl_loss(self, mean, logvar):
        kl_loss = 1 + logvar - mean**2 - tf.exp(logvar)
        kl_loss = tf.math.reduce_sum(kl_loss, axis=1)
        kl_loss *= -0.5

        return kl_loss

    def summarize(self):
        print(self.encoder.summary())
        print(self.decoder.summary())

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

class VAE(tf.keras.Model):
    def __init__(self, latent_dim: int, name: str) -> None:
        super(VAE, self).__init__(name=name)

        self.latent_dim = latent_dim

        self.encoder = VAE.encoder_network(self.latent_dim)
        self.decoder = VAE.decoder_network(self.latent_dim)

        # print(self.encoder.summary())
        # print(self.decoder.summary())

    @staticmethod
    def encoder_network(latent_dim: int) -> tf.keras.Model:
        inputs = keras.Input(shape=(28, 28, 1))

        X = keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu')(inputs)

        X = keras.layers.Conv2D(
                filters=64,
                kernel_size=3,
                strides=2,
                padding='same',
                activation='relu')(X)

        X = keras.layers.Flatten()(X)

        # TODO(): Maybe remove this.
        X = keras.layers.Dense(16, activation='relu')(X)

        # Here we use 2 * latent_dim, because each of the cells will represent a Gaussian dist.
        X = keras.layers.Dense(latent_dim + latent_dim)(X)

        model = keras.models.Model(inputs=inputs, outputs=X, name='Encoder')
        return model

    @staticmethod
    def decoder_network(latent_dim: int) -> tf.keras.Model:
        inputs = keras.Input(shape=(latent_dim, ))
        X = keras.layers.Dense(7 * 7 * 64, activation='relu')(inputs)

        X = keras.layers.Reshape((7, 7, 64))(X)

        X = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2,
                padding='same',
                activation='relu')(X)

        X = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                padding='same',
                activation='relu')(X)

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

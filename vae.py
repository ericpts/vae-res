import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

from coord_conv import CoordConv2D

from config import global_config


class VAE(tf.keras.Model):

    def __init__(self, latent_dim: int, name: str = 'VAE') -> None:
        super(VAE, self).__init__(name=name)

        self.nlayers = global_config.nlayers
        self.latent_dim = latent_dim

        self.layer_sizes = [4 * 2**i for i in range(self.nlayers)]

        self.encoder = self.encoder_network(self.latent_dim)
        self.decoder = self.decoder_network(self.latent_dim)

    def convolutional_layers(self, transp: bool):

        layer_sizes = self.layer_sizes
        if transp:
            layer_sizes.reverse()

        def fn(inputs):
            X = inputs

            for i in range(self.nlayers):
                if not transp:
                    layer_name = f'conv-layer-{i}'
                else:
                    layer_name = f'transp-conv-layer-{i}'

                nfilters = layer_sizes[i]

                X = CoordConv2D(
                    transp,
                    filters=nfilters,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    name=layer_name,
                )(X)

                X = keras.layers.BatchNormalization(axis=3)(X)
                X = keras.layers.Activation('relu')(X)

                # ReLU should go after.

                if not transp:
                    continue

                transp_layer_name = 'conv-layer-{}'.format(self.nlayers - i - 1)
                X_transp = self.encoder.get_layer(transp_layer_name)

                desired_shape = X_transp.input_shape[1:-1]
                cur_shape = X.shape[1:-1]

                if desired_shape != cur_shape:
                    dx = cur_shape[0] - desired_shape[0]
                    dy = cur_shape[1] - desired_shape[1]

                    X = keras.layers.Cropping2D(cropping=((dx, 0), (dy, 0)))(X)

            return X

        return fn

    def encoder_network(self, latent_dim: int) -> tf.keras.Model:
        inputs = keras.Input(
            shape=(28 * global_config.expand_per_height, 28 * global_config.expand_per_width,
                   1))

        X = inputs

        X = self.convolutional_layers(False)(X)

        X = keras.layers.Flatten(name='encoder-flatten')(X)

        X = keras.layers.Dense(64, name='encoder-last-fc', activation='relu')(X)

        mean = keras.layers.Dense(latent_dim)(X)
        logvar = keras.layers.Dense(latent_dim)(X)

        model = keras.models.Model(
            inputs=inputs,
            outputs=[mean, logvar],
            name='encoder')
        return model

    def decoder_network(self, latent_dim: int) -> tf.keras.Model:
        first_shape = self.encoder.get_layer('encoder-flatten').input_shape[1:]

        inputs = keras.Input(shape=(latent_dim,))
        X = inputs
        X = keras.layers.Dense(
            np.prod(first_shape),
            activation='relu',
            name='decoder-first-fc')(X)
        X = keras.layers.Reshape(first_shape)(X)

        X = self.convolutional_layers(True)(X)

        img = CoordConv2D(
            transp=True,
            filters=1,
            kernel_size=3,
            activation='sigmoid',
            padding='same',
            name='decoder-image')(X)

        confidence = CoordConv2D(
            transp=True,
            filters=1,
            kernel_size=3,
            padding='same',
            name='decoder-raw-confidence')(X)

        model = keras.models.Model(
            inputs=inputs, outputs=[img, confidence], name='decoder')

        return model


    def sample(self, eps):
        return self.decode(eps)


    def encode(self, x):
        (mean, logvar) = self.encoder(x)
        return (mean, logvar)


    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(1 / 2 * logvar) + mean


    def decode(self, z):
        return self.decoder(z)


    @staticmethod
    def compute_kl_loss(mean, logvar):
        kl_loss = 1 + logvar - mean**2 - tf.exp(logvar)
        kl_loss = tf.math.reduce_sum(kl_loss, axis=1)
        kl_loss *= -0.5

        return kl_loss


    def call(self, inputs):
        mean, logvar = self.encode(inputs)

        self.last_mean = mean
        self.last_logvar = logvar

        z = self.reparametrize(mean, logvar)
        return self.decode(z)


    def get_trainable_variables(self):
        return self.trainable_variables


    def summarize(self):
        self.encoder.summary()
        self.decoder.summary()

import tensorflow.keras as keras
import tensorflow as tf

from spatial_broadcast_decoder import SpatialBroadcastDecoder
from coord_conv import CoordConv2D

from config import global_config


class VAE(tf.keras.Model):

    def __init__(self, latent_dim: int, name: str = 'VAE') -> None:
        super(VAE, self).__init__(name=name)

        self.nlayers = global_config.nlayers
        self.latent_dim = latent_dim

        self.layer_sizes = []
        for i in range(self.nlayers):
            self.layer_sizes.append(32 * 2 ** (i // 2))

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
                    kernel_initializer='glorot_normal',
                )(X)
                # X = tf.debugging.check_numerics(X, 'CoordConv2D')

                X = keras.layers.BatchNormalization(axis=3)(X)
                # X = tf.debugging.check_numerics(X, 'BatchNorm')

                X = keras.layers.LeakyReLU()(X)
                # X = tf.debugging.check_numerics(X, 'LeakyReLU')

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
            shape=(
                global_config.img_height,
                global_config.img_width,
                global_config.img_channels
                )
            )

        X = inputs

        X = self.convolutional_layers(False)(X)

        X = keras.layers.Flatten(name='encoder-flatten')(X)

        X = keras.layers.Dense(
            256,
            name='encoder-fc',
            kernel_initializer='glorot_normal',
        )(X)
        # X = tf.debugging.check_numerics(X, 'FC')
        X = keras.layers.LeakyReLU()(X)
        # X = tf.debugging.check_numerics(X, 'FC-LeakyReLU')


        mean = keras.layers.Dense(latent_dim)(X)
        logvar = keras.layers.Dense(latent_dim)(X)

        model = keras.models.Model(
            inputs=inputs,
            outputs=[mean, logvar],
            name='encoder')
        return model

    def decoder_network(self, latent_dim: int) -> tf.keras.Model:
        inputs = keras.Input(shape=(latent_dim,))

        X = inputs
        X = SpatialBroadcastDecoder(
            global_config.img_height + 8,
            global_config.img_width + 8,
        )(X)

        for _ in range(self.nlayers):
            X = tf.keras.layers.Conv2D(
                kernel_size=3,
                filters=32,
                padding='valid',
                strides=1,
                kernel_initializer='glorot_normal',
                activation='relu',
            )(X)
            # X = tf.debugging.check_numerics(X, 'Conv2D')

        img = tf.keras.layers.Conv2D(
            filters=global_config.img_channels,
            kernel_size=3,
            activation='sigmoid',
            padding='same',
            name='decoder-image',
            kernel_initializer='glorot_normal',
        )(X)
        # X = tf.debugging.check_numerics(X, 'img')

        confidence = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            padding='same',
            name='decoder-raw-confidence',
            kernel_initializer='glorot_normal',
        )(X)
        # X = tf.debugging.check_numerics(X, 'confidence')

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

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


class CoordAdder(keras.layers.Layer):
    def __init__(
            self,
            **kwargs):
        super(CoordAdder, self).__init__(**kwargs)


    def build(self, input_shape):
        def build_once(dim):
            x, y = dim
            m = np.zeros(dim, dtype=np.float32)
            for i in range(x):
                m[i] = i
            m = (m / (x - 1)) * 2 - 1
            return m

        (x, y) = input_shape[-3:-1]

        def matrix_to_4d_tensor(m):
            m = np.expand_dims(m, 0)
            m = np.expand_dims(m, -1)
            return m

        self.im = matrix_to_4d_tensor(build_once((x, y)))
        self.jm = matrix_to_4d_tensor(np.transpose(build_once((y, x))))


    def call(self, X):
        batch_size = tf.shape(X)[0]
        itensor = tf.tile(self.im, [batch_size, 1, 1, 1])
        jtensor = tf.tile(self.jm, [batch_size, 1, 1, 1])
        return tf.concat(
            (X, itensor, jtensor),
            3
        )


class CoordConv2D(keras.layers.Layer):
    def __init__(
            self,
            transp: bool,
            name: str,
            **kwargs,
    ):
        super(CoordConv2D, self).__init__(name=name)

        if transp:
            conv_layer = keras.layers.Conv2DTranspose
        else:
            conv_layer = keras.layers.Conv2D

        self.conv = conv_layer(
            **kwargs
        )

        self.coord_adder = CoordAdder()


    def call(self, X):
        X = self.coord_adder(X)
        X = self.conv(X)
        return X

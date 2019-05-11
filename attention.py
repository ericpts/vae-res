import tensorflow as tf
"""
Class for 2D attention.

The underlying idea is that it generates a vector, which is then
dotted with the 2D matrix in order to represent which pixels it
should be focusing on.
"""


class Attention(tf.keras.layers.Layer):

    def __init__(self, vec_dim: int = 64, **kwargs) -> None:
        super(Attention, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.vec_dim = vec_dim

    def build(self, input_shape):
        image = tf.keras.Input(shape=input_shape)
        residual = tf.keras.Input(shape=input_shape)

        X = image
        X = tf.keras.layers.Dense(
            self.vec_dim,
            name='attn-vec',
            activation='relu',
        )(X)

        X = tf.keras.layers.Dense(
            self.x * self.y,
            name='attn-mask',
            activation='sigmoid',
        )(X)

        X = tf.keras.layers.Reshape(input_shape)(X)

        X = tf.keras.layers.Multiply(X, residual)

        self.model = tf.keras.models.Model(
            inputs=[image, residual], outputs=X, name='attn-2d')

    def call(self, X):
        return self.model(X)

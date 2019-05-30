import tensorflow as tf
from coord_conv import CoordAdder

class SpatialBroadcastDecoder(tf.keras.layers.Layer):
    def __init__(self, height, width, **kwargs):
        super(SpatialBroadcastDecoder, self).__init__()
        self.height = height
        self.width = width

        self.coord_adder = CoordAdder()


    def call(self, X):
        X = tf.expand_dims(X, 1)
        X = tf.expand_dims(X, 1)
        X = tf.tile(X, [1, self.height, self.width, 1])
        X = self.coord_adder(X)
        return X

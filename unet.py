import tensorflow as tf
import numpy as np
from instancenormalization import InstanceNormalization


class UNet(tf.keras.models.Model):
    def __init__(self, nblocks: int, **kwargs) -> None:
        super(UNet, self).__init__(**kwargs)
        self.nblocks = nblocks

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        X = inputs

        filters = [
            64 * 2**i for i in range(self.nblocks)
        ]

        skip_tensors = []

        # Downsample.
        for i in range(self.nblocks):
            shape = input_shape
            X = tf.keras.layers.Conv2D(
                filters[i], 3,
                use_bias=False,
            )(X)
            X = InstanceNormalization(
                axis=3,
                scale=False,
            )(X)
            X = tf.keras.layers.ReLU()(X)

            skip_tensors.append(X)

            if i == self.nblocks - 1:
                # No resizing happens on the last block.
                break

            new_shape = tuple([
                int(s / 2) for s in shape
            ])
            X = tf.image.resize(
                X,
                new_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            shape = new_shape

        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(128, activation='relu')(X)
        X = tf.keras.layers.Dense(128, activation='relu')(X)
        X = tf.keras.layers.Dense(
            np.prod(skip_tensors[-1].output_shape[1:]),
            activation='relu',
        )(X)

        X = tf.keras.layers.Reshape(skip_tensors[-1].output_shape[1:])

        # Upsample + skip connections.
        for i in range(self.nblocks):
            X = tf.keras.layers.Concatenate()([X, skip_tensors[-1]])
            skip_tensors = skip_tensors[:-1]
            X = tf.keras.layers.Conv2D(
                filters[i], 3,
                use_bias=False,
            )(X)
            X = InstanceNormalization(
                axis=3,
                scale=False,
            )(X)
            X = tf.keras.layers.ReLU()(X)
            new_shape = skip_tensors[-1].output_shape[1:]
            X = tf.image.resize(
                X,
                new_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )

        # Generate final logit probabilities
        X = tf.keras.layers.Conv2D(1, 1)

        X = tf.nn.log_softmax(
            tf.keras.layers.Concatenate()([
                X, tf.zeros_like(X)
            ])
        )
        logs, log1ms = tf.split(X, 2, axis=-1)

        self.model = tf.keras.models.Model(
            inputs=[inputs],
            outputs=[logs, log1ms]
        )

    def call(self, X, image, log1ms):
        return self.model(
            image, log1ms
        )

import tensorflow as tf
import numpy as np
from config import global_config
from instancenormalization import InstanceNormalization
import data_util


class UNet(tf.keras.models.Model):

    def __init__(self, nblocks: int, **kwargs) -> None:
        super(UNet, self).__init__(**kwargs)
        self.nblocks = nblocks

        self._build((None,
                     28 * global_config.expand_per_height,
                     28 * global_config.expand_per_width,
                     2))

    def _build(self, input_shape):
        assert len(input_shape) == 4

        cur_filter = 32
        filters = []
        for i in range(self.nblocks):
            filters.append(cur_filter)
            if len(filters) % 2 == 0:
                cur_filter *= 2

        # Drop the batch size.
        input_shape = input_shape[1:]
        skip_tensors = []
        shape = input_shape[0:2]

        inputs = tf.keras.Input(shape=input_shape)
        X = inputs

        X = data_util.assert_all_finite(X)

        # Downsample.
        for i in range(self.nblocks):
            X = tf.keras.layers.Conv2D(
                filters[i],
                3,
                use_bias=False,
                padding='same',
            )(X)

            X = data_util.assert_all_finite(X)

            X = InstanceNormalization(
                axis=3,
                scale=False,
            )(X)

            X = data_util.assert_all_finite(X)

            X = tf.keras.layers.ReLU()(X)

            X = data_util.assert_all_finite(X)

            skip_tensors.append(X)

            if i == self.nblocks - 1:
                # No resizing happens on the last block.
                break

            new_shape = tuple([int(s / 2) for s in shape])

            X = tf.image.resize(
                X,
                new_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )

            X = data_util.assert_all_finite(X)

            shape = new_shape

        X = tf.keras.layers.Flatten()(X)

        # TODO: Make this 128 if we want to make the model better.
        X = tf.keras.layers.Dense(32, activation='relu')(X)

        X = data_util.assert_all_finite(X)

        X = tf.keras.layers.Dense(32, activation='relu')(X)

        X = data_util.assert_all_finite(X)

        X = tf.keras.layers.Dense(
            np.prod(skip_tensors[-1].get_shape()[1:]),
            activation='relu',
        )(X)

        X = data_util.assert_all_finite(X)

        X = tf.keras.layers.Reshape(skip_tensors[-1].get_shape()[1:])(X)

        # Upsample + skip connections.
        for i in range(self.nblocks):
            X = tf.keras.layers.Concatenate()([X, skip_tensors[-1]])
            skip_tensors = skip_tensors[:-1]

            X = tf.keras.layers.Conv2D(
                filters[self.nblocks - 1 - i],
                3,
                use_bias=False,
                padding='same',
            )(X)

            X = data_util.assert_all_finite(X)
            X = InstanceNormalization(
                axis=3,
                scale=False,
            )(X)

            X = data_util.assert_all_finite(X)
            X = tf.keras.layers.ReLU()(X)

            X = data_util.assert_all_finite(X)

            if i == self.nblocks - 1:
                break

            new_shape = skip_tensors[-1].get_shape()[1:3]
            X = tf.image.resize(
                X,
                new_shape,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )

            X = data_util.assert_all_finite(X)

        assert X.get_shape()[1:3] == input_shape[0:2]

        # Generate final logit probabilities
        X = tf.keras.layers.Conv2D(1, 1, padding='same')(X)

        X = tf.nn.log_softmax(
            tf.keras.layers.Concatenate()([X, tf.zeros_like(X)]),
            axis=-1,
        )

        self.model = tf.keras.models.Model(
            inputs=[inputs], outputs=[X])


    def call(self, X):
        return self.model(X)

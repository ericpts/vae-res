#!/usr/bin/env python3

import tensorflow
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import os
from pathlib import Path
from typing import List, Tuple
from vae import VAE
from supervae import SuperVAE
from util import *
from config import *

try:
    tf.config.gpu.set_per_process_memory_growth(True)
except AttributeError:
    pass

os.makedirs('checkpoints', exist_ok=True)


@tf.function
def compute_loss(model, x):
    return model.compute_loss(x)


@tf.function
def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.get_trainable_variables()), loss


@tf.function
def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


def train_model(
        model: tf.keras.Model,
        D_train: tf.data.Dataset,
        D_test: tf.data.Dataset,
        epochs: int) -> tf.keras.Model:

    optimizer = tf.keras.optimizers.Adam()

    tf.random.set_seed(10)
    random_vector_for_gen = tf.random.normal((num_examples, latent_dim))
    tf.random.set_seed(time.time())

    start_epoch = get_latest_epoch(model.name)
    if start_epoch:
        print('Resuming training from epoch {}'.format(start_epoch))
        model.load_weights(checkpoint_for_epoch(model.name, start_epoch))
    start_epoch += 1

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()

        train_loss = 0
        train_size = 0
        for (X, y) in D_train.batch(batch_size, drop_remainder=True):
            gradients, loss = compute_gradients(model, X)

            apply_gradients(optimizer, gradients,
                    model.get_trainable_variables())

            train_loss += loss * X.shape[0]
            train_size += X.shape[0]

        end_time = time.time()

        delta_time = end_time - start_time

        print('Stats for epoch {}'.format(epoch))
        print('\t Time: {}'.format(delta_time))
        print('\t Train loss: {}'.format(train_loss / train_size))

        if epoch % 5 == 0:
            test_loss = 0
            test_size = 0
            for (X, y) in D_test.batch(batch_size * 8, drop_remainder=True):
                test_loss += compute_loss(model, X) * X.shape[0]
                test_size += X.shape[0]

            print('\t Test loss: {}'.format(test_loss / test_size))
            generate_pictures(model, random_vector_for_gen, epoch)

        if epoch % 20 == 0:
            print('Saving weights...')
            p = 'checkpoints/{}/cp_{}.ckpt'.format(model.name, epoch)
            model.save_weights(p)


def train_individual_digits(D_train, D_test):
    for d in range(1, 10):
        print('Training for digit {}'.format(d))
        model = VAE(latent_dim, 'digit-{}'.format(d))

        def filter_fn(X, y):
            return tf.math.equal(y, d)

        train_model(model, D_train.filter(filter_fn), D_test.filter(filter_fn))


def main():
    (D_init_train, D_init_test) = load_data()

    def make_filter_fn(digits):
        def filter_fn(X, y):
            return tf.math.reduce_any(
                    [tf.math.equal(y, d) for d in digits])
        return filter_fn

    def with_digits(digits):
        filter_fn = make_filter_fn(digits)

        D_train = D_init_train.filter(filter_fn)
        D_test = D_init_test.filter(filter_fn)

        D_train = combine_into_windows(D_train)
        D_test = combine_into_windows(D_test)

        return D_train, D_test

    dsets = [
            with_digits([0]),
            with_digits([0, 1]),
            ]

    model = SuperVAE(latent_dim)
    model.summarize()

    model.freeze_vae(1)
    train_model(model, *dsets[0], epochs=100)

    model.freeze_vae(0)
    model.unfreeze_vae(1)
    train_model(model, *dsets[1], epochs=200)



if __name__ == '__main__':
    main()

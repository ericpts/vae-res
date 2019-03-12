#!/usr/bin/env python3

import argparse
import tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

import time
import os
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
def compute_gradients(model, variables, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


def train_model(model: tf.keras.Model, D_train: tf.data.Dataset,
                D_test: tf.data.Dataset, epochs: int) -> tf.keras.Model:

    optimizer = tf.keras.optimizers.Adam()

    random_vector_for_gen = tf.random.normal((config.num_examples,
                                              config.latent_dim))

    start_epoch = get_latest_epoch(model.name) + 1

    variables = model.get_trainable_variables()

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()

        train_loss = 0
        train_size = 0
        for (X, y) in D_train.batch(config.batch_size, drop_remainder=True):
            gradients, loss = compute_gradients(model, variables, X)
            apply_gradients(optimizer, gradients, variables)

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
            for (X, y) in D_test.batch(
                    config.batch_size * 8, drop_remainder=True):
                test_loss += compute_loss(model, X) * X.shape[0]
                test_size += X.shape[0]

            print('\t Test loss: {}'.format(test_loss / test_size))
            fname = 'images/{}/image_at_epoch_{}.png'.format(model.name, epoch)
            save_pictures(model.sample(random_vector_for_gen), fname)

        if epoch % 20 == 0:
            print('Saving weights...')
            p = 'checkpoints/{}/cp_{}.ckpt'.format(model.name, epoch)
            model.save_weights(p)


def maybe_load_model_weights(model):
    start_epoch = get_latest_epoch(model.name)
    if start_epoch:
        print('Resuming training from epoch {}'.format(start_epoch))
        model.load_weights(checkpoint_for_epoch(model.name, start_epoch))
    start_epoch += 1


def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description='SuperVAE training utility')

    parser.add_argument('--beta', type=float, help='Beta hyperparmeter to use.')
    parser.add_argument(
        '--name', type=str, help='Name of the model.', required=True)

    args = parser.parse_args()

    (D_init_train, D_init_test) = load_data()

    def make_filter_fn(digits):

        def filter_fn(X, y):
            return tf.math.reduce_any([tf.math.equal(y, d) for d in digits])

        return filter_fn

    def with_digits(*digits):
        filter_fn = make_filter_fn(digits)

        D_train = D_init_train.filter(filter_fn)
        D_test = D_init_test.filter(filter_fn)

        D_train = combine_into_windows(D_train)
        D_test = combine_into_windows(D_test)

        return D_train, D_test

    config.beta = args.beta or config.beta

    model = SuperVAE(config.latent_dim, name=args.name)
    # model.summarize()

    maybe_load_model_weights(model)

    print('Training VAE for digit 0')
    model.unfreeze_vae(0)
    model.freeze_vae(1)
    train_model(model, *with_digits(0), epochs=40)

    print('Training VAE for digit 1')
    model.freeze_vae(0)
    model.unfreeze_vae(1)
    train_model(model, *with_digits(1), epochs=80)

    print('Training VAEs for both digits')
    model.unfreeze_vae(0)
    model.unfreeze_vae(1)
    train_model(model, *with_digits(0, 1), epochs=200)


if __name__ == '__main__':
    main()

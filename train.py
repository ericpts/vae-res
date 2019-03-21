#!/usr/bin/env python3

import datetime
from typing import Tuple
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


def train_model(
        model: tf.keras.Model,
        D: Tuple[tf.data.Dataset, tf.data.Dataset],
        start_epoch: int,
        total_epochs: int) -> tf.keras.Model:

    D_train, D_test = D

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def train_step():
        train_loss = model.fit(D_train)
        return train_loss

    def test_step():
        test_loss = model.evaluate_on_dataset(D_test)


        for (X, y) in D_test.take(config.num_examples).batch(config.num_examples):
            (softmax_confidences, vae_images) = model.run_on_input(X)
            X_output = tf.reduce_sum(softmax_confidences * vae_images, axis=0)
            fname = 'images/{}/image_at_epoch_{}.png'.format(model.name, epoch)

            imgs = (X, softmax_confidences, vae_images, X_output)
            save_pictures(X, softmax_confidences, vae_images, X_output, fname)

        return test_loss, imgs


    bar = tf.keras.utils.Progbar(total_epochs)
    bar.update(start_epoch)
    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train_step()
        test_loss, test_imgs = test_step()

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss, step=epoch)

            (X, softmax_confidences, vae_images, X_output) = test_imgs
            tf.summary.image('Input', X, step=epoch)

            for ivae in range(config.nvaes):
                tf.summary.image(f'VAE_{ivae}_softmax_confidences',
                                 softmax_confidences[ivae],
                                 step=epoch)
                tf.summary.image(f'VAE_{ivae}_images',
                                 vae_images[ivae],
                                 step=epoch)

            tf.summary.image('Output', X_output, step=epoch)

        bar.add(1, values=[("train_loss", train_loss), ("test_loss", test_loss)])

        if epoch % 20 == 0:
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

    parser = argparse.ArgumentParser(description='SuperVAE training.')

    parser.add_argument('--beta', type=float, help='Weight of the KL loss.')
    parser.add_argument('--gamma', type=float, help='Weight of the entropy loss.')
    parser.add_argument(
        '--name', type=str, help='Name of the model.', required=True)
    parser.add_argument('--epochs', type=int, nargs='+', help='Sequence of how much to train VAE_0, and then VAE_1 for.')

    args = parser.parse_args()

    (D_init_train, D_init_test, image_size, train_size, test_size) = load_data()

    def make_filter_fn(digits):
        def filter_fn(X, y):
            return tf.math.reduce_any([tf.math.equal(y, d) for d in digits])
        return filter_fn

    def with_digits(*digits):
        filter_fn = make_filter_fn(digits)

        D_train = D_init_train.filter(filter_fn)
        D_test = D_init_test.filter(filter_fn)

        D_train_0 = make_empty_windows(image_size, train_size * len(digits) // 10)
        D_test_0 = make_empty_windows(image_size, test_size * len(digits) // 10)

        D_train = D_train.concatenate(D_train_0)
        D_test = D_test.concatenate(D_test_0)

        D_train = combine_into_windows(D_train, 'data_sample_train')
        D_test = combine_into_windows(D_test, 'data_sample_test')

        return D_train, D_test

    config.epochs = args.epochs or config.epochs
    config.beta = args.beta or config.beta
    config.gamma = args.gamma or config.gamma

    model = SuperVAE(config.latent_dim, name=args.name)

    start_epoch = get_latest_epoch(model.name) + 1
    maybe_load_model_weights(model)

    print('Training VAE_0 for digit 0')
    model.unfreeze_vae(0)
    model.freeze_vae(1)
    train_model(model, with_digits(0), start_epoch, total_epochs=config.epochs[0])
    start_epoch += config.epochs[0]

    print('Training frozen VAE_0 and live VAE_1 for digits 0, 1')
    model.freeze_vae(0)
    model.unfreeze_vae(1)
    train_model(model, with_digits(0, 1), start_epoch, total_epochs=config.epochs[1])
    start_epoch += config.epochs[1]


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import datetime
from typing import Tuple, List
import argparse
import tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

import time
import os
from supervae import SuperVAE
from util import *
from config import global_config
import config

try:
    tf.config.gpu.set_per_process_memory_growth(True)
except AttributeError:
    pass

os.makedirs('checkpoints', exist_ok=True)

step_var = tf.Variable(tf.constant(0, dtype=tf.int64))
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


def train_model(
        model: tf.keras.Model,
        D_train: tf.data.Dataset,
        D_test: tf.data.Dataset,
        start_epoch: int,
        total_epochs: int) -> tf.keras.Model:

    def train_step():
        train_loss = model.fit_on_dataset(D_train)
        return train_loss

    def test_step():
        test_loss = model.evaluate_on_dataset(D_test)

        for (X, y) in D_test.shuffle(
                2**10).take(
                    global_config.num_examples).batch(
                        global_config.num_examples):
            (softmax_confidences, vae_images) = model.run_on_input(X)
            X_output = tf.reduce_sum(softmax_confidences * vae_images, axis=0)
            imgs = (X, softmax_confidences, vae_images, X_output)

        return test_loss, imgs

    def save_test_pictures(test_imgs, epoch):
        (X, softmax_confidences, vae_images, X_output) = test_imgs
        fname = 'images/{}/image_at_epoch_{}.png'.format(model.name, epoch)

        imgs = (X, softmax_confidences, vae_images, X_output)
        save_pictures(X, softmax_confidences, vae_images, X_output, fname)

        max_outputs = 4
        tf.summary.image('Input', X, max_outputs=max_outputs, step=None)

        for ivae in range(global_config.nvaes):
            tf.summary.image(f'VAE_{ivae}_softmax_confidences',
                            softmax_confidences[ivae],
                            step=None,
                            max_outputs=max_outputs)
            tf.summary.image(f'VAE_{ivae}_images',
                            vae_images[ivae],
                            step=None,
                            max_outputs=max_outputs)

        tf.summary.image('Output', X_output, max_outputs=max_outputs, step=None)


    bar = tf.keras.utils.Progbar(total_epochs)
    bar.update(start_epoch)

    for epoch in range(start_epoch, total_epochs + 1):
        step_var.assign(epoch)
        tf.summary.experimental.set_step(step_var)

        with train_summary_writer.as_default():
            train_loss = train_step()
            tf.summary.scalar('loss', train_loss, step=None)

        with test_summary_writer.as_default():

            test_loss, test_imgs = test_step()
            tf.summary.scalar('loss', test_loss, step=None)

            if epoch % 20 == 0:
                save_test_pictures(test_imgs, epoch)

        bar.add(1, values=[("train_loss", train_loss), ("test_loss", test_loss)])

    p = 'checkpoints/{}/cp_{}.ckpt'.format(model.name, total_epochs)
    model.save_weights(p)



def maybe_load_model_weights(model):
    start_epoch = get_latest_epoch(model.name)
    if start_epoch:
        print('Resuming training from epoch {}'.format(start_epoch))
        model.load_weights(checkpoint_for_epoch(model.name, start_epoch))
    start_epoch += 1


def make_filter_fn(digits: List[int]):
    def filter_fn(X, y):
        return tf.math.reduce_any([tf.math.equal(y, d) for d in digits])
    return filter_fn

def with_digits(
        digits: List[int],
        D_init: tf.data.Dataset,
        D_init_size: int,
):
    filter_fn = make_filter_fn(digits)

    D_empty_size = D_init_size * len(digits) // 10
    D = D_init
    D = D.filter(filter_fn)

    image_size = 28
    D_empty = make_empty_windows(image_size, D_empty_size)

    D = D.concatenate(D_empty)
    D = D.shuffle(D_init_size + D_empty_size)
    D = combine_into_windows(D)
    return D


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser(description='SuperVAE training.')

    config.setup_arg_parser(parser)

    parser.add_argument(
        '--name', type=str, help='Name of the model.', required=True)

    parser.add_argument(
        '--config',
        type=str,
        help='Extra yaml config file to use. It will override command line values.',
        required=False,
    )

    args = parser.parse_args()

    config.update_config_from_parsed_args(args)

    if args.config:
        cfg = Path(args.config)
        assert cfg.exists()
        config.update_config_from_yaml(cfg)

    if global_config.epochs is None:
        global_config.epochs =[
            80 + 30 * i
            for i in range(global_config.nvaes)
        ]

    print(f'Using {global_config.nvaes} VAEs')

    (D_init_train, D_init_test, image_size, train_size, test_size) = load_data()

    model = SuperVAE(global_config.latent_dim, name=args.name)

    with open('model_summary.txt', 'wt') as f:
        print_fn = lambda x : f.write(x + '\n')
        model.model.summary(print_fn=print_fn)
        model.vaes[0].encoder.summary(print_fn=print_fn)
        model.vaes[0].decoder.summary(print_fn=print_fn)

    start_epoch = get_latest_epoch(model.name) + 1
    maybe_load_model_weights(model)

    for i in range(global_config.nvaes):
        model.freeze_vae(i)

    epochs_so_far = 0
    for i in range(global_config.nvaes):
        print(f'Trainig VAE_{i} for digits up to {i}')
        model.unfreeze_vae(i)
        digits = list(range(i + 1))
        D_train = with_digits(digits, D_init_train, train_size)
        D_test = with_digits(digits + [i + 1], D_init_test, test_size)

        end_epoch = epochs_so_far + global_config.epochs[i]
        train_model(
            model,
            D_train,
            D_test,
            start_epoch,
            total_epochs=end_epoch
        )
        model.freeze_vae(i)
        start_epoch = max(start_epoch, end_epoch + 1)
        epochs_so_far = end_epoch


if __name__ == '__main__':
    main()

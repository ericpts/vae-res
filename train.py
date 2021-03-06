#!/usr/bin/env python3

import datetime
from pathlib import Path
from typing import List
import argparse
import os
import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
try:
    tf.config.gpu.set_per_process_memory_growth(True)
except AttributeError as e:
    # Memory growth must be set at program startup
    print(e)


from supervae import SuperVAE

import data_util
from data_util import BigDataset
import plot_util
import clevr_util

from config import global_config
import config

os.makedirs('checkpoints', exist_ok=True)

step_var = tf.Variable(tf.constant(0, dtype=tf.int64))
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


def train_model(
        model: tf.keras.Model,
        big_ds: BigDataset,
        start_epoch: int,
        total_epochs: int):

    def train_step():
        train_loss = model.fit_on_dataset(D_train)
        return train_loss

    def test_step():
        # TODO(ericpts): Fix this.
        # test_loss = model.evaluate_on_dataset(D_test)
        test_loss = 0

        for D in D_test.take(
                global_config.num_examples).batch(
                    global_config.num_examples):
            X = D['img']
            (softmax_confidences, vae_images) = model.run_on_input(X)
            X_output = tf.reduce_sum(softmax_confidences * vae_images, axis=0)
            imgs = (X, softmax_confidences, vae_images, X_output)

        return test_loss, imgs

    def save_test_pictures(test_imgs, epoch):
        (X, softmax_confidences, vae_images, X_output) = test_imgs
        fname = 'images/{}/image_at_epoch_{}.png'.format(model.name, epoch)

        # plot_util.save_pictures(X, softmax_confidences, vae_images, X_output, fname)

        max_outputs = 4
        tf.summary.image(
            'Input', X, max_outputs=max_outputs, step=None)

        for ivae in range(global_config.nvaes):
            tf.summary.image(f'VAE_{ivae}_softmax_confidences',
                             softmax_confidences[ivae],
                             step=None,
                             max_outputs=max_outputs)
            tf.summary.image(f'VAE_{ivae}_images',
                             vae_images[ivae],
                             step=None,
                             max_outputs=max_outputs)

        tf.summary.image(
            'Output', X_output, max_outputs=max_outputs, step=None)

    def save_model(epoch):
        p = 'checkpoints/{}/cp_{}.ckpt'.format(model.name, epoch)
        model.save_weights(p)

    D_train, D_test = big_ds

    print(f'Training from epoch {start_epoch} up to {total_epochs}')

    for epoch in range(start_epoch, total_epochs + 1):
        step_var.assign(epoch)
        tf.summary.experimental.set_step(step_var)

        with train_summary_writer.as_default():
            train_step()

        if epoch % 5 == 0:
            with test_summary_writer.as_default():
                test_loss, test_imgs = test_step()
                save_test_pictures(test_imgs, epoch)

        if epoch % 40 == 0:
            save_model(epoch)

    # save_model(total_epochs)


def maybe_load_model_weights(model):
    start_epoch = data_util.get_latest_epoch(model.name)
    if start_epoch:
        print('Resuming training from epoch {}'.format(start_epoch))
        model.load_weights(
            data_util.checkpoint_for_epoch(
                model.name, start_epoch))
    start_epoch += 1


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

    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Disable logging summary data during training.',
        required=False,
    )

    args = parser.parse_args()

    if args.no_summary:
        global train_summary_writer
        global test_summary_writer

        train_summary_writer = tf.summary.create_noop_writer()
        test_summary_writer = tf.summary.create_noop_writer()

    config.update_config_from_parsed_args(args)

    if args.config:
        cfg = Path(args.config)
        assert cfg.exists()
        config.update_config_from_yaml(cfg)

    config.dump_config_to_yaml(Path('cfg_all.yaml'))

    if global_config.clevr:
        print('Using the clevr dataset.')
        big_ds = clevr_util.Clevr(Path(global_config.clevr))
    else:
        big_ds = data_util.load_data()

    print(f'Using {global_config.nvaes} VAEs')

    model = SuperVAE(global_config.latent_dim, name=args.name)
    maybe_load_model_weights(model)

    with open('model_summary.txt', 'wt') as f:
        def print_fn(x):
            f.write(x + '\n')
        model.model.summary(print_fn=print_fn)
        model.vaes[0].encoder.summary(print_fn=print_fn)
        model.vaes[0].decoder.summary(print_fn=print_fn)

    start_epoch = data_util.get_latest_epoch(model.name) + 1
    epochs_so_far = 0

    def train_for_n_epochs(n: int, cur_big_ds: data_util.BigDataset):
        nonlocal epochs_so_far, start_epoch
        end_epoch = epochs_so_far + n

        train_model(
            model,
            cur_big_ds,
            start_epoch,
            total_epochs=end_epoch
        )
        start_epoch = max(start_epoch, end_epoch + 1)
        epochs_so_far = end_epoch

    big_ds_per_stage = []
    for i in range(global_config.nvaes):
        digits = [
            clevr_util.Clevr.OBJECTS[j] for j in range(i + 1)
        ]

        big_ds_per_stage.append(big_ds.filter_for_objects(digits))

        plot_util.plot_dataset_sample(big_ds_per_stage[-1].D_train, f'train-{i}')
        plot_util.plot_dataset_sample(big_ds_per_stage[-1].D_test, f'test-{i}')

    for i in range(global_config.nstages):
        for j in range(global_config.nvaes):
            model.freeze_all()
            model.unfreeze_vae(j)
            model.set_lr_for_new_stage(1e-4)
            train_for_n_epochs(global_config.stage_length, big_ds_per_stage[j])


if __name__ == '__main__':
    main()

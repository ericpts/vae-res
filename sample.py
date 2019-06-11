#!/usr/bin/env python3

import numpy as np
import datetime
import string
import argparse
from pathlib import Path
import tensorflow as tf
import shutil
import data_util
import plot_util
from supervae import SuperVAE
import config
from config import global_config
from vae import VAE
from tensorflow.python.framework import tensor_util
from clevr_util import Clevr

# tf.random.set_seed(1337)


def disable_tf_logging():
    import os
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
    logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)


disable_tf_logging()


def sample_digit(D_init: tf.data.Dataset, d: int) -> tf.data.Dataset:
    def filter_fn(X, y):
        return tf.math.equal(y, d)
    return D_init.filter(filter_fn).shuffle(2**20).take(1)


def generate_data_clevr(digits: str):
    assert len(digits) <= len(Clevr.OBJECTS)

    objs = []
    for d in digits:
        objs.append(Clevr.OBJECTS[int(d)])

    clevr = Clevr(global_config.clevr)
    big_ds = clevr.filter_for_objects(objs)

    X = []
    y = []

    for D in iter(big_ds[1].take(global_config.num_examples)):
        X.append(D['img'])
        y.append(D['bbox'])

    X = np.stack(X)
    y = np.stack(y)

    return (X, y)


def main():
    parser = argparse.ArgumentParser(
        description='Sample images from SuperVAE\'s')
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Name of the model to sample from.')
    parser.add_argument(
        '--digits',
        type=str,
        required=True,
        help='Digits the picture should contain. This should be a string containin digits, as well as the e character for an empty spot.')
    parser.add_argument(
        '--root-dir',
        type=str,
        required=True,
        help='Location to where the model was trained from, which contains the config files.'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=1,
        required=False,
        help='How many examples to draw.'
    )
    parser.add_argument(
        '--epoch',
        required=False,
        default='latest',
        help='Which epoch to load the checkpoints from. Either \'latest\' or an integer.'
    )
    parser.add_argument(
        '--clevr',
        required=True,
        help='Where to find the clevr dataset.'
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    assert root_dir.exists()

    # Setup a fake argparser, so that the default values are there.
    config.setup_arg_parser(argparse.ArgumentParser())

    global_config.checkpoint_dir = root_dir / 'checkpoints'
    global_config.clevr = Path(args.clevr)

    config.update_config_from_yaml(
        root_dir / 'cfg.yaml'
    )

    assert global_config.checkpoint_dir.exists()

    # Forcefully modify config, since combine_into_windows looks at it for batching.
    global_config.expand_per_width = len(args.digits)
    global_config.num_examples = args.num_examples

    (X, y) = generate_data_clevr(args.digits)

    model = SuperVAE(global_config.latent_dim, name=args.name)

    epoch = args.epoch
    if epoch == 'latest':
        epoch = data_util.get_latest_epoch(model.name)
    else:
        epoch = int(epoch)

    model.load_weights(data_util.checkpoint_for_epoch(model.name, epoch))

    (softmax_confidences, vae_images) = model.run_on_input(X)

    for i in range(model.nvaes):
        v = model.vaes[i]
        kl = VAE.compute_kl_loss(v.last_mean, v.last_logvar)
        print(f'KL-{i}: {kl}')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sample_log_dir = 'logs/gradient_tape/' + current_time + '/sample'
    sample_summary_writer = tf.summary.create_file_writer(sample_log_dir)
    step_var = tf.Variable(tf.constant(0, dtype=tf.int64))
    step_var.assign(epoch)
    tf.summary.experimental.set_step(step_var)
    with sample_summary_writer.as_default():
        model.compute_loss(X)

    log_file = None
    for d in Path(sample_log_dir).iterdir():
        log_file = d
    assert log_file

    for e in tf.compat.v1.train.summary_iterator(str(log_file)):
        for v in e.summary.value:
            value = tensor_util.MakeNdarray(v.tensor)
            print(f'{v.tag}: {value}')

    shutil.rmtree(str(Path(sample_log_dir).parent))

    X_output = tf.reduce_sum(softmax_confidences * vae_images, axis=0)
    plot_util.save_pictures(X, softmax_confidences, vae_images, X_output, 'sample.png')


if __name__ == '__main__':
    main()

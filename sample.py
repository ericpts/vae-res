#!/usr/bin/env python3

def disable_tf_logging():
    import os
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
    logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

disable_tf_logging()

import pickle
import numpy as np
import datetime
import string
import argparse
from pathlib import Path
import tensorflow as tf
import shutil
from supervae import SuperVAE
import util
import config
from config import global_config
from vae import VAE
from tensorflow.python.framework import tensor_util

tf.random.set_seed(1337)

def sample_digit(D_init: tf.data.Dataset, d: int) -> tf.data.Dataset:
    def filter_fn(X, y):
        return tf.math.equal(y, d)
    return D_init.filter(filter_fn).shuffle(2**20).take(1)


def generate_data(digits: str):
    (D_init_train, D_init_test, image_size, train_size, test_size) = util.load_data()

    # Since the training samples were already observed, we will only make use of the test ones.
    D_init = D_init_test.shuffle(test_size)
    D = None

    for _ in range(global_config.num_examples):
        for d in digits:
            if d in string.digits:
                d = int(d)
                D_cur = sample_digit(D_init, d)
            else:
                assert d == 'e'
                D_cur = util.make_empty_windows(image_size, 1)

            if D:
                D = D.concatenate(D_cur)
            else:
                D = D_cur

    D = util.combine_into_windows(D)
    D = D.batch(global_config.num_examples)

    (X, y) = next(iter(D))
    return (X, y)


def load_data(digits: str):
    dst = Path(
        f'.cache/data_{digits}_n{global_config.num_examples}.npz')

    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        (X, y) = generate_data(digits)
        np.savez(str(dst), X=X, y=y)
    else:
        bin = np.load(str(dst))
        X = bin['X']
        y = bin['y']
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

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    assert root_dir.exists()

    # Setup a fake argparser, so that the default values are there.
    config.setup_arg_parser(argparse.ArgumentParser())

    global_config.checkpoint_dir = root_dir / 'checkpoints'

    config.update_config_from_yaml(
        root_dir / 'cfg.yaml'
    )

    assert global_config.checkpoint_dir.exists()

    # Forcefully modify config, since combine_into_windows looks at it for batching.
    global_config.expand_per_width = len(args.digits)
    global_config.num_examples = args.num_examples

    model = SuperVAE(global_config.latent_dim, name=args.name)

    epoch = args.epoch
    if epoch == 'latest':
        epoch = util.get_latest_epoch(model.name)
    else:
        epoch = int(epoch)

    (X, y) = load_data(args.digits)

    model.load_weights(util.checkpoint_for_epoch(model.name, epoch))

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
    util.save_pictures(X, softmax_confidences, vae_images, X_output, None)


if __name__ == '__main__':
    main()

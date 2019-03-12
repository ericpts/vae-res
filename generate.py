#!/usr/bin/env python3
import argparse
import tensorflow as tf
from supervae import SuperVAE
from util import *
from config import config


def main():
    parser = argparse.ArgumentParser(
        description='Sample images from SuperVAE\'s')
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Name of the model to sample from.')
    parser.add_argument(
        '--sample-from-vaes',
        type=int,
        nargs='+',
        help='Which VAE\'s to sample from.')
    args = parser.parse_args()

    random_vector_for_gen = tf.random.normal((config.num_examples,
                                              config.latent_dim))
    model = SuperVAE(config.latent_dim, name=args.name)
    model.load_weights(
        checkpoint_for_epoch(model.name, get_latest_epoch(model.name)))

    vaes_to_use = args.sample_from_vaes

    pictures = model.sample(random_vector_for_gen, vaes_to_use=vaes_to_use)
    save_pictures(pictures, file_name='sample_from_{}.png'.format(model.name))


if __name__ == '__main__':
    main()

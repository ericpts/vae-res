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
        '--digits',
        type=int,
        nargs='+',
        help='Digits the picture should contain.')

    args = parser.parse_args()

    model = SuperVAE(config.latent_dim, name=args.name)
    model.load_weights(
        checkpoint_for_epoch(model.name, get_latest_epoch(model.name)))

    (D_init_train, D_init_test, image_size, train_size, test_size) = load_data()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
import tensorflow as tf
from pathlib import Path
from supervae import SuperVAE
from util import *
from config import latent_dim, num_examples


def main():
    parser = argparse.ArgumentParser(description='Sample images from SuperVAE\'s')
    parser.add_argument('--name', type=str, required=True, help='Name of the
            model to sample from.')
    args = parser.parse_args()

    random_vector_for_gen = tf.random.normal((num_examples, latent_dim))
    model = SuperVAE(latent_dim, name=args.name)
    model.load_weights(
            checkpoint_for_epoch(model.name, get_latest_epoch(model.name)))
    generate_pictures(model, random_vector_for_gen)

if __name__ == '__main__':
    main()

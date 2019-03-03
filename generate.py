#!/usr/bin/env python3

import tensorflow as tf
from pathlib import Path
from vae import VAE
from util import *
from config import latent_dim, num_examples


random_vector_for_gen = tf.random.normal((num_examples, latent_dim))

model = VAE(latent_dim)

model.load_weights(
        checkpoint_for_epoch(get_latest_epoch()))

generate_pictures(model, random_vector_for_gen)

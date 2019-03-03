#!/usr/bin/env python3

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from typing import List
from vae import VAE
from util import *
from config import latent_dim, num_examples

try:
    tf.config.gpu.set_per_process_memory_growth(True)
except AttributeError:
    pass

os.makedirs('checkpoints', exist_ok=True)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = np.reshape(X_train, (-1, 28, 28, 1)).astype(np.float32)
X_test = np.reshape(X_test, (-1, 28, 28, 1)).astype(np.float32)

X_train /= 255.0
X_test /= 255.0

train_size = X_train.shape[0]
test_size = X_test.shape[0]
batch_size = 64

X_train = tf.data.Dataset.from_tensor_slices(X_train).shuffle(train_size).batch(batch_size)
X_test = tf.data.Dataset.from_tensor_slices(X_test).shuffle(test_size).batch(batch_size)

# TODO(): figure out why reduction should be sum.
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='sum')
optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(x, mean, logvar):
    log2pi = tf.math.log(2. * np.pi)
    return tf.math.reduce_sum(
        -0.5 * (
            log2pi + logvar + (x - mean)**2 * tf.math.exp(-logvar)
        ),
        axis=1
    )

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparametrize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = loss_object(x, x_logit)
    logpx_z = -cross_ent
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

epochs = 100

tf.random.set_seed(10)
random_vector_for_gen = tf.random.normal((num_examples, latent_dim))
tf.random.set_seed(time.time())

model = VAE(latent_dim)

start_epoch = get_latest_epoch()
if start_epoch:
    print('Resuming training from epoch {}'.format(start_epoch))
    model.load_weights(checkpoint_for_epoch(start_epoch))
start_epoch += 1

for epoch in range(start_epoch, epochs + start_epoch):
    start_time = time.time()

    train_loss = 0
    for x in X_train:
        gradients, loss = compute_gradients(model, x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
        train_loss += loss
    end_time = time.time()

    delta_time = end_time - start_time

    print('Stats for epoch {}'.format(epoch))
    print('\t Time: {}'.format(delta_time))
    print('\t Train loss: {}'.format(train_loss / train_size))

    if epoch % 5 == 0:
        test_loss = 0
        for x in X_test:
            test_loss += compute_loss(model, x)

        print('\t Test loss: {}'.format(test_loss / test_size))
        generate_pictures(model, random_vector_for_gen, epoch)

    if epoch % 20 == 0:
        print('Saving weights...')
        p = 'checkpoints/cp_{}.ckpt'.format(epoch)
        model.save_weights(p)

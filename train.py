#!/usr/bin/env python3

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
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

image_size = X_train.shape[1]
X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
X_test = np.reshape(X_test, [-1, image_size, image_size, 1])
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

train_size = X_train.shape[0]
test_size = X_test.shape[0]
batch_size = 128

X_train = tf.data.Dataset.from_tensor_slices(X_train).shuffle(train_size).batch(batch_size)
X_test = tf.data.Dataset.from_tensor_slices(X_test).shuffle(test_size).batch(batch_size)

# TODO(): figure out why reduction should be sum.
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam()

def log_normal_pdf(x, mean, logvar):
    log2pi = tf.math.log(2. * np.pi)
    return tf.math.reduce_sum(
        -0.5 * (
            log2pi + logvar + (x - mean)**2 * tf.math.exp(-logvar)
        ),
        axis=1
    )

# def compute_loss2(model, x):
#     mean, logvar = model.encode(x)
#     z = model.reparametrize(mean, logvar)
#     x_pred = model.decode(z)
#
#     cross_ent = loss_object(
#             tf.reshape(x, (-1, )),
#             tf.reshape(x_pred, (-1, ))
#             )
#
#     cross_ent *= 28 * 28
#
#     logpx_z = -cross_ent
#     logpz = log_normal_pdf(z, 0., 0.)
#     logqz_x = log_normal_pdf(z, mean, logvar)
#
#     return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparametrize(mean, logvar)
    x_pred = model.decode(z)

    cross_ent = loss_object(x, x_pred)
    cross_ent *= 28 * 28

    kl_loss = 1 + logvar - mean**2 - tf.exp(logvar)
    kl_loss = tf.math.reduce_sum(kl_loss, axis=1)
    kl_loss *= -0.5

    vae_loss = tf.math.reduce_mean(cross_ent + kl_loss)

    return vae_loss


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

epochs = 30

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
        train_loss += loss * x.shape[0]
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

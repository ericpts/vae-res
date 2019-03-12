import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from typing import Tuple
import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from config import *

def get_latest_epoch(model_name: str) -> int:
    p = Path('checkpoints/{}'.format(model_name))

    ckpts = []
    for x in p.glob('cp_*.ckpt.index'.format(model_name)):
        m = re.search(r'cp_(\d+).ckpt.index'.format(model_name), str(x))
        n_ckpt = int(m.group(1))
        ckpts.append(n_ckpt)

    if len(ckpts) == 0:
        return 0

    ckpts = np.array(ckpts)
    return np.max(ckpts)

def checkpoint_for_epoch(model_name: str, epoch: int) -> str:
    p = Path('checkpoints/{}/cp_{}.ckpt'.format(model_name, epoch))
    return str(p)

def make_plot(pictures):
    fig = plt.figure(figsize=(16, 16))
    for i in range(8 * 8):
        plt.subplot(8, 8, i + 1)
        plt.imshow(pictures[i, :, :, 0], cmap='gray')
        plt.axis('off')


def save_pictures(pictures, file_name=None):
    make_plot(pictures)

    file_name = Path(file_name)
    file_name.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(file_name)


def load_data() -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset
        ]:
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    image_size = X_train.shape[1]
    X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
    X_test = np.reshape(X_test, [-1, image_size, image_size, 1])
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    D_train = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).shuffle(train_size)

    D_test = tf.data.Dataset.from_tensor_slices(
            (X_test, y_test)).shuffle(test_size)

    return (D_train, D_test)


def combine_into_windows(D: tf.data.Dataset) -> tf.data.Dataset:
    k = config.expand_per_width * config.expand_per_height
    D = D.repeat(k)
    D = D.shuffle(2048)
    D = D.batch(k, drop_remainder=True)

    def map_fn(X, y):
        r = []
        at = 0
        for i in range(config.expand_per_width):
            c = []
            for j in range(config.expand_per_height):
                c.append(X[at])
                at += 1
            r.append(tf.concat(c, 0))
        X = tf.concat(r, 1)
        return X, y

    D = D.map(map_fn)

    D_samples = D.take(config.num_examples)

    X = [ np.array(X) for (X, y) in D_samples ]
    X = np.array(X)

    make_plot(X)

    os.makedirs('images', exist_ok=True)
    plt.savefig('images/data_sample.png')

    return D



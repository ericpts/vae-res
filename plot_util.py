import tensorflow as tf
from typing import Optional
import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from config import global_config


def make_plot(pictures):
    fig = plt.figure(figsize=(16, 16))
    for i in range(4 * 4):
        plt.subplot(4, 4, i + 1)

        if pictures[i].shape[-1] == 1:
            plt.imshow(pictures[i, :, :, 0], cmap='gray')
        else:
            plt.imshow(pictures[i])
        plt.axis('off')


def save_pictures(
        X_input,
        vae_softmax_confidences,
        vae_images,
        X_output,
        file_name: Optional[str]):

    assert vae_softmax_confidences.shape[0] == global_config.nvaes
    assert vae_softmax_confidences.shape[1] == global_config.num_examples

    assert vae_images.shape[0] == global_config.nvaes
    assert vae_images.shape[1] == global_config.num_examples

    fig = plt.figure(figsize=(16, 32))
    plt.subplots_adjust(
        wspace=0.4,
        hspace=0.4,
    )

    k = int(global_config.num_examples ** .5)
    assert k * k == global_config.num_examples # They should display in a nice square grid.

    for i in range(k * k):

        plt.subplot(k, k, i + 1, frame_on=True)

        to_stack = []
        to_stack.append(X_input[i])

        for v in range(global_config.nvaes):
            if vae_images[v, i].shape[-1] == 3:
                to_stack.append(
                    tf.image.grayscale_to_rgb(vae_softmax_confidences[v, i]))
            else:
                to_stack.append(vae_softmax_confidences[v, i])
            to_stack.append(vae_images[v, i])

        to_stack.append(X_output[i])

        # import ipdb; ipdb.set_trace()

        to_show = np.vstack(to_stack)

        if to_show.shape[-1] == 1:
            plt.imshow(to_show[:, :, 0], cmap='gray')
        else:
            plt.imshow(to_show)

        plt.yticks(
            np.arange(2 * global_config.nvaes + 2) * 128
        )
        plt.gca().xaxis.set_visible(False)

    if file_name:
        file_name = Path(file_name)
        file_name.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(file_name)
    else:
        plt.show()

    plt.close()


def plot_dataset_sample(D: tf.data.Dataset, img_name: str):
    D_samples = D.take(global_config.num_examples)

    X = [np.array(X['img']) for X in D_samples]
    X = np.array(X)

    make_plot(X)

    os.makedirs('images', exist_ok=True)
    plt.savefig(f'images/{img_name}.png')

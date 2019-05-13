import tensorflow as tf
from typing import Optional
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from config import global_config


def make_plot(pictures):
    fig = plt.figure(figsize=(16, 16))
    for i in range(4 * 4):
        plt.subplot(4, 4, i + 1)
        plt.imshow(pictures[i, :, :, 0], cmap='gray')
        plt.axis('off')


def save_pictures(
        X_input,
        masks,
        vae_masks,
        vae_images,
        X_output,
        file_name: Optional[str]):

    assert masks.shape[0] == global_config.nvaes
    assert masks.shape[1] == global_config.num_examples

    assert vae_images.shape[0] == global_config.nvaes
    assert vae_images.shape[1] == global_config.num_examples

    fig = plt.figure(figsize=(16, 32))
    plt.subplots_adjust(
        wspace=0.4,
        hspace=0.4,
    )

    k = int(global_config.num_examples**.5)
    assert k * k == global_config.num_examples  # They should display in a nice square grid.

    for i in range(k * k):

        plt.subplot(k, k, i + 1)

        to_stack = []
        to_stack.append(X_input[i, :, :, 0])

        for v in range(global_config.nvaes):
            to_stack.append(masks[v, i, :, :, 0])
            to_stack.append(vae_masks[v, i, :, :, 0])
            to_stack.append(vae_images[v, i, :, :, 0])

        to_stack.append(X_output[i, :, :, 0])

        to_show = np.vstack(to_stack)

        plt.imshow(to_show, cmap='gray')

        plt.axis('off')

    if file_name:
        file_name = Path(file_name)
        file_name.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(file_name)
    else:
        plt.show()

    plt.close()


def plot_dataset_sample(D: tf.data.Dataset, img_name: str):
    D_samples = D.take(global_config.num_examples)

    X = [np.array(X) for (X, y) in D_samples]
    X = np.array(X)

    make_plot(X)

    os.makedirs('images', exist_ok=True)
    plt.savefig(f'images/{img_name}.png')

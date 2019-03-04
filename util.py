import matplotlib
matplotlib.use('Agg')

import os
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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


def generate_pictures(model, eps, epoch=None):
    os.makedirs('images/{}/'.format(model.name), exist_ok=True)
    pictures = model.sample(eps)

    make_plot(pictures)

    if epoch:
        plt.savefig('images/{}/image_at_epoch_{}.png'.format(model.name, epoch))
    else:
        plt.show()


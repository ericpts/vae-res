import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_latest_epoch() -> int:
    p = Path('checkpoints/')

    ckpts = []
    for x in p.glob('cp_*.ckpt.index'):
        m = re.search(r'checkpoints/cp_(\d+).ckpt.index', str(x))
        n_ckpt = int(m.group(1))
        ckpts.append(n_ckpt)

    if len(ckpts) == 0:
        return 0

    ckpts = np.array(ckpts)
    return np.max(ckpts)

def checkpoint_for_epoch(epoch: int) -> str:
    p = Path('checkpoints/cp_{}.ckpt'.format(epoch))
    return str(p)


def generate_pictures(model, eps, epoch=None):
    pictures = model.sample(eps)

    fig = plt.figure(figsize=(8, 8))
    for i in range(8 * 8):
        plt.subplot(8, 8, i + 1)
        plt.imshow(pictures[i, :, :, 0], cmap='gray')
        plt.axis('off')

    if epoch:
        plt.savefig('image_at_epoch_{}.png'.format(epoch))
    else:
        plt.show()


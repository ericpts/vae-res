import tensorflow as tf
from collections import namedtuple
from typing import Callable, List, Dict
import re
import numpy as np
from config import global_config


BigDataset = namedtuple(
    'BigDataset',
    ['D_train', 'D_test'])


def filter_big_dataset(
        big_ds: BigDataset,
        filter_fn: Callable[[np.array, float], bool]
) -> BigDataset:

    (Dr, Dt) = big_ds
    return BigDataset(
        Dr.filter(filter_fn),
        Dt.filter(filter_fn))


def make_big_dataset_with_empty_images(img_size: int,
                                  size_train: int,
                                  size_test: int) -> BigDataset:
    def make_tf_dataset(n):
        X = np.zeros((n, img_size, img_size, 1), dtype=np.float32)
        y = np.array([-1] * n, dtype=np.uint8)
        D = tf.data.Dataset.from_tensor_slices((X, y))
        return D

    return BigDataset(
        make_tf_dataset(size_train),
        make_tf_dataset(size_test)
    )


def augment_big_dataset_with_empty_images(big_ds: BigDataset,
                                          f_empty: float) -> BigDataset:
    size_train = int(
        f_empty * get_tf_dataset_size(big_ds.D_train)
    )
    size_test = int(
        f_empty * get_tf_dataset_size(big_ds.D_test)
    )
    return shuffle_big_dataset(
        chain_big_datasets(
            big_ds,
            make_big_dataset_with_empty_images(
                28,
                size_train,
                size_test
            )
        )
    )


def chain_big_datasets(
        *big_ds: List[BigDataset]
) -> BigDataset:

    Dr, Dt = big_ds[0]
    big_ds = big_ds[1:]

    for Drp, Dtp in big_ds:
        Dr = Dr.concatenate(Drp)
        Dt = Dt.concatenate(Dtp)

    return BigDataset(
        Dr, Dt)


def get_tf_dataset_size(ds: tf.data.Dataset):
    ret = 0
    for x in ds:
        ret += 1
    return ret


def shuffle_big_dataset(big_ds: BigDataset) -> BigDataset:
    (Dr, Dt) = big_ds
    return BigDataset(
        Dr.shuffle(get_tf_dataset_size(Dr),
                   reshuffle_each_iteration=True
        ),
        Dt.shuffle(get_tf_dataset_size(Dt),
                   reshuffle_each_iteration=True
        )
    )


def map_big_dataset(big_ds: BigDataset, map_fn) -> BigDataset:
    (Dr, Dt) = big_ds
    return BigDataset(
        Dr.map(map_fn),
        Dt.map(map_fn)
    )


def get_latest_epoch(model_name: str) -> int:
    p = global_config.checkpoint_dir / f'{model_name}'

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
    p = global_config.checkpoint_dir / f'{model_name}/cp_{epoch}.ckpt'
    return str(p)


def load_big_dataset_by_name(name: str) -> BigDataset:
    assert name in [
        'mnist',
        'fashion_mnist',
    ]

    (X_train, y_train), (X_test, y_test) = getattr(
        tf.keras.datasets, name).load_data()

    image_size = X_train.shape[1]
    assert image_size == 28

    X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
    X_test = np.reshape(X_test, [-1, image_size, image_size, 1])
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    D_train = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train))

    D_test = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test))

    return BigDataset(D_train, D_test)


def make_filter_fn(labels: List[int]):
    def filter_fn(X, y):
        return tf.math.reduce_any(
            [tf.math.equal(y, d) for d in labels])
    return filter_fn


def remap_labels(d: Dict[int, int], big_ds: BigDataset) -> BigDataset:
    ret = []
    for k, v in d.items():
        v = tf.convert_to_tensor(v, dtype=tf.uint8)

        cur_ds = filter_big_dataset(
            big_ds,
            make_filter_fn([k])
        )

        def map_fn(X, y):
            return X, v
        cur_ds = map_big_dataset(cur_ds, map_fn)
        ret.append(cur_ds)
    return chain_big_datasets(*ret)


def load_data() -> BigDataset:

    return shuffle_big_dataset(
        chain_big_datasets(
            filter_big_dataset(
                load_big_dataset_by_name('mnist'),
                make_filter_fn([0, 1])
            ),
            remap_labels(
                {
                    3: 2,
                    5: 3,
                },
                filter_big_dataset(
                    load_big_dataset_by_name('fashion_mnist'),
                    make_filter_fn([3, 5])
                )
            )
        )
    )


def combine_into_windows(
        D: tf.data.Dataset,
) -> tf.data.Dataset:
    k = global_config.expand_per_width * global_config.expand_per_height
    D = D.repeat(k)
    D = D.batch(k, drop_remainder=True)

    def map_fn(X, y):
        r = []
        at = 0
        for i in range(global_config.expand_per_width):
            c = []
            for j in range(global_config.expand_per_height):
                c.append(X[at])
                at += 1
            r.append(tf.concat(c, 0))
        X = tf.concat(r, 1)
        return X, y

    D = D.map(map_fn)

    return D

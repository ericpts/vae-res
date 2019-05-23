import tensorflow as tf
from data_util import BigDataset
import random
import numpy as np
from typing import List
from pathlib import Path
import json
from config import global_config
import gc
import time


class Clevr(object):

    OBJECTS = ['cylinder', 'sphere', 'cube']

    def __init__(self, clevr_root: Path):
        assert clevr_root.exists()

        self.clevr_root = clevr_root
        self.train = _add_bounding_boxes_to_file(json.loads(
            Path(clevr_root / 'scenes' / 'CLEVR_train_scenes.json').read_text()
        )['scenes'])

        self.test = _add_bounding_boxes_to_file(json.loads(
            Path(clevr_root / 'scenes' / 'CLEVR_val_scenes.json').read_text()
        )['scenes'])

        global_config.img_height = 320 // 4
        global_config.img_width = 480 // 4
        global_config.img_channels = 3

        print('Loaded clevr dataset.')


    def filter_for_objects(self, filter_objs: List[str]):
        def filter_once(scenes, is_test):
            ret = []
            for si, s in enumerate(scenes):
                good = True
                objs = s['objects']
                for o in objs:
                    if o['shape'] not in filter_objs:
                        good = False
                if good:
                    ret.append(si)

            def gen():
                lst = list(range(len(ret)))
                lst = random.choices(lst, k=32)

                for i in lst:
                    cur = scenes[ret[i]]
                    img_path = self.clevr_root / 'images' / cur['split'] / cur['image_filename']
                    assert img_path.exists()
                    img = tf.io.decode_image(
                        tf.io.read_file(str(img_path)),
                        channels=3,
                        dtype=tf.dtypes.float32,
                    )
                    img = tf.image.resize(
                        img,
                        (
                            global_config.img_height,
                            global_config.img_width,
                        )
                    )

                    data = {
                        'img': img
                    }

                    if is_test:
                        all_bb = []
                        for i, tp in enumerate(Clevr.OBJECTS):
                            cur_bb = np.zeros(
                                (*img.shape[0:2], 1),
                                np.float32
                            )
                            for obj in cur['objects']:
                                if obj['shape'] != tp:
                                    continue
                                bb = obj['bounding_box']

                                cur_bb[
                                    bb['xmin']: bb['xmax'],
                                    bb['ymin']: bb['ymax']] = 1.0
                            all_bb.append(cur_bb)

                        all_bb = np.stack(all_bb)

                        data['bbox'] = all_bb

                    yield data

                    gc.collect()
                    # tf.random.set_seed(1)

            output_types = {
                'img': tf.float32,
            }

            if is_test:
                output_types = {
                    'img': tf.float32,
                    'bbox': tf.float32,
                }

            D = tf.data.Dataset.from_generator(
                gen,
                output_types=output_types
                )

            return D

        return BigDataset(
            filter_once(self.train, False),
            filter_once(self.test, True)
        )


def _extract_bounding_boxes(scene):
    objs = scene['objects']
    rotation = scene['directions']['right']

    for i, obj in enumerate(objs):
        [x, y, z] = obj['pixel_coords']

        [x1, y1, z1] = obj['3d_coords']

        cos_theta, sin_theta, _ = rotation

        x1 = x1 * cos_theta + y1 * sin_theta
        y1 = x1 * -sin_theta + y1 * cos_theta

        height_d = 6.9 * z1 * (15 - y1) / 2.0
        height_u = height_d
        width_l = height_d
        width_r = height_d

        if obj['shape'] == 'cylinder':
            d = 9.4 + y1
            h = 6.4
            s = z1

            height_u *= (s * (h / d + 1)) / (
                (s * (h / d + 1)) - (s * (h - s) / d))
            height_d = height_u * (h - s + d) / (h + s + d)

            width_l *= 11 / (10 + y1)
            width_r = width_l

        if obj['shape'] == 'cube':
            height_u *= 1.3 * 10 / (10 + y1)
            height_d = height_u
            width_l = height_u
            width_r = height_u

        obj['bounding_box'] = {
            'ymin': int(y - height_d),
            'ymax': int(y + height_u),
            'xmin': int(x - width_l),
            'xmax': int(x + width_r),
        }


def _add_bounding_boxes_to_file(scenes: dict) -> dict:
    for scene in scenes:
        _extract_bounding_boxes(scene)
    return scenes
import tensorflow as tf
from data_util import BigDataset
import numpy as np
from typing import List
from pathlib import Path
import json
from config import global_config


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

        global_config.img_height = 128
        global_config.img_width = 128
        global_config.img_channels = 1

        print('Loaded clevr dataset.')


    def ensure_converted_image_exists(self, split, image_filename):
        cache_path = Path('.cache') / split / image_filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            return

        img_path = self.clevr_root / 'images' / split / image_filename

        assert img_path.exists()
        img = _image_transformation(
            tf.io.decode_image(
                tf.io.read_file(str(img_path)),
                channels=global_config.img_channels,
                dtype=tf.dtypes.float32,
            )
        )

        tf.io.write_file(
            str(cache_path),
            tf.image.encode_png(
                tf.image.convert_image_dtype(
                    img, tf.dtypes.uint8)
            )
        )
        del img


    def read_image(self, split, image_filename):
        cache_path = tf.strings.join(
            ['.cache', split, image_filename],
            separator='/')

        return tf.io.decode_image(
            tf.io.read_file(cache_path),
            channels=global_config.img_channels,
            dtype=tf.dtypes.float32,
        )


    def convert_all_images_for_objects(self, scenes, filter_objs: List[str]):
        for si, s in enumerate(scenes):
            good = True
            objs = s['objects']
            for o in objs:
                if o['shape'] not in filter_objs:
                    good = False
            if not good:
                continue
            self.ensure_converted_image_exists(s['split'], s['image_filename'])


    def filter_for_objects(self, filter_objs: List[str]):
        def filter_once(scenes, is_test):

            chosen = []
            for si, s in enumerate(scenes):
                good = True
                objs = s['objects']
                for o in objs:
                    if o['shape'] not in filter_objs:
                        good = False
                if not good:
                    continue
                t = []

                t.append(
                    self.read_image(s['split'], s['image_filename'])
                )

                if is_test:
                    all_bb = []
                    for i, fobj in enumerate(filter_objs):
                        cur_bb = np.zeros(
                            (global_config.img_height,
                             global_config.img_width,
                             1),
                            np.float32
                        )
                        for obj in s['objects']:
                            if obj['shape'] != fobj:
                                continue
                            bb = obj['bounding_box']

                            cur_bb[
                                bb['xmin']: bb['xmax'],
                                bb['ymin']: bb['ymax']] = 1.0
                        all_bb.append(cur_bb)
                    all_bb = np.stack(all_bb)
                    t.append(all_bb)

                t = tuple(t)
                chosen.append(t)

            # Columns are: image_filename; image_id; split; ?bbox
            # Currently, each entry in `chosen` stores one row. However, tensorflow
            # wants each entry to store one column.
            tensors = []
            for c in chosen:
                for i, v in enumerate(c):
                    while len(tensors) <= i:
                        tensors.append([])
                    tensors[i].append(v)

            if is_test:
                D = tf.data.Dataset.from_tensor_slices(
                    {
                        'img': tensors[0],
                        'bbox': tensors[1],
                    })
            else:
                D = tf.data.Dataset.from_tensor_slices(
                    {
                        'img': tensors[0],
                    })
            D = D.shuffle(len(chosen))

            if not is_test:
                D = D.repeat()
                D = D.take(global_config.batch_size * global_config.epoch_length)

            print(f'Raw dataset has {len(chosen)} examples; is_test: {is_test}')

            return D

        self.convert_all_images_for_objects(self.train, filter_objs)
        self.convert_all_images_for_objects(self.test, filter_objs)

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


def _image_transformation(img):
    y0, y1 = 97, 300
    x0, x1 = 64, 256
    img = tf.image.crop_to_bounding_box(
        img,
        y0,
        x0,
        y1 - y0,
        x1 - x0,
    )

    img = tf.image.resize(
        img,
        (
            global_config.img_height,
            global_config.img_width,
        )
    )

    return img

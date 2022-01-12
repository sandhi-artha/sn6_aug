"""
BASE_IN_PATH
    elee        -> in_aug_path
        3
        5
        7
    frost
    gmap
"""
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from cfg import dg_cfg
from datagen import get_image, get_label, serialize_image, serialize_label
from datagen import get_fp_orient, get_fps_folds, _bytes_feature


def get_serialize_aug_image(raster_path, in_aug_path):
    """
        returns a list of aug filter images,
            shape: 900,900,1
            dtype: uint8
    """
    fn = os.path.basename(raster_path)  # take basename since they're same for all augs
    fn = fn.replace('.tif','.png')  # augs are in .png
    ser_aug_images = []

    for w in WIN_LIST:
        fp = os.path.join(in_aug_path, str(w), fn)
        # read image to np array
        image_aug = np.array(Image.open(fp))
        image_aug = np.expand_dims(image_aug, axis=-1)
        # serialize image
        image_aug_tensor = tf.constant(image_aug, dtype=tf.uint8)
        ser_aug_images.append(tf.io.serialize_tensor(image_aug_tensor))

    return ser_aug_images


def create_aug_tfrecord(raster_paths, cfg, base_fn, in_aug_path):
    """
    100 images per tfrecord
    image in float32 serialized
    mask in binary serialized
    size : int
        examples per tfrecord. for 3ch, 640 res, uint8, use 100 -> ~150MB/file
    output : {base_fn}01-100.tfrec
    """
    size = cfg['tfrec_size']
    tot_ex = len(raster_paths)  # total examples
    tot_tf = int(np.ceil(tot_ex/size))  # total tfrecords

    for i in range(tot_tf):
        print(f'Writing TFRecord {i} of {tot_tf}..')
        size2 = min(size, tot_ex - i*size)  # size=size2 unless for remaining in last file
        fn = f'{base_fn}-{i:02}-{size2}.tfrec'

        with tf.io.TFRecordWriter(fn) as writer:
            for j in range(size2):
                idx = i*size+j  # ith tfrec * num_img per tfrec as the start of this iteration
                image = get_image(raster_paths[idx], cfg['channel'])
                image_serial = serialize_image(image, cfg['out_precision'])

                image_augs = get_serialize_aug_image(raster_paths[idx], in_aug_path)

                label = get_label(raster_paths[idx])
                label_serial = serialize_label(label)

                fn = os.path.basename(raster_paths[idx]).split('.')[0]

                feature = {
                    'image': _bytes_feature(image_serial.numpy()),
                    'image3': _bytes_feature(image_augs[0].numpy()),
                    'image5': _bytes_feature(image_augs[1].numpy()),
                    'image7': _bytes_feature(image_augs[2].numpy()),
                    'label': _bytes_feature(label_serial.numpy()),
                    'fn' : _bytes_feature(tf.compat.as_bytes(fn))
                }

                # write tfrecords
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())



if __name__=='__main__':
    BASE_IN_PATH = '../dataset/sn6_aug/filter'

    FILT_LIST = ['elee', 'frost', 'gmap']
    WIN_LIST = [3,5,7]

    fps = get_fp_orient(dg_cfg['base_dir'], dg_cfg['orient'])
    idxs_folds = get_fps_folds(fps, dg_cfg['folds'])

    # create 1 tfrecord dataset for each filter
    for filt in FILT_LIST:

        ds_out_dir = os.path.join(dg_cfg['out_dir'], filt)
        if not os.path.isdir(ds_out_dir):
            os.makedirs(ds_out_dir)

        # create folds so it match prev version
        for i, idxs_fold in enumerate(idxs_folds):
            print(f'creating tfrecords for fold {i}')
            fps_fold = [fps[idx[0]] for idx in idxs_fold]
            out_path = os.path.join(ds_out_dir, f'fold{i}')
            
            in_aug_path = os.path.join(BASE_IN_PATH, filt)
            create_aug_tfrecord(fps_fold, dg_cfg, out_path, in_aug_path)


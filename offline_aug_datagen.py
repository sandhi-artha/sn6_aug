"""
    creates tfrecords for offline augmentation
"""
import os
import pickle

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from cfg import dg_cfg
from gpu_datagen import get_image_label, serialize_image, serialize_label
from gpu_datagen import add_base_path
from gpu_datagen import _int64_feature, _bytes_feature


def get_serialize_aug_image(raster_path, feature):
    """
        appends the feature dictionary with filtered images,
            ex: 'image3_frost' for 3x3 window size from frost filter
            shape: 900,900,1
            dtype: .mat single (fp32)
    """
    fn = os.path.basename(raster_path)  # take basename since they're same for all augs
    fn = '_'.join(fn.split('_')[-4:])
    fn = fn.replace('.tif','.mat')  # augs are in .mat

    for fil in FILT_LIST:
        for w in WIN_LIST:
            fp = os.path.join(BASE_IN_PATH, fil, str(w), fn)
            # read image from .mat to np array
            mat = loadmat(fp)
            image_aug = mat['sar_res']  # np.arr fp32
            image_aug = np.expand_dims(image_aug, axis=-1)
            # serialize image and add as feature dict
            image_aug_tensor = tf.constant(image_aug, dtype=tf.float32)
            feature[f'image{w}_{fil}'] = _bytes_feature(
                tf.io.serialize_tensor(image_aug_tensor).numpy())

    return feature


def create_aug_tfrecord(raster_paths, cfg, base_fn, orient):
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
                image, mask, data_region_idx = get_image_label(
                    raster_paths[idx], cfg['channel'])

                image_serial = serialize_image(image, cfg['out_precision'])
                label_serial = serialize_label(mask)


                fn = os.path.basename(raster_paths[idx]).split('.')[0]

                feature = {
                    'image': _bytes_feature(image_serial.numpy()),
                    'label': _bytes_feature(label_serial.numpy()),
                    'data_idx': _int64_feature(data_region_idx),
                    'fn' : _bytes_feature(tf.compat.as_bytes(fn)),
                    'orient': _int64_feature([orient]),
                }

                feature = get_serialize_aug_image(raster_paths[idx], feature)

                # write tfrecords
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

if __name__=='__main__':
    BASE_IN_PATH = '../dataset/sn6_aug/filter_crop'
    BASE_OUT_PATH = '../dataset/sn6_aug/filter_crop_tfrec'

    FILT_LIST = ['frost']  # 'elee', 'gmap'
    WIN_LIST = [3,5,7]
    ORIENT = 1

    take_folds = [0,1,2,3,4]  # only create tfrec for these folds

    # get new filenames
    with open(f'fps{ORIENT}_5folds.pickle', 'rb') as f:
        fns_folds = pickle.load(f)

    fps_folds = add_base_path(fns_folds)

    # create 1 tfrecord dataset for each filter
    if not os.path.isdir(BASE_OUT_PATH):
        os.makedirs(BASE_OUT_PATH)

    # create folds so it match prev version
    for i in take_folds:
        print(f'creating tfrecords for fold {i}')
        out_path = os.path.join(BASE_OUT_PATH, f'fold{i}_o{ORIENT}')

        fps_fold = fps_folds[i]
        create_aug_tfrecord(fps_fold, dg_cfg, out_path, ORIENT)


"""
    creates tfrecords for offline augmentation (speckle filter)
    spreading each filtered image in a dataset
    effectively creating 3x train images for each filter type
"""
import os
import pickle
import random

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from cfg import dg_cfg
from gpu_datagen import get_image_label, serialize_image, serialize_label
from gpu_datagen import _int64_feature, _bytes_feature

def get_raster_path(mat_path):
    fn = os.path.basename(mat_path)
    fn = fn.replace('.mat', '.tif')
    fn = '_'.join(['SN6_Train_AOI_11_Rotterdam_SAR-Intensity', fn])
    raster_path = os.path.join(dg_cfg['base_dir'], 'SAR-Intensity', fn)
    return raster_path

def get_aug_image(mat_path):
    """
        loads .mat with scipy and return image in numpy
    """
    mat = loadmat(mat_path)
    image_aug = mat['sar_res']  # np.arr fp32
    image_aug = np.expand_dims(image_aug, axis=-1)
    return image_aug

def create_aug_tfrecord(mat_paths, cfg, base_fn, orient):
    """
    100 images per tfrecord
    image in float32 serialized
    mask in binary serialized
    size : int
        examples per tfrecord. for 3ch, 640 res, uint8, use 100 -> ~150MB/file
    output : {base_fn}01-100.tfrec
    """
    size = cfg['tfrec_size']
    tot_ex = len(mat_paths)  # total examples
    tot_tf = int(np.ceil(tot_ex/size))  # total tfrecords

    for i in range(tot_tf):
        print(f'Writing TFRecord {i} of {tot_tf}..')
        size2 = min(size, tot_ex - i*size)  # size=size2 unless for remaining in last file
        fn = f'{base_fn}-{i:02}-{size2}.tfrec'

        with tf.io.TFRecordWriter(fn) as writer:
            for j in range(size2):
                idx = i*size+j  # ith tfrec * num_img per tfrec as the start of this iteration
                _, mask, data_region_idx = get_image_label(
                    get_raster_path(mat_paths[idx]), cfg['channel'])
                
                image = get_aug_image(mat_paths[idx])
                image_serial = serialize_image(image, cfg['out_precision'])
                label_serial = serialize_label(mask)

                # frost_5_20190804115211_20190804115445_tile_8031.mat
                fn = '_'.join(mat_paths[idx].split(os.path.sep)[1:])

                feature = {
                    'image': _bytes_feature(image_serial.numpy()),
                    'label': _bytes_feature(label_serial.numpy()),
                    'data_idx': _int64_feature(data_region_idx),
                    'fn' : _bytes_feature(tf.compat.as_bytes(fn)),
                    'orient': _int64_feature([orient]),
                }

                # write tfrecords
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def bloat_and_randomize(fns_folds):
    """for every image tile, add path to filtered folder with every window size
    creating 3x the number of original paths
    shuffle for each fold to prevent neighbors of same image
    """
    new_fns_folds = []
    for fns_fold in fns_folds:
        new_fns_fold = []
        for fn in fns_fold:
            # comply with format from matlab
            fn = '_'.join(fn.split('_')[-4:])
            fn = fn.replace('.tif','.mat')
            # add basepath for every filtered window size
            for win in WIN_LIST:
                new_fns_fold.append(
                    os.path.join(BASE_IN_PATH, FILT, str(win), fn))
        # shuffle a fold
        random.shuffle(new_fns_fold)
        new_fns_folds.append(new_fns_fold)

    return new_fns_folds

if __name__=='__main__':
    BASE_IN_PATH = '../dataset/sn6_aug/filter_crop'
    BASE_OUT_PATH = '../dataset/sn6_aug/filter_crop_tfrec_new'

    FILT = 'gmap'  # 'frost, 'elee', 'gmap'
    WIN_LIST = [3,5,7]
    ORIENT = 1

    take_folds = [0,1,2,3,4]  # only create tfrec for these folds

    # get new filenames
    # SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804111224_20190804111453_tile_8685.tif
    with open(f'fps{ORIENT}_5folds.pickle', 'rb') as f:
        fns_folds = pickle.load(f)

    # add base_path pointing to filtered folder with every window size and shuffle them
    fps_folds = bloat_and_randomize(fns_folds)

    # create 1 tfrecord dataset for each filter
    TFREC_DIR = os.path.join(BASE_OUT_PATH, FILT)
    if not os.path.isdir(TFREC_DIR):
        os.makedirs(TFREC_DIR)

    # create folds so it match prev version
    for i in take_folds:
        print(f'creating tfrecords for fold {i}')
        out_path = os.path.join(TFREC_DIR, f'fold{i}_o{ORIENT}')

        fps_fold = fps_folds[i]
        create_aug_tfrecord(fps_fold, dg_cfg, out_path, ORIENT)


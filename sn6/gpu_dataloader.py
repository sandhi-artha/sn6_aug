import os
import re    # count tfrec
import gc    # deleting stuff
import json
import yaml
import random

import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback


### GLOBAL VARIABLES ###
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_CH = len(tr_cfg['SAR_CH'])
IS_TPU = 1 if tr_cfg['DEVICE'] == 'tpu' else 0
IS_AUG_ALBU = 0
IS_AUG_TF = 0




### DATALOADER ###
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, 
    # i.e. test10-687.tfrec = 687 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    
    return np.sum(n)

def get_filenames(splits, ds_path, off_ds_path='', out=False):
    """
    splits : list
        contain what folds to use during training or val
    ds_path : str
        path to train or val dataset
    """
    fns = []  # will become ['foldx_ox-yy-zz.tfrec', ...]
    for fold in splits:
        fol_path = os.path.join(ds_path, f'{fold}_o{tr_cfg["ORIENT"]}*.tfrec')
        fold_fns  = tf.io.gfile.glob(fol_path)
        for fn in fold_fns:
            fns.append(fn)

        if off_ds_path:
            fol_path = os.path.join(off_ds_path, f'{fold}_o{tr_cfg["ORIENT"]}*.tfrec')
            fold_fns  = tf.io.gfile.glob(fol_path)
            for fn in fold_fns:
                fns.append(fn)
    
    random.shuffle(fns)

    num_img = count_data_items(fns)
    steps = num_img//tr_cfg['BATCH_SIZE']

    if out:
        print(f'{splits} files: {len(fns)} with {num_img} images')
    
    return fns, steps


def read_tfrecord(feature):
    """data_idx is [r0,r1,c0,c1]
    config controlled by global flags
    parse a serialized example
    if off_aug: use prob to select one image
    reshape to return shape of img
    rescale image to have max val of 1.0
    """
    TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'data_idx': tf.io.VarLenFeature(tf.int64),
        'fn': tf.io.FixedLenFeature([], tf.string),
        'orient': tf.io.FixedLenFeature([], tf.int64),
    }

    # validation will go here
    features = tf.io.parse_single_example(feature, TFREC_FORMAT)
    image = tf.io.parse_tensor(features["image"], tf.float32)
    label = tf.io.parse_tensor(features["label"], tf.bool)
    label = tf.cast(label, tf.float32)

    data_idx = tf.sparse.to_dense(features["data_idx"])
    h = data_idx[1] - data_idx[0]
    w = data_idx[3] - data_idx[2]
    
    image = tf.reshape(image, [h, w, IMAGE_CH])
    label = tf.reshape(label, [h, w, 1])

    image = tf.math.divide(image, tf.math.reduce_max(image))
    
    if IS_AUG_ALBU:
        image, label = aug_albu(image, label)
        # add back shape bcz numpy_function makes it unknown
        image = tf.reshape(image, [h, w, IMAGE_CH])
        label = tf.reshape(label, [h, w, 1])
    
    image, label = REDUCE_RES(image, label)

    if IS_AUG_TF:
        image, label = aug_tf(image, label)

    fn = features['fn']

    return image, label, fn


def remove_fn(img, label, fn):
    """removes meta as arg to not raise error when training"""
    return img, label


def get_training_dataset(
    filenames,
    load_fn=False,
    ordered=False,
    shuffle=True):
    """
    takes list of .tfrec files, read using TFRecordDataset,
    parse and decode using read_tfrecord func,
    returns : image, label
        both without shape and with IMAGE_DIM resolution
    ordered=True will read each files in same order.
    """
    options = tf.data.Options()
    if not ordered: options.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(options)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    if not load_fn:
        dataset = dataset.map(remove_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(tr_cfg['SHUFFLE_BUFFER'])  # 2000
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def read_tfrecord_val(feature):
    """
    val has less variation
    only needs a reduce_res
    
    """
    TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'data_idx': tf.io.VarLenFeature(tf.int64),
        'fn': tf.io.FixedLenFeature([], tf.string),
        'orient': tf.io.FixedLenFeature([], tf.int64),
    }
    
    # validation will go here
    features = tf.io.parse_single_example(feature, TFREC_FORMAT)
    image = tf.io.parse_tensor(features["image"], tf.float32)
    label = tf.io.parse_tensor(features["label"], tf.bool)
    label = tf.cast(label, tf.float32)

    data_idx = tf.sparse.to_dense(features["data_idx"])
    h = data_idx[1] - data_idx[0]
    w = data_idx[3] - data_idx[2]
    
    image = tf.reshape(image, [h, w, IMAGE_CH])
    label = tf.reshape(label, [h, w, 1])

    image = tf.math.divide(image, tf.math.reduce_max(image))

    image, label = VAL_REDUCE_RES(image, label)

    return image, label

def get_validation_dataset(files):
    """
    only variation is reduce function (will be pad_resize anyway)            
    """
    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(options)
    dataset = dataset.map(read_tfrecord_val, num_parallel_calls=AUTOTUNE)
    if IS_TPU: dataset = dataset.cache()
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_config_wandb(run_path):
    # restore file and read yaml as dict
    cfg_file = wandb.restore('config.yaml', run_path=run_path)
    cfg_y = yaml.load(cfg_file, Loader=yaml.FullLoader)
    cfg_file.close()
    
    cfg = {}  # create new dictionary
    
    # get only capital keys, cfg that you wrote
    for key in cfg_y.keys():
        if key.isupper():
            cfg[key] = cfg_y[key]['value']
    
    return cfg

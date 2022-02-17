import os
import re    # count tfrec
import gc    # deleting stuff
import json
import yaml

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
IS_OFF_AUG = 0
OFF_FILTER = ''



### DATALOADER ###
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, 
    # i.e. test10-687.tfrec = 687 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    
    return np.sum(n)

def get_filenames(split, ds_path, sub_path='', out=False):
    """
    fold_list : list of str of each fold
    """
    fns = []
    for fold in split:
        fol_path = os.path.join(ds_path, sub_path, f'{fold}_o{tr_cfg["ORIENT"]}*.tfrec')
        fold_fns  = tf.io.gfile.glob(fol_path)
        for fn in fold_fns:
            fns.append(fn)
    
    num_img = count_data_items(fns)
    steps = num_img//tr_cfg['BATCH_SIZE']

    if out:
        print(f'{split} files: {len(fns)} with {num_img} images')
    
    return fns, steps

def off_aug_selector(features):
    # take a filtered image with a random filter strength
    p_filter = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    if p_filter > .75:
        image = tf.io.parse_tensor(features[f"image3_{OFF_FILTER}"], tf.float32)
    elif p_filter > .5:
        image = tf.io.parse_tensor(features[f"image5_{OFF_FILTER}"], tf.float32)
    elif p_filter > .25:
        image = tf.io.parse_tensor(features[f"image7_{OFF_FILTER}"], tf.float32)
    else:
        image = tf.io.parse_tensor(features["image"], tf.float32)
    return image

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
    
    if IS_OFF_AUG:
        TFREC_FORMAT[f'image3_{OFF_FILTER}'] = tf.io.FixedLenFeature([], tf.string)
        TFREC_FORMAT[f'image5_{OFF_FILTER}'] = tf.io.FixedLenFeature([], tf.string)
        TFREC_FORMAT[f'image7_{OFF_FILTER}'] = tf.io.FixedLenFeature([], tf.string)
        features = tf.io.parse_single_example(feature, TFREC_FORMAT)
        image = off_aug_selector(features)
    else:
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

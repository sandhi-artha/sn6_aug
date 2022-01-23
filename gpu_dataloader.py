import os
import re    # count tfrec
# import gc    # deleting stuff
# import json
import yaml

import numpy as np
import tensorflow as tf
# import wandb
# from wandb.keras import WandbCallback

from gpu_aug import reduce_res, resize_example, aug_albu
# read raster
# read label
# reduce_res



### DATALOADER ###
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, 
    # i.e. test10-687.tfrec = 687 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    
    return np.sum(n)

def get_filenames(cfg, sub_path='', out=False):
    """
    fold_list : list of str of each fold
    """
    fns = []
    for fold in cfg['TRAIN_SPLITS']:
        fol_path = os.path.join(cfg['TRAIN_PATH'], sub_path, f'{fold}_o{cfg["ORIENT"]}*.tfrec')
        fold_fns  = tf.io.gfile.glob(fol_path)
        for fn in fold_fns:
            fns.append(fn)
    
    num_img = count_data_items(fns)
    steps = num_img//cfg['BATCH_SIZE']

    if out:
        print(f'{cfg["TRAIN_SPLITS"]} files: {len(fns)} with {num_img} images')
    
    return fns, steps

TFREC_FORMAT = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'data_idx': tf.io.VarLenFeature(tf.int64),
    'fn': tf.io.FixedLenFeature([], tf.string),
    'orient': tf.io.FixedLenFeature([], tf.int64),
}

AUTOTUNE = tf.data.experimental.AUTOTUNE

def read_tfrecord(feature):
    """data_idx is [r0,r1,c0,c1]
    """
    features = tf.io.parse_single_example(feature, TFREC_FORMAT)
    image = tf.io.parse_tensor(features["image"], tf.float32)
    label = tf.io.parse_tensor(features["label"], tf.bool)
    label = tf.cast(label, tf.float32)

    meta = {}
    meta['fn'] = features["fn"]
    meta['data_idx'] = tf.sparse.to_dense(features["data_idx"])
    meta['orient'] = features["orient"]
    
    return image, label, meta


def load_dataset(filenames, ordered=False):
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
    return dataset


def return_data(image, label, meta):
    """use only image and label as arg for training"""
    return image, label
    

def get_training_dataset(
    files, cfg,
    on_aug=True,
    shuffle=True,
    ordered=False):
    """
    train:
        - load_fn = 0
        - ext_val = 0
        - resize or crop

    repeat for several epochs
    augment only on train_ds
    shuffle before batch
    """
    dataset = load_dataset(files, ordered=ordered)  # [900,900]
    dataset = dataset.map(reduce_res, num_parallel_calls=AUTOTUNE)  # [320,320]
    dataset = dataset.map(return_data, num_parallel_calls=AUTOTUNE)
    if on_aug: dataset = dataset.map(aug_albu)
    dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(cfg['SHUFFLE_BUFFER'])  # 2000
    dataset = dataset.batch(cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_validation_dataset(files, cfg, cache=False):
    """
    val:
        - load_fn = 0
        - ext_val = maybe
        - resize only
    """
    dataset = load_dataset(files)
    dataset = dataset.map(return_data, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(resize_example, num_parallel_calls=AUTOTUNE)
    if cache: dataset = dataset.cache()
    dataset = dataset.batch(cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_preview_dataset(files, n_show, shuffle=False):
    """
    prev:
        - load_fn = yes
        - resize only
    """
    dataset = load_dataset(files, load_fn=1, ordered=1)
    dataset = dataset.map(lambda img,label,fn: resize_example(img,label,fn,IS_EXT_VAL),
                          num_parallel_calls=AUTOTUNE)
    if shuffle: dataset = dataset.shuffle(250)
    dataset = dataset.batch(n_show)
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
import os
import re    # count tfrec
# import gc    # deleting stuff
# import json
import yaml

import numpy as np
import tensorflow as tf
# import wandb
# from wandb.keras import WandbCallback




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



def read_tfrecord(feature):
    """data_idx is [r0,r1,c0,c1]
    """
    features = tf.io.parse_single_example(feature, TFREC_FORMAT)
    image = tf.io.parse_tensor(features["image"], tf.float32)
    label = tf.io.parse_tensor(features["label"], tf.bool)
    label = tf.cast(label, tf.float32)

    data_idx = tf.sparse.to_dense(features["data_idx"])
    h = data_idx[1] - data_idx[0]
    w = data_idx[3] - data_idx[2]
    
    image = tf.reshape(image, [h, w, IMAGE_CH])
    label = tf.reshape(label, [h, w, 1])

    # meta = {}
    # meta['fn'] = features["fn"]
    # meta['data_idx'] = tf.sparse.to_dense(features["data_idx"])
    # meta['orient'] = features["orient"]
    fn = features['fn']

    return image, label, fn#, meta


def remove_fn(img, label, fn):
    """removes meta as arg to not raise error when training"""
    return img, label

def load_dataset(filenames, load_fn=False, ordered=False):
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
    return dataset
    

def get_training_dataset(files, on_aug=True, shuffle=True, ordered=False):
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
    dataset = dataset.map(reduce_res_albu, num_parallel_calls=AUTOTUNE)  # [320,320]
    if on_aug: dataset = dataset.map(aug_albu)
    dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(tr_cfg['SHUFFLE_BUFFER'])  # 2000
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_validation_dataset(files, cache=False):
    """
    val:
        - load_fn = 0
        - ext_val = maybe
        - resize only
    """
    dataset = load_dataset(files)
    dataset = dataset.map(val_reduce_res_albu, num_parallel_calls=AUTOTUNE)
    if cache: dataset = dataset.cache()
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_preview_dataset(files, n_show, shuffle=False):
    """
    prev:
        - load_fn = yes
        - resize only
    """
    dataset = load_dataset(files, load_fn=1, ordered=1)
    dataset = dataset.map(val_reduce_res_albu, num_parallel_calls=AUTOTUNE)
    if shuffle: dataset = dataset.shuffle(tr_cfg['SHUFFLE_BUFFER'])
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



# Global variables
TFREC_FORMAT = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'data_idx': tf.io.VarLenFeature(tf.int64),
    'fn': tf.io.FixedLenFeature([], tf.string),
    'orient': tf.io.FixedLenFeature([], tf.int64),
}

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_CH = len(tr_cfg['SAR_CH'])
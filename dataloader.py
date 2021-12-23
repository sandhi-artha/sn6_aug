import os
import re    # count tfrec
import gc    # deleting stuff
import json

import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback



### KEEP TRACK - VOLATILE CHANGES
print(tf.__version__)
print(f'Wandb Version: {wandb.__version__}')







### GLOBAL VARIABLES ###
AUTOTUNE = tf.data.experimental.AUTOTUNE

# convert string and other types to bool for faster change
IS_EXT_VAL = 1 if tr_cfg['VAL_PATH'] == 'base-val-8' else 0
IMAGE_CH = len(tr_cfg['SAR_CH'])



def seed_everything(seed):
#     random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)



### DATALOADER ###
def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, 
    # i.e. test10-687.tfrec = 687 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    
    return np.sum(n)

def get_filenames(split, ds_path, sub_path='', out=False):
    if isinstance(split, list):
        fns = []
        for fold in split:
            fol_path = os.path.join(ds_path, sub_path, f'{fold}*.tfrec')
            fold_fns  = tf.io.gfile.glob(fol_path)
            for fn in fold_fns:
                fns.append(fn)
    else:
        fol_path = os.path.join(ds_path, sub_path, f'{split}*.tfrec')
        fns  = tf.io.gfile.glob(fol_path)
    
    num_img = count_data_items(fns)
    steps = num_img//tr_cfg['BATCH_SIZE']

    if out:
        print(f'{split} files: {len(fns)} with {num_img} images')
    
    return fns, steps

TFREC_FORMAT = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'fn': tf.io.FixedLenFeature([], tf.string)
}

def read_tfrecord(feature):
    features = tf.io.parse_single_example(feature, TFREC_FORMAT)
    fn = features["fn"]
    image = tf.io.parse_tensor(features["image"], tf.uint8)
    image = tf.cast(image, tf.float32)/255.0
    label = tf.io.parse_tensor(features["label"], tf.bool)
    label = tf.cast(label, tf.float32)
    
    return image, label, fn

def remove_fn(img, label, fn):
    """removes fn as arg to not raise error when training"""
    return img, label

def load_dataset(filenames, load_fn=False, ordered=False):
    """
    takes list of .tfrec files, read using TFRecordDataset,
    parse and decode using read_tfrecord func,
    returns : image, label
        both without shape and with IMAGE_DIM resolution
    if load_fn is True, returns: image, label, fn
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    if not load_fn:
        dataset = dataset.map(remove_fn, num_parallel_calls=AUTOTUNE)
    return dataset


    

def get_training_dataset(files, augment=False, shuffle=True):
    """
    train:
        - load_fn = 0
        - ext_val = 0
        - resize or crop

    repeat for several epochs
    augment only on train_ds
    shuffle before batch
    """
    dataset = load_dataset(files)  # [900,900]
    dataset = dataset.map(reduce_res, num_parallel_calls=AUTOTUNE)  # [640,640]
    if augment: dataset = dataset.map(data_augment)
    dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(tr_cfg['SHUFFLE_BUFFER'])  # 2000
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_validation_dataset(files):
    """
    val:
        - load_fn = 0
        - ext_val = maybe
        - resize only
    """
    dataset = load_dataset(files)
    dataset = dataset.map(lambda img,label: resize_example(img,label,None,IS_EXT_VAL),
                          num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_preview_dataset(files, n_show, shuffle=False):
    """
    prev:
        - load_fn = maybe
        - ext_val = maybe
        - resize only
    """
    dataset = load_dataset(files, load_fn=1)
    dataset = dataset.map(lambda img,label,fn: resize_example(img,label,fn,IS_EXT_VAL),
                          num_parallel_calls=AUTOTUNE)
    if shuffle: dataset = dataset.shuffle(250)
    dataset = dataset.batch(n_show)
    return dataset












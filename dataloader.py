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


### LOAD CONFIG ###
if os.path.isfile('tr_cfg.json'):
    print('using Kaggle config')
    with open('tr_cfg.json', 'r') as fp:
        tr_cfg = json.load(fp)
else:
    from cfg import tr_cfg
    print('using saved config')






### GLOBAL VARIABLES ###
AUTOTUNE = tf.data.experimental.AUTOTUNE

# convert string and other types to bool for faster change
IS_EXT_VAL = 1 if tr_cfg['VAL_PATH'] == 'base-val-8' else 0
if tr_cfg['ORIENT'] == 1: tr_cfg['IS_ROT_ORIENT0'] = 0  # make sure not to rotate when loading orient1
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
    """
    fold is complete with orient if 'ORIENT'==2. ex: fold0_o1
    """
    if tr_cfg["ORIENT"] == 2:
        orient = ''
    else:
        orient = f'_o{tr_cfg["ORIENT"]}'
    if isinstance(split, list):
        fns = []
        for fold in split:
            fol_path = os.path.join(ds_path, sub_path, f'{fold}{orient}*.tfrec')
            fold_fns  = tf.io.gfile.glob(fol_path)
            for fn in fold_fns:
                fns.append(fn)
    else:
        fol_path = os.path.join(ds_path, sub_path, f'{split}{orient}*.tfrec')
        fns  = tf.io.gfile.glob(fol_path)
    
    num_img = count_data_items(fns)
    steps = num_img//tr_cfg['BATCH_SIZE']

    if out:
        print(f'{split} files: {len(fns)} with {num_img} images')
    
    return fns, steps

TFREC_FORMAT = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'fn': tf.io.FixedLenFeature([], tf.string),
    'orient': tf.io.FixedLenFeature([], tf.int64)
}

def read_tfrecord(feature):
    features = tf.io.parse_single_example(feature, TFREC_FORMAT)
    image = tf.io.parse_tensor(features["image"], tf.uint8)
    image = tf.cast(image, tf.float32)/255.0
    label = tf.io.parse_tensor(features["label"], tf.bool)
    label = tf.cast(label, tf.float32)

    orient = features["orient"]
    fn = features["fn"]

    return image, label, orient, fn

def remove_args(img, label, orient, fn):
    """removes fn as arg to not raise error when training
    also removes orient based on dataset scope variable"""
    if tr_cfg['IS_ROT_ORIENT0']:
        return img, label, orient
    else:
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
        dataset = dataset.map(remove_args, num_parallel_calls=AUTOTUNE)
    return dataset

def orient_proc(image, label, orient, fn=None):
    """check orient, if it's 0 then rotate image by 180
        can't just use if not orient, unless do orient.numpy()
            but potentially hurt performance
        TODO: incorporate IS_EXT_VAL
    """
    orient = tf.cast(orient, dtype=tf.bool)
    label = tf.expand_dims(label, axis=-1)   # to rotate image, must be 3d
    image = tf.cond(tf.equal(orient, tf.constant(False)),
        lambda: tf.image.rot90(image, k=2),  # cond if true
        lambda: image                        # cond if false
    )

    label = tf.cond(tf.equal(orient, tf.constant(False)),
        lambda: tf.image.rot90(label, k=2),  # cond if true
        lambda: label                        # cond if false
    )

    if fn is None:
        return image, label
    else:
        return image, label, fn
    

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
    if tr_cfg['IS_ROT_ORIENT0']:
        dataset = dataset.map(orient_proc, num_parallel_calls=AUTOTUNE)     # rotate 180 orient0
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
    if tr_cfg['IS_ROT_ORIENT0']:
        dataset = dataset.map(orient_proc, num_parallel_calls=AUTOTUNE)     # rotate 180 orient0
    dataset = dataset.map(lambda img,label: resize_example(img,label,None,IS_EXT_VAL),
                          num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def remove_orient(img, label, orient, fn):
    return img, label, fn


def get_preview_dataset(files, n_show, shuffle=False):
    """
    prev:
        - load_fn = maybe
        - ext_val = maybe
        - resize only
    """
    dataset = load_dataset(files, load_fn=1)
    if tr_cfg['IS_ROT_ORIENT0']:
        dataset = dataset.map(orient_proc, num_parallel_calls=AUTOTUNE)     # rotate 180 orient0
    else:
        dataset = dataset.map(remove_orient, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda img,label,fn: resize_example(img,label,fn,IS_EXT_VAL),
                          num_parallel_calls=AUTOTUNE)
    if shuffle: dataset = dataset.shuffle(250)
    dataset = dataset.batch(n_show)
    return dataset












import os
import re    # count tfrec
import gc    # deleting stuff
import json
import yaml
import random

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
TFREC_FORMAT = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'data_idx': tf.io.VarLenFeature(tf.int64),
    'fn': tf.io.FixedLenFeature([], tf.string),
    'orient': tf.io.FixedLenFeature([], tf.int64),
}

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, 
    # i.e. test10-687.tfrec = 687 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

class Reader():
    def __init__(
        self, image_ch, reduce_fun, load_fn=False,
        aug_albu_fun=None, aug_tf_fun=None
    ):
        """reads tfrecord"""
        self.image_ch = image_ch
        self.reduce_fun = reduce_fun
        self.load_fn = load_fn
        self.aug_albu_fun = aug_albu_fun
        self.aug_tf_fun = aug_tf_fun

    def read(self, feature):
        # read tfrecord features
        features = tf.io.parse_single_example(feature, TFREC_FORMAT)
        image = tf.io.parse_tensor(features['image'], tf.float32)
        label = tf.io.parse_tensor(features['label'], tf.bool)
        label = tf.cast(label, tf.float32)

        data_idx = tf.sparse.to_dense(features['data_idx'])
        h = data_idx[1] - data_idx[0]
        w = data_idx[3] - data_idx[2]
        
        image = tf.reshape(image, [h, w, self.image_ch])
        label = tf.reshape(label, [h, w, 1])

        # normalize so max value is 1.0
        image = tf.math.divide(image, tf.math.reduce_max(image))

        if self.aug_albu_fun:
            image, label = self.aug_albu_fun(image, label, h, w, self.image_ch)

        image, label = self.reduce_fun(image, label)

        if self.aug_tf_fun:
            image, label = self.aug_tf_fun(image, label)

        if self.load_fn:
            return image, label, features['fn']
        else:
            return image, label


class DataLoader():
    """
    1. load ds for train (augments)
    2. load ds for val (no augments)
    3. preview train ds (ordered, no shuffle, augments)
    4. view results (n_img)
    """
    def __init__(
        self, cfg, train_reduce, val_reduce,
        aug_albu_fun, aug_tf_fun,
        # load_fn, ordered, shuffle
    ):
        self.cfg = cfg
        self.image_ch = len(cfg.SAR_CH)
        self.train_reduce = train_reduce
        self.val_reduce = val_reduce
        self.aug_albu_fun = aug_albu_fun
        self.aug_tf_fun = aug_tf_fun
        self.train_fns = []
        self.val_fns = []

    def preview_train_ds(self):
        """load some images and view augmented results"""
        pass
            

    def get_train_ds(self, train_fns):
        """
        takes list of .tfrec files, read using TFRecordDataset,
        parse and decode using read_tfrecord func,
        returns : image, label
            both without shape and with IMAGE_DIM resolution
        ordered=True will read each files in same order.
        """
        train_reader = Reader(
            self.image_ch, self.train_reduce, self.aug_albu_fun, self.aug_tf_fun)

        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = tf.data.TFRecordDataset(train_fns, num_parallel_reads=AUTOTUNE) \
            .with_options(options) \
            .map(train_reader.read, num_parallel_calls=AUTOTUNE) \
            .repeat() \
            .shuffle(self.cfg.SHUFFLE_BUFFER) \
            .batch(self.cfg.BATCH_SIZE) \
            .prefetch(AUTOTUNE)
        return ds

    def get_val_ds(self, val_fns):
        """
        only variation is reduce function (will be pad_resize anyway)            
        """
        val_reader = Reader(self.image_ch, self.val_reduce)
        options = tf.data.Options()
        options.experimental_deterministic = False
        ds = tf.data.TFRecordDataset(val_fns, num_parallel_reads=AUTOTUNE) \
            .with_options(options) \
            .map(val_reader.read, num_parallel_calls=AUTOTUNE) \
            .batch(self.cfg.BATCH_SIZE) \
            .prefetch(AUTOTUNE)
        return ds

    def load_data(self):
        train_fns, train_steps = get_filenames(
            self.cfg, self.cfg.TRAIN_SPLITS, self.cfg.TRAIN_PATH)
        val_fns, val_steps = get_filenames(
            self.cfg, self.cfg.VAL_SPLITS, self.cfg.VAL_PATH)
        train_ds = self.get_train_ds(train_fns)
        val_ds = self.get_val_ds(val_fns)
        return train_ds, train_steps, val_ds, val_steps


def get_filenames(cfg, splits, ds_path, off_ds_path='', verbose=False):
        """
        collect tfrecord filenames for selected splits (folds)
        splits : list
            contain what folds to use during training or val
        ds_path : str
            path to train or val dataset
        off_ds_path: str
            path to speckle filtered dataset, will load along normal dataset
        """
        fns = []  # will become ['foldx_ox-yy-zz.tfrec', ...]
        for fold in splits:
            fol_path = os.path.join(ds_path, f'{fold}_o{cfg.ORIENT}*.tfrec')
            fold_fns  = tf.io.gfile.glob(fol_path)
            for fn in fold_fns:
                fns.append(fn)

            if off_ds_path:
                fol_path = os.path.join(off_ds_path, f'{fold}_o{cfg.ORIENT}*.tfrec')
                fold_fns  = tf.io.gfile.glob(fol_path)
                for fn in fold_fns:
                    fns.append(fn)
        
        random.shuffle(fns)

        num_img = count_data_items(fns)
        steps = num_img//cfg.BATCH_SIZE

        if verbose:
            print(f'{splits} files: {len(fns)} with {num_img} images')
        
        return fns, steps








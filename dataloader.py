# returns dataset loaded from a given path to tfrecords
# mod by config
# for train and val
import os
import json
import re

import numpy as np
import tensorflow as tf

### GLOBAL CFG ###
if os.path.isfile('tr_cfg.json'):
    print('loading new config')
    with open('tr_cfg.json', 'r') as fp:
        tr_cfg = json.load(fp)
else:
    from cfg import tr_cfg
    print('using saved config')





def seed_everything(seed):
#     random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

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



def random_crop(image, label):
    """image and mask must be same dtype"""
    # make [900,900,4]
    size = [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']]
    stacked_image = tf.concat([image, label], axis=-1)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[*size, 4])
    
    image_crop = tf.reshape(image, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS'], IMAGE_CH])
    label_crop = tf.reshape(label, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS'], 1])
    # for label, if you want [w,h,1] shape, use -1:
    return cropped_image[:,:,:3], cropped_image[:,:,-1:]

def read_tfrecord(feature, ext_val):
    features = tf.io.parse_single_example(feature, TFREC_FORMAT)
    # decode image and label
    image = tf.io.parse_tensor(features["image"], tf.uint8)
    image = tf.cast(image, tf.float32)/255.0
#     image = tf.reshape(image, [tr_cfg['IMAGE_DIM'], tr_cfg['IMAGE_DIM'], IMAGE_CH])
#     image = tf.image.resize(image, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    
    label = tf.io.parse_tensor(features["label"], tf.bool)
    label = tf.cast(label, tf.float32)

    if ext_val:
        label = tf.reshape(label, [640, 640, 1])
    else:
        label = tf.reshape(label, [tr_cfg['IMAGE_DIM'], tr_cfg['IMAGE_DIM'], 1])

#     label = tf.image.resize(label, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])

    image, label = random_crop(image, label)
   # fn = features["fn"]
    
    return image, label#, fn


def load_dataset(filenames, fp32=True, ext_val=False, ordered=False):
    """
    takes list of .tfrec files, read using TFRecordDataset,
    parse and decode using read_tfrecord func, returns image&label/image_name(test)
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda x: read_tfrecord(x, ext_val),
                          num_parallel_calls=AUTOTUNE)
    return dataset

def get_training_dataset(files, augment=True, shuffle=True):
    dataset = load_dataset(files, fp32=tr_cfg['IS_FP32'])
    if augment: dataset = dataset.map()
    dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(2000)  # 2000
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset

def get_validation_dataset(files):
    dataset = load_dataset(files, fp32=tr_cfg['IS_FP32'], ext_val=IS_EXT_VAL)
    dataset = dataset.batch(tr_cfg['BATCH_SIZE'])
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE)
    
    return dataset


if __name__=='__main__':
    # all variables in this script will be available in notebook!
    # but you need to find a way for notebook variables to show in script
    # one way is using the cfg!

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'fn': tf.io.FixedLenFeature([], tf.string)
    }
    
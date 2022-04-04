# read raster, label
# create tfrecord
import os
import json
import glob
import pickle

import numpy as np
import rasterio as rs
from rasterio import features as feat
import geopandas as gpd
import tensorflow as tf
import matplotlib.pyplot as plt

# add sn6_aug folder to PYTHONPATH env variable
# abspath of __file__ run from sn6_aug dir is '/root/sn6_aug/datasets/gpu_datagen.py'
#   it doesn't matter where you run, bcz it uses abspath
# this will add '/root/sn6_aug' to PYTHONPATH and you can do: from solaris.base import Evaluator
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
print(os.path.abspath(__file__))
print(BASE_PATH)

from datasets.dg_cfg import dg_cfg
from lib.raster import get_data_region_idx
from lib.proc import to_hwc
from lib.viz import show_hist, show_stats




### GET DATA ###
def get_label_path(raster_path):
    vector_path = raster_path
    vector_path = vector_path.replace(MODE, 'geojson_buildings',1)
    vector_path = vector_path.replace(MODE, 'Buildings')
    vector_path = vector_path.replace('.tif', '.geojson')
    return vector_path

def get_image_label(raster_path, ch=None):
    # get raster and info for label
    raster = rs.open(raster_path)
    h = raster.height  # rows
    w = raster.width   # cols
    transform = raster.transform

    # read image
    image = raster.read(indexes=ch)
    image = to_hwc(image)

    # read label
    vector = gpd.read_file(get_label_path(raster_path))
    if vector.shape[0]==0:  # handle when no buildings are in the tile
        mask = np.zeros((h,w),dtype=bool)
    else:
        mask = feat.geometry_mask(
            vector.geometry,
            out_shape=(h,w),
            transform=transform,
            invert=True  # pixel buildings == 1
        )

    # crop nodata
    r0,r1,c0,c1 = get_data_region_idx(raster)

    image = image[r0:r1, c0:c1, :]
    mask = mask[r0:r1, c0:c1]
    data_region_idx = [r0,r1,c0,c1]

    return image, mask, data_region_idx

### TFRECORD GENERATOR ###
def serialize_image(image, out_precision=32):
    """
    image : np.array
        image with pixel range 0-1
    out_precision : int
        8, 16 or 32 for np.uint8, np.uint16 or np.float32
        for float32, nan will be replaced by 0.0
        for uint, casting auto converts nan to 0.0
    """
    if out_precision==8:
        dtype = tf.uint8
        image = tf.cast(image*(2**8 - 1), dtype=dtype)
    elif out_precision==16:
        dtype = tf.bfloat16
        image = tf.cast(np.nan_to_num(image, nan=0.0), dtype=dtype)
    else:
        dtype = tf.float32
        image = tf.cast(np.nan_to_num(image, nan=0.0), dtype=dtype)

    image_tensor = tf.constant(image, dtype=dtype)
    image_serial = tf.io.serialize_tensor(image_tensor)
    return image_serial

def serialize_label(label):
    """
    label : np.array
    """
    label_tensor = tf.constant(label, dtype=tf.bool)
    label_serial = tf.io.serialize_tensor(label_tensor)
    return label_serial

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor. intended for the image data
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def create_tfrecord(raster_paths, cfg, base_fn, orient):
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

                # write tfrecords
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


### CONFIG ###
def add_base_path(fns_folds):
    """add base_path to the saved filenames"""
    fps_folds = []
    for fns_fold in fns_folds:
        fps_fold = []
        for fn in fns_fold:
            if MODE == 'PAN':
                fn = fn.replace('SAR-Intensity', 'PAN')
            fps_fold.append(os.path.join(dg_cfg['base_dir'], MODE, fn))
        fps_folds.append(fps_fold)
    return fps_folds

def datagen_orient(orient):
    """generate tfrecords for a given orient"""
    # read filenames for each folds
    with open(f'fps{orient}_5folds.pickle', 'rb') as f:
        fns_folds = pickle.load(f)
    
    fps_folds = add_base_path(fns_folds)
    print(fps_folds[0][0])

    print(f'orient {orient}')
    if not os.path.isdir(dg_cfg['out_dir']):
        os.makedirs(dg_cfg['out_dir'])

    for i, fps_fold in enumerate(fps_folds):
        print(f'creating tfrecords for fold {i}')
        out_path = os.path.join(dg_cfg['out_dir'], f'fold{i}_o{orient}')
        
        create_tfrecord(fps_fold, dg_cfg, out_path, orient)

def test_tfrec(show=4):
    TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'data_idx': tf.io.VarLenFeature(tf.int64),
        'fn': tf.io.FixedLenFeature([], tf.string),
        'orient': tf.io.FixedLenFeature([], tf.int64),
    }

    def _read_tfrecord(feature):
        features = tf.io.parse_single_example(feature, TFREC_FORMAT)
        image = tf.io.parse_tensor(features["image"], tf.float32)
        label = tf.io.parse_tensor(features["label"], tf.bool)
        label = tf.cast(label, tf.float32)

        data_idx = tf.sparse.to_dense(features["data_idx"])
        h = data_idx[1] - data_idx[0]
        w = data_idx[3] - data_idx[2]
        image = tf.reshape(image, [h, w, len(dg_cfg['channel'])])
        label = tf.reshape(label, [h, w, 1])

        image = tf.math.divide(image, tf.math.reduce_max(image))
        fn = features['fn']

        return image, label, fn

    filenames = glob.glob(os.path.join(dg_cfg['out_dir'], f'*.tfrec'))
    filename = filenames[0]
    print(f'loading {filename}')
    ds = tf.data.TFRecordDataset([filename])
    ds = ds.map(_read_tfrecord)

    for img, label, fn in ds.take(show):
        f,[ax1,ax2,ax3] = plt.subplots(1,3,figsize=(9,3))
        ax1.imshow(img.numpy()[:,:,0])
        ax2.imshow(label.numpy()[:,:,0])
        show_hist(img.numpy(),ax=ax3)
        plt.show()
    
    show_stats(img.numpy())
    print(fn.numpy())

if __name__=='__main__':
    if os.path.isfile('dg_cfg.json'):
        print('using Kaggle config')
        with open('dg_cfg.json', 'r') as fp:
            dg_cfg = json.load(fp)
    else:
        print('using saved config')

    if dg_cfg['mode'] == 'sar':
        MODE = 'SAR-Intensity'
    elif dg_cfg['mode'] == 'pan':
        MODE = 'PAN'
    else:
        print('Invalid dataset, use "pan" or "sar"')
        sys.exit()

    if dg_cfg['orient'] == 0 or dg_cfg['orient'] == 2:
        datagen_orient(0)
    if dg_cfg['orient'] == 1 or dg_cfg['orient'] == 2:
        datagen_orient(1)
    
    test_tfrec()
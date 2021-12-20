# create tfrecords for selected config
# based on https://www.kaggle.com/sandhiwangiyana/sn6-competition-data
import os
import glob
import json

import numpy as np
from sklearn.cluster import KMeans
import rasterio as rs
from rasterio import features as feat
import geopandas as gpd
import tensorflow as tf

from cfg import dg_cfg
from lib.raster import get_tile_bounds
from lib.proc import to_hwc, normalize



### TILE SELECTOR ###
def get_ts_orient(base_dir, orient):
    """reads SAR_orientations
    return list of timestamps with orientation 0 or 1
    """
    orient_fp = os.path.join(base_dir, 'SummaryData/SAR_orientations.txt')
    with open(orient_fp, 'rb') as f:
        timsestamps = []
        for i,ts in enumerate(f):
            ts_split = ts.decode().split(' ')
            if str(orient) in ts_split[1]:
                timsestamps.append(ts_split[0])
    print(f'total timestamps: {i+1}')
    print(f'timestamps with orientation {orient}: {len(timsestamps)}')
    return timsestamps

def get_fp_orient(base_dir, orient):
    """reads SAR_orientations
    return list of sar file paths with orientation 0 or 1
    """
    timestamps = get_ts_orient(base_dir, orient)
    fps = []
    for ts in timestamps:
        # get list of file_paths for every timestamps
        _fps = glob.glob(f'{base_dir}/SAR-Intensity/*{ts}*.tif')
        for fp in _fps:
            fps.append(fp)

    print(f'total SAR images with orientation {orient}: {len(fps)}')
    return fps

def get_fps_folds(fps, folds=4):
    """fps : list
        filepaths of rasters
    folds: int
        how many splits to create
    returns: list of fps for every fold
    """
    sar_bounds = get_tile_bounds(fps)

    # mid = top-bot/2 + bot
    mid = [((b[3]-b[1])/2) + b[1] for b in sar_bounds]
    # left for 2d clustering
    # left = [b[0] for b in sar_bounds]

    mid_arr = np.array(mid)
    mid_arr = np.expand_dims(mid_arr, axis=-1)

    # create n folds dataset
    kmeans = KMeans(n_clusters=folds)
    y_kmeans = kmeans.fit_predict(mid_arr)
    
    print('Cluster distribution:')
    print(np.unique(y_kmeans, return_counts=1))

    idxs_folds = []
    # get indexes belonging to a cluster
    for cluster in range(folds):
        s = np.argwhere(y_kmeans == cluster)
        idxs_folds.append(s)

    return idxs_folds


### GET DATA ###


def get_image(raster_path, ch=None):
    """
    ch = list or int
        starts at 1, if None, return all channels
    returns: np.array
        type same as raster (float32), range [0,1]
    """
    raster = rs.open(raster_path)
    image = raster.read(indexes=ch, masked=True)
    image = to_hwc(image)
    image = normalize(image)
    raster.close()  # close the opened dataset
    return image

def get_label_path(raster_path):
    vector_path = raster_path
    vector_path = vector_path.replace('SAR-Intensity', 'geojson_buildings',1)
    vector_path = vector_path.replace('SAR-Intensity', 'Buildings')
    vector_path = vector_path.replace('.tif', '.geojson')
    return vector_path

def get_label(raster_path):
    """
    returns : np.array type: bool
        mask array where pixel buildings=1 and background=0
    """
    raster = rs.open(raster_path)
    h = raster.height  # rows
    w = raster.width   # cols
    transform = raster.transform
    
    raster.close()  # close the opened dataset

    vector = gpd.read_file(get_label_path(raster_path))
    
    # handle when no buildings are in the tile
    if vector.shape[0]==0:
        mask = np.zeros((h,w),dtype=bool)
    else:
        mask = feat.geometry_mask(
            vector.geometry,
            out_shape=(h,w),
            transform=transform,
            invert=True  # pixel buildings == 1
        )

    return mask



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

def create_tfrecord(raster_paths, cfg, base_fn):
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
                image = get_image(raster_paths[idx], cfg['channel'])
                image_serial = serialize_image(image, cfg['out_precision'])

                label = get_label(raster_paths[idx])
                label_serial = serialize_label(label)

                fn = os.path.basename(raster_paths[idx]).split('.')[0]

                feature = {
                    'image': _bytes_feature(image_serial.numpy()),
                    'label': _bytes_feature(label_serial.numpy()),
                    'fn' : _bytes_feature(tf.compat.as_bytes(fn))
                }

                # write tfrecords
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())




if __name__=='__main__':
    # must be in sn6_aug folder
    # read config dictionary from kaggle script
    if os.path.isfile('dg_cfg.json'):
        print('loading new config')
        with open('dg_cfg.json', 'r') as fp:
            dg_cfg = json.load(fp)
    else:
        print('using saved config')
    
    fps = get_fp_orient(dg_cfg['base_dir'], dg_cfg['orient'])
    idxs_folds = get_fps_folds(fps, dg_cfg['folds'])

    if not os.path.isdir(dg_cfg['out_dir']):
        os.makedirs(dg_cfg['out_dir'])

    for i, idxs_fold in enumerate(idxs_folds):
        print(f'creating tfrecords for fold {i}')
        fps_fold = [fps[idx[0]] for idx in idxs_fold]
        out_path = os.path.join(dg_cfg['out_dir'], f'fold{i}')
        
        create_tfrecord(fps_fold, dg_cfg, out_path)

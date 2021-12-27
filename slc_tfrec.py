import os
import glob

import numpy as np
import rasterio as rs
from rasterio import features as feat
import geopandas as gpd
import tensorflow as tf

from datagen import serialize_image, serialize_label, _bytes_feature
from slc_cfg import slc_cfg
from lib.proc import to_hwc, normalize, hist_clip

def get_slc_label_path(raster_path):
    vector_path = raster_path
    vector_path = vector_path.replace('raster', 'vector',1)
    vector_path = vector_path.replace('.tif', '.geojson')
    return vector_path

def get_image(raster_path, ch=None):
    """
    ch = list or int
        starts at 1, if None, return all channels
    returns: np.array
        type same as raster (float32), range [0,1]
    """
    raster = rs.open(raster_path)
    image = raster.read(indexes=ch, masked=True)
    image = hist_clip(image, 40, clip_high=0)
    image = to_hwc(image)
    image = normalize(image)
    raster.close()  # close the opened dataset
    return image

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

    vector = gpd.read_file(get_slc_label_path(raster_path))
    
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
    raster_save_path = os.path.join(slc_cfg['out_dir'],'raster', '*.tif')
    raster_fps = glob.glob(raster_save_path)

    out_dir = os.path.join(slc_cfg['out_dir'], 'tfrec')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    out_path = os.path.join(out_dir, 'val')
    create_tfrecord(raster_fps, slc_cfg, out_path)
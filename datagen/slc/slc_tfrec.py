import os
import glob

import numpy as np
import rasterio as rs
from rasterio import features as feat
import geopandas as gpd
import tensorflow as tf

from gpu_datagen import _bytes_feature, _int64_feature
from gpu_datagen import serialize_image, serialize_label
from slc_cfg import slc_cfg
from lib.proc import to_hwc, normalize, hist_clip
from lib.raster import get_data_region_idx

def get_slc_label_path(raster_path):
    vector_path = raster_path
    vector_path = vector_path.replace('raster', 'vector',1)
    vector_path = vector_path.replace('.tif', '.geojson')
    return vector_path


def get_image_label(raster_path, ch=None):
    # get raster and info for label
    raster = rs.open(raster_path)
    h = raster.height  # rows
    w = raster.width   # cols
    transform = raster.transform

    # read image, clip and remove negative to 10^-5
    image = raster.read(indexes=ch, masked=1)
    image = hist_clip(image, 40, clip_high=0)
    image = image - np.min(image) + 1e-5
    image = to_hwc(image)

    # read label
    vector = gpd.read_file(get_slc_label_path(raster_path))
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


if __name__=='__main__':
    raster_save_path = os.path.join(slc_cfg['out_dir'],'raster', '*.tif')
    raster_fps = glob.glob(raster_save_path)

    orient = slc_cfg['orient']

    out_dir = os.path.join(slc_cfg['out_dir'], 'tfrec')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    out_path = os.path.join(out_dir, f'val_o{orient}')
    create_tfrecord(raster_fps, slc_cfg, out_path, orient)
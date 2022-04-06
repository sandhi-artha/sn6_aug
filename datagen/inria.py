import os, sys
BASE_PATH = os.path.dirname(os.path.abspath(''))
sys.path.append(BASE_PATH)


import json, glob, random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from lib.viz import show_stats, show_hist

cfg = {
    'base_dir'      : 'D:/Projects/python/dataset/inria',
    'train_city'    : ['austin','chicago'],
    'n_train'       : 30,
    'val_city'      : ['vienna'],
    'n_val'         : 10,
    'out_dir'       : 'test',
    'tile_size'     : (1000,1000),
    'tfrec_size'    : 50,
}

class Dict2Obj(object):
    """Turns a dictionary into a class"""
    def __init__(self, dictionary):
        for key in dictionary: setattr(self, key, dictionary[key])

def get_city_fps(base_dir, city_list, n_tile):
    fps = []
    for city in city_list:
        city_fps = glob.glob(f'{base_dir}/images/{city}*.tif')
        random.shuffle(city_fps)
        for city_fp in city_fps[:n_tile]:
            fps.append(city_fp)
    return fps

def tile_image(out_dir, fp):
    w = 1000
    h = 1000
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with Image.open(fp) as img:
        fn = os.path.basename(fp).split('.')[0]
        image = np.array(img.convert('L'))
        # tile into 25
        count = 0
        for i in range(5):
            for j in range(5):
                im = Image.fromarray(image[i*w:(i+1)*w, j*h:(j+1)*h])
                im.save(f'{out_dir}/{fn}_{str(count).zfill(2)}.png')
                count += 1

def get_serialized_image_label(raster_path):
    """returns image and label in tf serial tensor
    """
    raster = Image.open(raster_path)
    image = np.array(raster.convert('L'))   # to grayscale
    raster.close()

    # data_region_idx not required here, but used just to comply with code from SN6
    h = image.shape[0]
    w = image.shape[1]
    data_region_idx = [0,h,0,w]

    # serialize image
    image_tensor = tf.constant(image, dtype=tf.float32)
    image_serial = tf.io.serialize_tensor(image_tensor)

    # read label
    raster_label = Image.open(raster_path.replace('images','gt'))
    label = np.array(raster_label.convert('1'))   # to bool
    raster_label.close()

    # serialize label
    label_tensor = tf.constant(label, dtype=tf.bool)
    label_serial = tf.io.serialize_tensor(label_tensor)
    return image_serial, label_serial, data_region_idx

def create_tfrecord(raster_paths, size, base_fn):
    """
    image in float32 serialized
    mask in binary serialized
    size : int
    output : tfrecord with filenames: '{base_fn}01-{size}.tfrec'
    """
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor. intended for the image data
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    tot_ex = len(raster_paths)  # total examples
    tot_tf = int(np.ceil(tot_ex/size))  # total tfrecords

    for i in range(tot_tf):
        print(f'Writing TFRecord {i} of {tot_tf}..')
        size2 = min(size, tot_ex - i*size)  # size=size2 unless for remaining in last file
        fn = f'{base_fn}-{i:02}-{size2}.tfrec'

        with tf.io.TFRecordWriter(fn) as writer:
            for j in range(size2):
                idx = i*size+j  # ith tfrec * num_img per tfrec as the start of this iteration
                image, label, data_region_idx = get_serialized_image_label(
                    raster_paths[idx])

                fn = os.path.basename(raster_paths[idx]).split('.')[0]

                feature = {
                    'image': _bytes_feature(image.numpy()),
                    'label': _bytes_feature(label.numpy()),
                    'data_idx': _int64_feature(data_region_idx),
                    'fn' : _bytes_feature(tf.compat.as_bytes(fn)),
                    'orient': _int64_feature([0]),
                }

                # write tfrecords
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

def test_tfrec(out_dir, show=4):
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
        image = tf.reshape(image, [h, w, 1])
        label = tf.reshape(label, [h, w, 1])

        image = tf.math.divide(image, tf.math.reduce_max(image))
        fn = features['fn']

        return image, label, fn

    filenames = glob.glob(os.path.join(out_dir, f'*.tfrec'))
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




if __name__ == '__main__':
    if os.path.isfile('cfg.json'):
        print('using Kaggle config')
        with open('cfg.json', 'r') as fp:
            cfg = json.load(fp)
    else:
        print('using saved config')
    
    cfg = Dict2Obj(cfg)

    train_fps = get_city_fps(cfg.train_city, cfg.n_train)
    val_fps = get_city_fps(cfg.val_city, cfg.n_val)

    print('Tiling..')
    for train_fp in train_fps:
        tile_image(f'{cfg.out_dir}/train/images', train_fp)
        tile_image(f'{cfg.out_dir}/train/gt', train_fp.replace('images','gt'))

    for val_fp in val_fps:
        tile_image(f'{cfg.out_dir}/val/images', val_fp)
        tile_image(f'{cfg.out_dir}/val/gt', val_fp.replace('images','gt'))
    
    print('Creating TFRecord..')
    train_tile_fps = glob.glob(f'{cfg.out_dir}/train/images/*.tif')
    val_tile_fps = glob.glob(f'{cfg.out_dir}/val/images/*.tif')
    print(f'train: {len(train_tile_fps)}, val: {len(val_tile_fps)}')

    create_tfrecord(train_tile_fps, cfg.tfrec_size, f'{cfg.out_dir}/train_o0')
    create_tfrecord(val_tile_fps, cfg.tfrec_size, f'{cfg.out_dir}/val_o0')

    test_tfrec(cfg.out_dir)
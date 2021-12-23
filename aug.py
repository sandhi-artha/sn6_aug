# use bayesian
# only to import modules to train.py
# TODO: implement augmentations for both image and mask (use concat)
import os
import json
import tensorflow as tf
# from tensorflow.keras import backend as K
# import math


def resize_example(image, label, fn=None, ext_val=False):
    """image must be type float"""
    if ext_val:
        # if ext validation (from SLC) it takes different res input
        image = tf.reshape(image, [640, 640, 3])
        label = tf.reshape(label, [640, 640, 1])
    else:
        # reshape here is needed bcz from parse_tensor, image and label don't have shape
        image = tf.reshape(image, [tr_cfg['IMAGE_DIM'], tr_cfg['IMAGE_DIM'], IMAGE_CH])
        label = tf.reshape(label, [tr_cfg['IMAGE_DIM'], tr_cfg['IMAGE_DIM'], 1])

    image = tf.image.resize(image, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    label = tf.image.resize(label, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    
    if fn is None:
        return image, label
    else:
        return image, label, fn

def random_crop(image, label):
    """image and mask must be same dtype"""
    size = [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']]
    image = tf.reshape(image, [tr_cfg['IMAGE_DIM'], tr_cfg['IMAGE_DIM'], IMAGE_CH])
    label = tf.reshape(label, [tr_cfg['IMAGE_DIM'], tr_cfg['IMAGE_DIM'], 1])
    stacked_image = tf.concat([image, label], axis=-1)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[*size, 4])
    
    # for label, if you want [w,h,1] shape, use -1:
    return cropped_image[:,:,:3], cropped_image[:,:,-1:]

def reduce_res(image, label):
    """either random crop or resize
    if using both, when p_reduce > .5 do resize, if lower do crop"""

    if ag_cfg['IS_RESIZE'] and not ag_cfg['IS_CROP']: # 100% resize
        image, label = resize_example(image, label)
    if ag_cfg['IS_CROP'] and not ag_cfg['IS_RESIZE']:   # 100% random crop
        image, label = random_crop(image, label)
    if ag_cfg['IS_RESIZE'] and ag_cfg['IS_CROP']:
        # 50-50 chance of resize and random crop
        p_reduce = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_reduce > .5:
            image, label = resize_example(image, label)
        else:
            image, label = random_crop(image, label)

    if not ag_cfg['IS_RESIZE'] and not ag_cfg['IS_CROP']:
        raise ValueError('must choose RESIZE or CROP or both')

    return image, label

# random bayessian control from https://www.kaggle.com/dimitreoliveira/flower-with-tpus-advanced-augmentations
def data_augment(image, label):
    
    # Flips
    if ag_cfg['IS_VFLIP']:
        print('vflip')
        p_vflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_vflip >= .5:
            image = tf.image.random_flip_up_down(image)
    
    if ag_cfg['IS_HFLIP']:
        print('hflip')
        p_hflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_hflip >= .5:
            image = tf.image.random_flip_left_right(image)
        
    # Rotates
    if ag_cfg['IS_ROT']:
        print('rotates')
        p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_rotate > .75:
            image = tf.image.rot90(image, k=3) # rotate 270º
        elif p_rotate > .5:
            image = tf.image.rot90(image, k=2) # rotate 180º
        elif p_rotate > .25:
            image = tf.image.rot90(image, k=1) # rotate 90º
    
    return image, label



if __name__ =='__main__':
    # bad practice, somehow python reads
    if os.path.isfile('ag_cfg.json'):
        print('aug using Kaggle config')
        with open('ag_cfg.json', 'r') as fp:
            ag_cfg = json.load(fp)
    else:
        from cfg import tr_cfg
        from cfg import ag_cfg
        print('aug using saved config')

    print(ag_cfg)

    
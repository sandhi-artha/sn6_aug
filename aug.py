# use bayesian
# only to import modules to train.py
# TODO: implement augmentations for both image and mask (use concat)
import tensorflow as tf
import tensorflow_addons as tfa
print(f'tensorflow_addons version: {tfa.__version__}')
# from tensorflow.keras import backend as K
import math




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
        stacked_image, size=[*size, IMAGE_CH+1])
    
    # for label, if you want [w,h,1] shape, use -1:
    return cropped_image[:,:,:IMAGE_CH], cropped_image[:,:,-1:]

def reduce_res(image, label):
    """either random crop or resize
    if using both, when p_reduce > .5 do resize, if lower do crop"""

    if tr_cfg['IS_RESIZE'] and not tr_cfg['IS_CROP']: # 100% resize
        image, label = resize_example(image, label)
    if tr_cfg['IS_CROP'] and not tr_cfg['IS_RESIZE']:   # 100% random crop
        image, label = random_crop(image, label)
    if tr_cfg['IS_RESIZE'] and tr_cfg['IS_CROP']:
        # 50-50 chance of resize and random crop
        p_reduce = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_reduce > .5:
            image, label = resize_example(image, label)
        else:
            image, label = random_crop(image, label)

    if not tr_cfg['IS_RESIZE'] and not tr_cfg['IS_CROP']:
        raise ValueError('must choose RESIZE or CROP or both')

    return image, label

# random bayessian control from https://www.kaggle.com/dimitreoliveira/flower-with-tpus-advanced-augmentations
def data_augment(image, label):
    
    # Flips
    if tr_cfg['IS_VFLIP']:
        p_vflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_vflip >= .5:
            # not using random_flip_up_down bcz a chance that image not flip with label
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)
    
    if tr_cfg['IS_HFLIP']:
        p_hflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_hflip >= .5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        
    # Rotates
    if tr_cfg['IS_ROT']:
        p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_rotate > .75:
            image = tf.image.rot90(image, k=3) # rotate 270º
            label = tf.image.rot90(label, k=3) # rotate 270º
        elif p_rotate > .5:
            image = tf.image.rot90(image, k=2) # rotate 180º
            label = tf.image.rot90(label, k=2) # rotate 180º
        elif p_rotate > .25:
            image = tf.image.rot90(image, k=1) # rotate 90º
            label = tf.image.rot90(label, k=1) # rotate 90º

    if tr_cfg['IS_FINE_ROT']:
        min_rad = tr_cfg['ROT_RANGE'][0] * math.pi / 180
        max_rad = tr_cfg['ROT_RANGE'][1] * math.pi / 180
        rot = tf.random.uniform([], min_rad, max_rad, dtype=tf.float32)
        image = tfa.image.rotate(image, rot)
        label = tfa.image.rotate(label, rot)


    # Shear
    if tr_cfg['IS_SHEAR_X']:
        """not optimal, shear from tfa works only for rgb
        """
        p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_shear > 0.5:
            min_shear = tr_cfg['SHEAR_RANGE'][0] * math.pi / 180
            max_shear = tr_cfg['SHEAR_RANGE'][1] * math.pi / 180
            shear = tf.random.uniform([], min_shear, max_shear, dtype=tf.float32)
            image = tf.image.grayscale_to_rgb(image)
            image = tfa.image.shear_x(image, shear, 0.0)
            image = tf.image.rgb_to_grayscale(image)

            label = tf.image.grayscale_to_rgb(label)
            label = tfa.image.shear_x(label, shear, 0.0)
            label = tf.image.rgb_to_grayscale(label)
           
    if tr_cfg['IS_SHEAR_Y']:
        p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_shear > 0.5:
            min_shear = tr_cfg['SHEAR_RANGE'][0] * math.pi / 180
            max_shear = tr_cfg['SHEAR_RANGE'][1] * math.pi / 180
            shear = tf.random.uniform([], min_shear, max_shear, dtype=tf.float32)
            image = tf.image.grayscale_to_rgb(image)
            image = tfa.image.shear_y(image, shear, 0.0)
            image = tf.image.rgb_to_grayscale(image)

            label = tf.image.grayscale_to_rgb(label)
            label = tfa.image.shear_y(label, shear, 0.0)
            label = tf.image.rgb_to_grayscale(label)


    # Filters
    if tr_cfg['IS_F_GAUS']:
        p_filter = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        sigma = tr_cfg['GAUS_SIGMA']

        if p_filter > .75:
            image = tfa.image.gaussian_filter2d(
                image, filter_shape=(2,2), sigma=sigma, padding='REFLECT')
        if p_filter > .5:
            image = tfa.image.gaussian_filter2d(
                image, filter_shape=(3,3), sigma=sigma, padding='REFLECT')
        elif p_filter > .25:
            image = tfa.image.gaussian_filter2d(
                image, filter_shape=(4,4), sigma=sigma, padding='REFLECT')

    if tr_cfg['IS_F_MED']:
        p_filter = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_filter > .75:
            image = tfa.image.median_filter2d(
                image, filter_shape=(2,2), padding='REFLECT')
        if p_filter > .5:
            image = tfa.image.median_filter2d(
                image, filter_shape=(3,3), padding='REFLECT')
        elif p_filter > .25:
            image = tfa.image.median_filter2d(
                image, filter_shape=(4,4), padding='REFLECT')

    

    return image, label


    # if tr_cfg['IS_LIGHT_FILT']:
    #     p_filter = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    #     if p_filter > .75:
    #         image = tfa.image.mean_filter2d(
    #             image, filter_shape=(2,2), padding='CONSTANT', constant_values=0.0)
    #     elif p_filter > .5:
    #         image = tfa.image.median_filter2d(
    #             image, filter_shape=(2,2), padding='CONSTANT', constant_values=0.0)
    #     elif p_filter > .25:
    #         image = tfa.image.gaussian_filter2d(
    #             image, filter_shape=(2,2), sigma=0.8, padding='CONSTANT', constant_values=0.0)
    
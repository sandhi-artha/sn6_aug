import tensorflow as tf
import math

### GLOBAL VARIABLES ###
IS_AUG_TF = 0
IS_OFF_AUG = 0

### REDUCE RESIZE FUNCTION ###
def resize(image, label):
    """center crop and resize to size IMAGE_RS"""
    image = tf.image.resize(image, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    label = tf.image.resize(label, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    return image, label

def pad_resize(image, label):
    """add paddings to create the original square image but resized"""
    image = tf.image.resize_with_pad(image, tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS'])
    label = tf.image.resize_with_pad(label, tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS'])
    return image, label

def random_crop(image, label):
    """crops a patch of size IMAGE_RS, preserves scale"""
    size = [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']]
    stacked_image = tf.concat([image, label], axis=-1)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[*size, IMAGE_CH+1])
    
    # for label, if you want [w,h,1] shape, use -1:
    return cropped_image[:,:,:IMAGE_CH], cropped_image[:,:,-1:]

def random_crop_resize(image, label):
    """does 2 ops
    1. random crop with random scale as percentage from minimum length
    2. resize the patch to size IMAGE_RS"""
    scale = tf.random.uniform([], 0.5, 1.0, dtype=tf.float32)
    # identify smallest edge
    image_shape = image.get_shape()  # so I can get dims
    if image_shape.ndims == 3:
        less_edge = tf.math.reduce_min(tf.shape(image)[:2])
    if image_shape.dims == 4:
        less_edge = tf.math.reduce_min(tf.shape(image)[1:3])

    # random crop
    target_crop = tf.cast((tf.cast(less_edge, tf.float32) * scale), tf.uint32)
    size = [target_crop, target_crop]
    stacked_image = tf.concat([image, label], axis=-1)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[*size, IMAGE_CH+1])

    # return resize the crop
    return resize(cropped_image[:,:,:IMAGE_CH], cropped_image[:,:,-1:])

def get_reduce_res_func(
    is_resize=0,
    is_pad_resize=1,
    is_crop=0,
    is_crop_resize=0
):
    """acts as a function selector and also updates the cfg
    returns a tf function that reduces resolution. does OneOf ops"""
    if (is_resize+is_pad_resize+is_crop+is_crop_resize)>1:
        raise ValueError('can only do 1 reduce resize ops')

    if is_resize:
        tr_cfg['REDUCE_RES'] = 'resize'
        return resize
    elif is_pad_resize:
        tr_cfg['REDUCE_RES'] = 'pad_resize'
        return pad_resize
    elif is_crop:
        tr_cfg['REDUCE_RES'] = 'crop'
        return random_crop
    elif is_crop_resize:
        tr_cfg['REDUCE_RES'] = 'crop_resize'
        return random_crop_resize
    else:
        raise ValueError('must provide reduce_res function')

def get_val_reduce_res_func(
    is_resize=0,
    is_pad_resize=1,
    is_crop=0,
    is_crop_resize=0
):
    """acts as a function selector and also updates the cfg
    returns a tf function that reduces resolution. does OneOf ops"""
    if (is_resize+is_pad_resize+is_crop+is_crop_resize)>1:
        raise ValueError('can only do 1 reduce resize ops')

    if is_resize:
        tr_cfg['VAL_REDUCE_RES'] = 'resize'
        return resize
    elif is_pad_resize:
        tr_cfg['VAL_REDUCE_RES'] = 'pad_resize'
        return pad_resize
    elif is_crop:
        tr_cfg['VAL_REDUCE_RES'] = 'crop'
        return random_crop
    elif is_crop_resize:
        tr_cfg['VAL_REDUCE_RES'] = 'crop_resize'
        return random_crop_resize
    else:
        raise ValueError('must provide reduce_res function')


### AUGMENTATION FUNCTION ###
def aug_tf(image, label):
    """maps augmentation from tf based on tr_cfg"""
    # Flips
    if tr_cfg['IS_HFLIP']:
        p_hflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_hflip >= .5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
    
    if tr_cfg['IS_VFLIP']:
        p_vflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_vflip >= .5:
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)
    
        
    # Rotates
    if tr_cfg['IS_ROT90']:
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

    return image, label


def update_aug_tf(
    is_hflip=0,
    is_vflip=0,
    is_rot90=0,
    is_fine_rot=0,
    is_shear_x=0,
    is_shear_y=0,
    rot_range=[-10,10],
    shear_range=[-10,10]
):
    """configure augmentations from tf"""
    tr_cfg['IS_HFLIP'] = is_hflip
    tr_cfg['IS_VFLIP'] = is_vflip
    tr_cfg['IS_ROT90'] = is_rot90
    tr_cfg['IS_FINE_ROT'] = is_fine_rot
    tr_cfg['IS_SHEAR_X'] = is_shear_x
    tr_cfg['IS_SHEAR_Y'] = is_shear_y
    tr_cfg['ROT_RANGE'] = rot_range
    tr_cfg['SHEAR_RANGE'] = shear_range
    
    # toggle the map function in dataloader
    if (is_hflip+is_vflip+is_rot90+is_fine_rot+is_shear_x+is_shear_y)>0:
        IS_AUG_TF = 1
    else:
        IS_AUG_TF = 0
        
def update_aug_off(
    is_elee=0,
    is_frost=0,
    is_gmap=0
):
    IS_OFF_AUG = 1
    if is_elee:
        tr_cfg['OFF_DS'] = 'elee'
        SUB_PATH_TRAIN = 'elee'
        SUB_PATH_VAL = 'elee'
    elif is_frost:
        tr_cfg['OFF_DS'] = 'frost'
        SUB_PATH_TRAIN = 'frost'
        SUB_PATH_VAL = 'frost'
    elif is_gmap:
        tr_cfg['OFF_DS'] = 'gmap'
        SUB_PATH_TRAIN = 'gmap'
        SUB_PATH_VAL = 'gmap'
    else:
        IS_OFF_AUG = 0
        SUB_PATH_TRAIN = ''
        SUB_PATH_VAL = ''







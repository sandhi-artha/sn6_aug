import tensorflow as tf
import tensorflow_addons as tfa
import math

### GLOBAL VARIABLES ###
IS_AUG_TF = 0

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
    scale = tf.random.uniform([], 0.7, 1.0, dtype=tf.float32)
    # identify smallest edge
    image_shape = image.get_shape()  # so I can get dims
    if image_shape.ndims == 3:
        less_edge = tf.math.reduce_min(tf.shape(image)[:2])
    if image_shape.ndims == 4:
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



class Resize():
    def __init__(self, target_res, image_ch, pad=False):
        """
        target_res: target resolution of square image
        image_ch: num of channels
        pad: if False, forces rectangular image to be square (distorted)
             if True, add paddings to create the original square image but resized"""
        self.target_res = target_res
        self.image_ch = image_ch
        if pad:
            self.resize_fun = tf.image.resize_with_pad
        else:
            self.resize_fun = tf.image.resize

    def aug(self, image, label):
        image = self.resize_fun(image, [self.target_res, self.target_res])
        label = self.resize_fun(label, [self.target_res, self.target_res])
        return image, label

class RandomCrop():
    def __init__(self, target_res, image_ch):
        """
        target_res: target resolution of square image
        image_ch: num of channels"""
        self.size = [target_res, target_res]
        self.image_ch = image_ch
    
    def aug(self, image, label):
        """crops a patch of size IMAGE_RS, preserves scale"""
        stacked_image = tf.concat([image, label], axis=-1)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[*self.size, self.image_ch+1])
        
        # for label, if you want [w,h,1] shape, use -1:
        return cropped_image[:,:,:self.image_ch], cropped_image[:,:,-1:]

class RandomCropResize():
    def __init__(self, target_res, image_ch):
        """
        target_res: target resolution of square image
        image_ch: num of channels"""
        self.target_res = target_res
        self.image_ch = image_ch

    def aug(self, image, label):
        """does 2 ops
        1. random crop with random scale as percentage from minimum length
        2. resize the patch to size IMAGE_RS
        """
        scale = tf.random.uniform([], 0.7, 1.0, dtype=tf.float32)
        
        # identify smallest edge
        image_shape = image.get_shape()
        if image_shape.ndims == 3:
            less_edge = tf.math.reduce_min(tf.shape(image)[:2])
        if image_shape.ndims == 4:
            less_edge = tf.math.reduce_min(tf.shape(image)[1:3])

        # random crop
        target_crop = tf.cast((tf.cast(less_edge, tf.float32) * scale), tf.uint32)
        size = [target_crop, target_crop]
        stacked_image = tf.concat([image, label], axis=-1)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[*size, self.image_ch+1])

        # return the resized crop
        image = tf.image.resize(
            cropped_image[:,:,:self.image_ch],
            [self.target_res, self.target_res])
        label = tf.image.resize(
            cropped_image[:,:,-1:],
            [self.target_res, self.target_res])

        return image, label









class HFlip():
    def aug(self, image, label):
        p_hflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_hflip >= .5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        return image, label

class VFlip():
    def aug(self, image, label):
        p_vflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_vflip >= .5:
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)
        return image, label

class Rot90():
    def aug(self, image, label):
        """applies either 90, 180 or 270 degree rotation"""
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
        return image, label

class FineRot():
    def __init__(self, min_rad, max_rad, reflect=False):
        """reflect: if True, invalid regions will be mirrored to mask the empty space"""
        self.min_rad = min_rad * math.pi / 180
        self.max_rad = max_rad * math.pi / 180
        if reflect:
            self.fill_mode = 'reflect'
            self.interpolation = 'bilinear'
        else:
            self.fill_mode = 'constant'
            self.interpolation = 'nearest'

    def aug(self, image, label):
        p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_rotate > .5:
            rot = tf.random.uniform([], self.min_rad, self.max_rad, dtype=tf.float32)
            image = tfa.image.rotate(
                image, rot, interpolation=self.interpolation, fill_mode=self.fill_mode)
            label = tfa.image.rotate(
                label, rot, interpolation=self.interpolation, fill_mode=self.fill_mode)
        return image, label

class Shear():
    def __init__(self, min_shear, max_shear, axis='x'):
        self.min_shear = min_shear * math.pi / 180
        self.max_shear = max_shear * math.pi / 180
        if axis=='x':
            self.shear_fun = tfa.image.shear_x
        if axis=='y':
            self.shear_fun = tfa.image.shear_y

    def aug(self, image, label):
        """not optimal, shear from tfa works only for rgb
        """
        p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_shear > 0.5:
            shear = tf.random.uniform([], self.min_shear, self.max_shear, dtype=tf.float32)
            image = tf.image.grayscale_to_rgb(image)
            image = self.shear_fun(image, shear, 0.0)
            image = tf.image.rgb_to_grayscale(image)

            label = tf.image.grayscale_to_rgb(label)
            label = self.shear_fun(label, shear, 0.0)
            label = tf.image.rgb_to_grayscale(label)
        return image, label


def get_aug_tf_func(cfg):
    """transformation list for tf ops"""
    transforms_list = []
    if cfg.IS_HFLIP:
        transforms_list.append(HFlip)
    if cfg.IS_VFLIP:
        transforms_list.append(VFlip)
    if cfg.IS_ROT90:
        transforms_list.append(Rot90)
    if cfg.IS_FINE_ROT:
        transforms_list.append(FineRot(
            min_rad=cfg.ROT_RANGE[0],
            max_rad=cfg.ROT_RANGE[1],
            reflect=cfg.ROT_REFLECT))
    if cfg.IS_SHEAR_X:
        transforms_list.append(Shear(
            min_rad=cfg.SHEAR_RANGE[0],
            max_rad=cfg.SHEAR_RANGE[1],
            axis='x'))
    if cfg.IS_SHEAR_Y:
        transforms_list.append(Shear(
            min_rad=cfg.SHEAR_RANGE[0],
            max_rad=cfg.SHEAR_RANGE[1],
            axis='y'))

    if len(transforms_list)>0:
        IS_AUG_TF = 1
    else:
        IS_AUG_TF = 0

    return transforms_list, IS_AUG_TF





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
        p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_rotate > .5:
            min_rad = tr_cfg['ROT_RANGE'][0] * math.pi / 180
            max_rad = tr_cfg['ROT_RANGE'][1] * math.pi / 180
            rot = tf.random.uniform([], min_rad, max_rad, dtype=tf.float32)
            if tr_cfg['ROT_REFLECT']:
                image = tfa.image.rotate(image, rot, interpolation='bilinear', fill_mode='reflect')
                label = tfa.image.rotate(label, rot, interpolation='bilinear', fill_mode='reflect')
            else:
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
    rot_reflect=0,
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
    tr_cfg['ROT_REFLECT'] = rot_reflect
    tr_cfg['SHEAR_RANGE'] = shear_range
    
    logs = []
    if is_hflip:
        logs.append('hflip')
    if is_vflip:
        logs.append('vflip')
    if is_rot90:
        logs.append('rot90')
    if is_fine_rot:
        logs.append(f'fine_rot: [{rot_range[0]},{rot_range[1]}]')
    if is_shear_x:
        logs.append(f'shear_x: [{shear_range[0]},{shear_range[1]}]')
    if is_shear_y:
        logs.append(f'shear_y: [{shear_range[0]},{shear_range[1]}]')
    print(f'active aug tf: {logs}')

    # toggle the map function in dataloader
    if len(logs)>0:
        IS_AUG_TF = 1
    else:
        IS_AUG_TF = 0
    
    return IS_AUG_TF

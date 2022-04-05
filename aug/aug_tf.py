import tensorflow as tf
import tensorflow_addons as tfa
import math

### TF REDUCE ###
class Resize():
    def __init__(self, target_res, image_ch):
        """forces rectangular image to be square (distorted)
            target_res: target resolution of square image
            image_ch: num of channels
        """
        self.target_res = target_res
        self.image_ch = image_ch

    def aug(self, image, label):
        image = tf.image.resize(image, [self.target_res, self.target_res])
        label = tf.image.resize(label, [self.target_res, self.target_res])
        return image, label

class PadResize():
    def __init__(self, target_res, image_ch):
        """add paddings to create the original square image but resized
            target_res: target resolution of square image
            image_ch: num of channels
        """
        self.target_res = target_res
        self.image_ch = image_ch

    def aug(self, image, label):
        image = tf.image.resize_with_pad(image, self.target_res, self.target_res)
        label = tf.image.resize_with_pad(label, self.target_res, self.target_res)
        return image, label

class RandomCrop():
    def __init__(self, target_res, image_ch, comb=True):
        """
        target_res: target resolution of square image
        image_ch: num of channels"""
        self.target_res = target_res
        self.image_ch = image_ch
        self.comb = comb
    
    def _pad_resize(self, image, label):
        image = tf.image.resize_with_pad(image, self.target_res, self.target_res)
        label = tf.image.resize_with_pad(label, self.target_res, self.target_res)
        return image, label
    
    def _random_crop(self, image, label):
        """crops a patch of size IMAGE_RS, preserves scale"""
        stacked_image = tf.concat([image, label], axis=-1)
        cropped_image = tf.image.random_crop(
            stacked_image, size=[self.target_res, self.target_res, self.image_ch+1])
        
        # for label, if you want [w,h,1] shape, use -1:
        return cropped_image[:,:,:self.image_ch], cropped_image[:,:,-1:]

    def aug(self, image, label):
        """if comb is True, assigns 50% chance of doing either
        random crop or pad_resize"""
        if self.comb:
            p_res = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        else:
            p_res = tf.constant(1.0)
        image, label = tf.cond(
            p_res>.5,
            lambda: self._random_crop(image, label),    # if true
            lambda: self._pad_resize(image, label)      # if false
        )
        return image, label

class RandomCropResize():
    def __init__(self, target_res, image_ch, comb=True):
        """
        target_res: target resolution of square image
        image_ch: num of channels"""
        self.target_res = target_res
        self.image_ch = image_ch
        self.comb = comb

    def _pad_resize(self, image, label):
        image = tf.image.resize_with_pad(image, self.target_res, self.target_res)
        label = tf.image.resize_with_pad(label, self.target_res, self.target_res)
        return image, label

    def _random_crop_resize(self, image, label):
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
    
    def aug(self, image, label):
        """if comb is True, assigns 50% chance of doing either
        random crop or pad_resize"""
        if self.comb:
            p_res = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        else:
            p_res = tf.constant(1.0)
        image, label = tf.cond(
            p_res>.5,
            lambda: self._random_crop_resize(image, label),     # if true
            lambda: self._pad_resize(image, label)              # if false
        )
        return image, label


def get_reduce_fun(method, cfg):
    """returns a reduce function"""
    image_ch = len(cfg.SAR_CH)
    target_res = cfg.IMAGE_RS
    comb = cfg.COMB_REDUCE
    if method=='resize':
        return Resize(target_res, image_ch)

    elif method=='pad_resize':
        return PadResize(target_res, image_ch)
    
    elif method=='random_crop':
        return RandomCrop(target_res, image_ch, comb=comb)
    
    elif method=='random_crop_resize':
        return RandomCropResize(target_res, image_ch, comb=comb)
    else:
        raise ValueError(f'"{method}" reduce method not found')
        



### TF AUG ###
class HFlip():
    def aug(image, label):
        p_hflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_hflip >= .5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        return image, label

class VFlip():
    def aug(image, label):
        p_vflip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_vflip >= .5:
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)
        return image, label

class Rot90():
    def aug(image, label):
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

class AugTF():
    def __init__(self, cfg):
        """augmentation list for tf ops"""
        self.aug_list = []
        if cfg.IS_HFLIP:
            self.aug_list.append(HFlip)
        if cfg.IS_VFLIP:
            self.aug_list.append(VFlip)
        if cfg.IS_ROT90:
            self.aug_list.append(Rot90)
        if cfg.IS_FINE_ROT:
            self.aug_list.append(FineRot(
                min_rad=cfg.ROT_RANGE[0],
                max_rad=cfg.ROT_RANGE[1],
                reflect=cfg.ROT_REFLECT))
        if cfg.IS_SHEAR_X:
            self.aug_list.append(Shear(
                min_shear=cfg.SHEAR_RANGE[0],
                max_shear=cfg.SHEAR_RANGE[1],
                axis='x'))
        if cfg.IS_SHEAR_Y:
            self.aug_list.append(Shear(
                min_shear=cfg.SHEAR_RANGE[0],
                max_shear=cfg.SHEAR_RANGE[1],
                axis='y'))

        self.IS_AUG = len(self.aug_list)>0

    def transform(self, image, label):
        """perform the chain augmentations"""
        for aug in self.aug_list:
            image, label = aug.aug(image, label)
        
        return image, label

def update_aug_tf(
    cfg,
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
    cfg['IS_HFLIP'] = is_hflip
    cfg['IS_VFLIP'] = is_vflip
    cfg['IS_ROT90'] = is_rot90
    cfg['IS_FINE_ROT'] = is_fine_rot
    cfg['IS_SHEAR_X'] = is_shear_x
    cfg['IS_SHEAR_Y'] = is_shear_y
    cfg['ROT_RANGE'] = rot_range
    cfg['ROT_REFLECT'] = rot_reflect
    cfg['SHEAR_RANGE'] = shear_range
    
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
    return cfg
import tensorflow as tf
import tensorflow_addons as tfa
import math



### TF REDUCE ###
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


def get_reduce_fun(method, cfg):
    """returns a reduce function"""
    image_ch = len(cfg.SAR_CH)
    target_res = cfg.IMAGE_RS
    if method=='resize':
        return Resize(target_res, image_ch, pad=False)

    if method=='pad_resize':
        return Resize(target_res, image_ch, pad=True)
    
    if method=='random_crop':
        return RandomCrop(target_res, image_ch)
    
    if method=='random_crop_resize':
        return RandomCropResize(target_res, image_ch)
        



### TF AUG ###
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
                min_rad=cfg.SHEAR_RANGE[0],
                max_rad=cfg.SHEAR_RANGE[1],
                axis='x'))
        if cfg.IS_SHEAR_Y:
            self.aug_list.append(Shear(
                min_rad=cfg.SHEAR_RANGE[0],
                max_rad=cfg.SHEAR_RANGE[1],
                axis='y'))

        self.IS_AUG = len(self.aug_list)>0

    def transform(self, image, label):
        """perform the chain augmentations"""
        for aug in self.aug_list:
            image, label = aug.aug(image, label)
        
        return image, label













def get_aug_tf_fun(self, cfg):
    """transformation list for tf ops"""
    aug_list = []
    if cfg.IS_HFLIP:
        aug_list.append(HFlip)
    if cfg.IS_VFLIP:
        aug_list.append(VFlip)
    if cfg.IS_ROT90:
        aug_list.append(Rot90)
    if cfg.IS_FINE_ROT:
        aug_list.append(FineRot(
            min_rad=cfg.ROT_RANGE[0],
            max_rad=cfg.ROT_RANGE[1],
            reflect=cfg.ROT_REFLECT))
    if cfg.IS_SHEAR_X:
        aug_list.append(Shear(
            min_rad=cfg.SHEAR_RANGE[0],
            max_rad=cfg.SHEAR_RANGE[1],
            axis='x'))
    if cfg.IS_SHEAR_Y:
        aug_list.append(Shear(
            min_rad=cfg.SHEAR_RANGE[0],
            max_rad=cfg.SHEAR_RANGE[1],
            axis='y'))

    if len(aug_list)>0:
        IS_AUG_TF = 1
    else:
        IS_AUG_TF = 0

    return aug_list, IS_AUG_TF

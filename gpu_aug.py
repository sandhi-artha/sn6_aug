import tensorflow as tf
import albumentations as A
import cv2
from cfg import tr_cfg

IMAGE_CH = len(tr_cfg['SAR_CH'])

def resize_example(image, label, meta):
    """image must be type float. TPUable"""
    h = meta['data_idx'][1] - meta['data_idx'][0]
    w = meta['data_idx'][3] - meta['data_idx'][2]
    # reshape here is needed bcz from parse_tensor, image and label don't have shape
    image = tf.reshape(image, [h, w, IMAGE_CH])
    label = tf.reshape(label, [h, w, 1])

    image = tf.image.resize(image, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    label = tf.image.resize(label, [tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS']])
    return image, label

def crop_example(image, label, meta):
    """image and label must be same dtype. TPUable"""
    h = meta['data_idx'][1] - meta['data_idx'][0]
    w = meta['data_idx'][3] - meta['data_idx'][2]

    # reshape here is needed bcz from parse_tensor, image and label don't have shape
    image = tf.reshape(image, [h, w, IMAGE_CH])
    label = tf.reshape(label, [h, w, 1])

    stacked_image = tf.concat([image, label], axis=-1)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[tr_cfg['IMAGE_RS'], tr_cfg['IMAGE_RS'], IMAGE_CH+1])
    return cropped_image[:,:,:IMAGE_CH], cropped_image[:,:,-1:]
    
def reduce_res(image, label, meta):
    """either random crop or resize
    if using both, when p_reduce > .5 do resize, if lower do crop"""

    if tr_cfg['IS_RESIZE'] and not tr_cfg['IS_CROP']: # 100% resize
        image, label = resize_example(image, label, meta)
    if tr_cfg['IS_CROP'] and not tr_cfg['IS_RESIZE']:   # 100% random crop
        image, label = crop_example(image, label, meta)
    if tr_cfg['IS_RESIZE'] and tr_cfg['IS_CROP']:
        # 50-50 chance of resize and random crop
        p_reduce = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if p_reduce > .5:
            image, label = resize_example(image, label, meta)
        else:
            image, label = crop_example(image, label, meta)

    if not tr_cfg['IS_RESIZE'] and not tr_cfg['IS_CROP']:
        raise ValueError('must choose RESIZE or CROP or both')

    return image, label, meta


def get_transform():
    """transformation list for albumentations"""
    transforms = []

    # spatial transformation
    if tr_cfg['IS_HFLIP']:
        transforms.append(A.transforms.HorizontalFlip())
    if tr_cfg['IS_VFLIP']:
        transforms.append(A.transforms.VerticalFlip())
    if tr_cfg['IS_ROT90']:
        transforms.append(A.augmentations.geometric.RandomRotate90())
    if tr_cfg['IS_FINE_ROT']:
        transforms.append(A.augmentations.geometric.rotate.Rotate(
            limit=tr_cfg['ROT_RANGE'], border_mode=cv2.BORDER_REFLECT
        ))

    # pixel transformation
    if tr_cfg['IS_GAUS_NOISE']:
        transforms.append(A.augmentations.transforms.GaussNoise(
            var_limit=tr_cfg['GAUS_NOISE_VAR']
        ))

    if tr_cfg['IS_MOT_BLUR']:
        transforms.append(A.transforms.MotionBlur(blur_limit=12, p=0.5))
    
    # elee filter
    # frost filter
    # gmap filter

    return A.Compose(transforms)

# TRANSFORMS = get_transform()

def albu_transform(image, label):
    """spatial transformation done on both image and label
    pixel trans only on image"""
    transform_data = TRANSFORMS(image=image, mask=label)
    return transform_data['image'], transform_data['mask']

def aug_albu(image, label):
    """wrapper to feed numpy format data to TRANSFORMS"""
    aug_img, aug_mask = tf.numpy_function(
        func=albu_transform, inp=[image, label], Tout=[tf.float32, tf.float32])
    return aug_img, aug_mask
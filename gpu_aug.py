import tensorflow as tf
import albumentations as A
import cv2



# methods for reducing resolution size
def get_reduce_res_transforms(
    resize=0,
    pad_resize=1,
    pad_reflect_resize=0,
    crop=0,
    resize_crop=0
):

    reduce_res_list = []
    if resize:
        reduce_res_list.append(
            A.augmentations.geometric.resize.Resize(320,320,interpolation=INTER)
        )
    if pad_resize:
        reduce_res_list.append(
            A.Sequential([
                A.augmentations.transforms.PadIfNeeded(
                    900,900, border_mode=cv2.BORDER_CONSTANT, value=0.0, mask_value=0.0),
                A.augmentations.geometric.resize.Resize(320,320,interpolation=INTER)])
        )
    if pad_reflect_resize:
        reduce_res_list.append(
            A.Sequential([
                A.augmentations.transforms.PadIfNeeded(
                    900,900, border_mode=cv2.BORDER_REFLECT),
                A.augmentations.geometric.resize.Resize(320,320,interpolation=INTER)])
        )

    if crop:
        reduce_res_list.append(
            A.augmentations.crops.transforms.RandomCrop(320,320)
        )
    if resize_crop:
        reduce_res_list.append(
            A.augmentations.crops.transforms.RandomResizedCrop(
                320,320,scale=(0.3,0.7),ratio=(1.0,1.0),interpolation=INTER,p=1.0)
        )

    reduce_res = [A.OneOf(reduce_res_list, p=1.0)]
    return A.Compose(reduce_res)

def reduce_res_transform(image, label):
    """performs transformation based on global
    albu transforms that reduces resolution"""
    transform_data = REDUCE_RES_TRANSFORMS(image=image, mask=label)
    return transform_data['image'], transform_data['mask']

def reduce_res_albu(image, label):
    """wrapper to feed numpy format data to TRANSFORMS"""
    aug_img, aug_mask = tf.numpy_function(
        func=reduce_res_transform, inp=[image, label], Tout=[tf.float32, tf.float32])
    return aug_img, aug_mask


def val_reduce_res_transform(image, label):
    """performs transformation based on global
    albu transforms that reduces resolution"""
    transform_data = VAL_REDUCE_RES_TRANSFORMS(image=image, mask=label)
    return transform_data['image'], transform_data['mask']

def val_reduce_res_albu(image, label):
    """wrapper to feed numpy format data to TRANSFORMS"""
    aug_img, aug_mask = tf.numpy_function(
        func=reduce_res_transform, inp=[image, label], Tout=[tf.float32, tf.float32])
    return aug_img, aug_mask



def get_aug_transforms(
    hflip=0,
    vflip=0,
    rot90=0,
    fine_rot=0, rot_range=[-10,10],
    mot_blur=0
):
    """transformation list for augmentations in albu"""
    transforms = []

    # spatial transformation
    if hflip:
        transforms.append(A.transforms.HorizontalFlip())
    if vflip:
        transforms.append(A.transforms.VerticalFlip())
    if rot90:
        transforms.append(A.augmentations.geometric.RandomRotate90())
    if fine_rot:
        transforms.append(A.augmentations.geometric.rotate.Rotate(
            limit=rot_range, border_mode=BORDER))

    # pixel transformation
    # if tr_cfg['IS_GAUS_NOISE']:  # still produces whites
    #     transforms.append(A.augmentations.transforms.GaussNoise(
    #         var_limit=tr_cfg['GAUS_NOISE_VAR']
    #     ))

    if mot_blur:
        transforms.append(A.transforms.MotionBlur(blur_limit=12))
    
    # elee filter
    # frost filter
    # gmap filter

    return A.Compose(transforms)

def aug_transform(image, label):
    """spatial transformation done on both image and label
    pixel trans only on image"""
    transform_data = TRANSFORMS(image=image, mask=label)
    return transform_data['image'], transform_data['mask']

def aug_albu(image, label):
    """wrapper to feed numpy format data to TRANSFORMS"""
    aug_img, aug_mask = tf.numpy_function(
        func=aug_transform, inp=[image, label], Tout=[tf.float32, tf.float32])
    return aug_img, aug_mask



# Global Variables
INTER = cv2.INTER_NEAREST
BORDER = cv2.BORDER_REFLECT
TRANSFORMS = get_aug_transforms()
REDUCE_RES_TRANSFORMS = get_reduce_res_transforms()  # should use from tr_cfg
VAL_REDUCE_RES_TRANSFORMS = get_reduce_res_transforms()  # use default (only pad_resize)
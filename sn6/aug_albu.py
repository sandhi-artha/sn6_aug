import albumentations as A
import albumentations.augmentations.transforms as AAT
import tensorflow as tf

### AUGMENTATION TRANSFORMS ###
def get_aug_albu_func(cfg):
    """transformation list for augmentations in albu"""
    one_of_transforms = []
    transforms_list = []

    # pixel transformation
    if cfg.IS_MOT_BLUR:
        one_of_transforms.append(AAT.MotionBlur(blur_limit=12))
    if cfg.IS_SHARPEN:
        one_of_transforms.append(
            AAT.Sharpen(alpha=(0.1,0.4), lightness=(.9,1.0))
        )
    if cfg.IS_CLAHE:
        one_of_transforms.append(
            A.Sequential([
            AAT.FromFloat(dtype='uint8', max_value=255),
            AAT.CLAHE(clip_limit=4.0),
            AAT.ToFloat(max_value=255)])
        )
    if cfg.IS_GAUS_NOISE:
        one_of_transforms.append(AAT.GaussNoise(var_limit=.01))
    if cfg.IS_SPECKLE_NOISE:
        one_of_transforms.append(
            AAT.MultiplicativeNoise(multiplier=(0.8,1.2), elementwise=True)
        )

    if len(one_of_transforms) > 0:
        transforms_list.append(A.OneOf(one_of_transforms))
    
    # geometric transformation, done after filtering
    if cfg.IS_COARSE_DO:
        transforms_list.append(
            A.augmentations.CoarseDropout(max_holes=6, max_height=40,
                max_width=40, min_holes=2, min_height=30, min_width=30,
                fill_value=0.0, mask_fill_value=0)
        )
    
    if len(transforms_list)>0:
        IS_AUG_ALBU = 1
    else:
        IS_AUG_ALBU = 0

    return A.Compose(transforms_list), IS_AUG_ALBU


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
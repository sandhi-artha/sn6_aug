import albumentations as A
import albumentations.augmentations.transforms as AAT



### AUGMENTATION TRANSFORMS ###
def get_aug_albu_func(
    is_mot_blur=0,
    is_sharpen=0,
    is_clahe=0,
    is_gaus_noise=0,
    is_speckle_noise=0,
    is_coarse_do=0,
):
    """transformation list for augmentations in albu"""
    # update cfg
    tr_cfg['IS_MOT_BLUR'] = is_mot_blur
    tr_cfg['IS_SHARPEN'] = is_sharpen
    tr_cfg['IS_CLAHE'] = is_clahe
    tr_cfg['IS_GAUS_NOISE'] = is_gaus_noise
    tr_cfg['IS_SPECKLE_NOISE'] = is_speckle_noise
    tr_cfg['IS_COARSE_DO'] = is_coarse_do


    one_of_transforms = []
    transforms_list = []

    # pixel transformation
    if is_mot_blur:
        one_of_transforms.append(AAT.MotionBlur(blur_limit=12))
    if is_sharpen:
        one_of_transforms.append(
            AAT.Sharpen(alpha=(0.1,0.4), lightness=(.9,1.0))
        )
    if is_clahe:
        one_of_transforms.append(
            A.Sequential([
            AAT.FromFloat(dtype='uint8', max_value=255),
            AAT.CLAHE(clip_limit=4.0),
            AAT.ToFloat(max_value=255)])
        )
    if is_gaus_noise:
        one_of_transforms.append(AAT.GaussNoise(var_limit=.01))
    if is_speckle_noise:
        one_of_transforms.append(
            AAT.MultiplicativeNoise(multiplier=(0.8,1.2), elementwise=True)
        )
    

    if len(one_of_transforms) > 0:
        transforms_list.append(A.OneOf(one_of_transforms))
    
    # geometric transformation, done after filtering
    if is_coarse_do:
        transforms_list.append(
            A.augmentations.CoarseDropout(max_holes=10, max_height=20,
                max_width=10, min_holes=1, min_height=5, min_width=5,
                fill_value=0.9, mask_fill_value=0)
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
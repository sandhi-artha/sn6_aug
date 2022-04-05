import albumentations as A
import albumentations.augmentations.transforms as AAT
import tensorflow as tf

### AUGMENTATION TRANSFORMS ###
class AugAlbu():
    def __init__(self, cfg):
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
        
        self.IS_AUG = len(transforms_list)>0
        self.composer = A.Compose(transforms_list)
    
    def transform(self, image, label, h, w, c):
        """wrapper to feed numpy format data to TRANSFORMS"""
        def _aug_transform(image, label):
            """spatial transformation done on both image and label
            pixel trans only on image"""
            transform_data = self.composer(image=image, mask=label)
            return transform_data['image'], transform_data['mask']

        aug_img, aug_mask = tf.numpy_function(
            func=_aug_transform, inp=[image, label], Tout=[tf.float32, tf.float32])

        # add back shape bcz numpy_function makes it unknown
        image = tf.reshape(aug_img, [h,w,c]) 
        label = tf.reshape(aug_mask, [h,w,1])
        return image, label
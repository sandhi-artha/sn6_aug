import math
import tensorflow as tf

tr_cfg = {
    # dataset cfg
    'IMAGE_RS'      : 320,          # target resolution
    'TRAIN_PATH'    : '../../image_folder/sn6-crop',    # kaggle ds name
    'TRAIN_SPLITS'  : ['fold0', 'fold1'],               # folds used for training
    'VAL_PATH'      : '../../image_folder/sn6-crop',    # if None, don't validate
    'VAL_SPLITS'    : ['fold4'],    # folds used for validation
    'SAR_CH'        : [1],          # HH=1, HV=2, VH=3, VV=4. use None to read all channel
    'ORIENT'        : 1,
    
    # model cfg
    'BACKBONE'      : 'effb4',      # 'effb4', 'res50'
    'ARC'           : 'fpn',        # 'unet', 'fpn'
    'WEIGHT'        : None,         # 'imagenet', 'pre-trained from:..', None
    
    # training cfg
    'SEED'          : 17,
    'BATCH_SIZE'    : 8,
    'SHUFFLE_BUFFER': 150,
    'EPOCHS'        : 60,
    'LF'            : 'dice',       # 'bce', 'jaccard_distance', 'focal', 'giou'
    'L_RATE'        : 32e-4,        # 32e-4, 4e-4, 5e-5
    'IS_CV'         : 0,            # cross validation
    'IS_3_FOLD'     : 0,            # do same training 3x to get an average value
    'IS_CB_ES'      : 0,            # early stopping
    'IS_CB_LRS'     : 1,            # learning rate scheduler, if false uses lr_ramp
    
    # reduce method: 'resize', 'pad_resize', 'random_crop', 'random_crop_resize'
    'REDUCE_RES'    : 'pad_resize',
    'COMB_REDUCE'   : True, # when using rand_crop or rand_crop_resize, randomize reduce method with pad_resize
    'VAL_REDUCE_RES': 'pad_resize',

    # spatial transformations
    'IS_HFLIP'      : 0,
    'IS_VFLIP'      : 0,
    'IS_ROT90'      : 0,
    'IS_FINE_ROT'   : 0,
    'IS_SHEAR_X'    : 0,
    'IS_SHEAR_Y'    : 0,

    # aug magnitude
    'ROT_REFLECT'   : 0,
    'ROT_RANGE'     : [-10, 10],
    'SHEAR_RANGE'   : [-10, 10],

    # pixel transformations
    'IS_MOT_BLUR'   : 0,
    'IS_SHARPEN'    : 0,
    'IS_CLAHE'      : 0,
    'IS_GAUS_NOISE' : 0,
    'IS_SPECKLE_NOISE': 0,
    'IS_COARSE_DO'  : 0,

    # offline augs
    'OFF_DS'        : '',   # 'elee', 'frost', 'gmap' NOT ADOPTED YET
    'OFF_AUG_PATH'  : '',
}


ev_cfg = {
    'run_path'      : '',
    'base_dir'      : '../dataset/sn6_aug/val',  # where raster and vector dir are located
    'save_dir'      : '../dataset/sn6_aug',
    'chart'         : 'tp',     # 'tp','recall'
}

if not tr_cfg['IS_CB_LRS']:
    LR_START = tr_cfg['L_RATE'] # 1e-7 # 1e-6, 1e-7, 1e-8
    LR_MIN = 3e-6  # 3e-6, 3e-7, 3e-8
    LR_MAX = tr_cfg['L_RATE']
    LR_RAMPUP_EPOCHS = 0  # 5
    LR_SUSTAIN_EPOCHS = 0  # 2
    N_CYCLES = .5
    print(f'Learning rate schedule: {LR_START} to {LR_MAX} to {LR_MIN}')

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        progress = (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) / (tr_cfg['EPOCHS'] - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)
        lr = LR_MAX * (0.5 * (1.0 + tf.math.cos(math.pi * N_CYCLES * 2.0 * progress)))
        if LR_MIN is not None:
            lr = tf.math.maximum(LR_MIN, lr)
            
    return lr
